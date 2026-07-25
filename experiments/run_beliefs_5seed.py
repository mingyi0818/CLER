"""Run MovieLens Beliefs experiments with 5 seeds and report mean±std.

Methods: BPR, Multimodal CF, CLER, CV-CLER (=ARFusion collab), ARFusion-Rec
Seeds: 42, 123, 456, 789, 2024
Save: experiments/results/beliefs_5seed_results.json

每完成一个种子即保存一次, 屏幕实时显示进度。
CV-CLER = ARFusionModel(score_mode="collab", pbd_weight=0, fuse_weight=0)
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    train_model, evaluate_topk,
    RecData, TrainConfig, MFModel, MultimodalCFModel, CLERModel,
    BPRDataset, bpr_loss, set_seed,
)
from arfusion_recommender.arfusion_model import (
    ARFusionModel, warm_start_from_cler,
)
from load_movielens_beliefs import load_movielens_beliefs

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_PATH = ROOT / "experiments" / "results" / "beliefs_5seed_results.json"
PROGRESS_LOG = ROOT / "experiments" / "results" / "beliefs_5seed_log.txt"


def log(msg: str):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def save_incremental(all_results: dict, completed_seeds: list, summary: dict = None):
    """每完成一个种子即保存, 防止长时实验数据丢失。"""
    output = {
        "experiment": "MovieLens Beliefs 5-seed",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": SEEDS,
        "completed_seeds": completed_seeds,
        "per_seed_results": all_results,
    }
    if summary:
        output["summary"] = summary
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"[SAVED] {len(completed_seeds)} seeds completed -> {RESULTS_PATH}")


def train_cler_warmstart(data, cfg, device):
    """训练 CLER 用于热启动 CV-CLER 和 ARFusion。"""
    cler = CLERModel(data.n_users, data.n_items, cfg.embed_dim, ui_temperature=cfg.ui_temperature)
    cler = cler.to(device)
    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(cler.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, patience_ctr, best_state = float("inf"), 0, None
    for epoch in range(cfg.max_epochs):
        cler.train()
        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            opt.zero_grad()
            loss = bpr_loss(cler.score(users, pos), cler.score(users, neg))
            loss = loss + cfg.cl_weight * cler.contrastive_loss(users, pos)
            loss.backward()
            opt.step()
        cler.eval()
        vl = []
        with torch.no_grad():
            for batch in val_loader:
                users, pos, neg = [x.to(device) for x in batch]
                vl.append(bpr_loss(cler.score(users, pos), cler.score(users, neg)).item())
        v = np.mean(vl)
        if v < best_val:
            best_val, patience_ctr = v, 0
            best_state = {k: v_cpu.cpu().clone() for k, v_cpu in cler.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break
    if best_state:
        cler.load_state_dict(best_state)
        cler = cler.to(device)
    return cler


def run_single_seed(seed, data, cfg):
    """单种子实验: BPR -> MMCF -> CLER -> CV-CLER -> ARFusion"""
    set_seed(seed)
    feat_dim = data.user_features.shape[1]
    results = {}
    device = torch.device(cfg.device)

    # --- BPR ---
    log(f"  [Seed {seed}] Training BPR ...")
    t0 = time.time()
    bpr = MFModel(data.n_users, data.n_items, cfg.embed_dim)
    trained, _ = train_model(bpr, data, cfg, "BPR")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["BPR"] = m
    log(f"  [Seed {seed}] BPR: NDCG@10={m['ndcg@10']:.4f} P@10={m['precision@10']:.4f} ({time.time()-t0:.1f}s)")

    # --- Multimodal CF ---
    log(f"  [Seed {seed}] Training Multimodal CF ...")
    t0 = time.time()
    mmcf = MultimodalCFModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim)
    trained, _ = train_model(mmcf, data, cfg, "Multimodal CF")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["Multimodal CF"] = m
    log(f"  [Seed {seed}] Multimodal CF: NDCG@10={m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")

    # --- CLER (用于热启动) ---
    log(f"  [Seed {seed}] Training CLER (warm-start base) ...")
    t0 = time.time()
    cler = train_cler_warmstart(data, cfg, device)
    m = evaluate_topk(cler, data, k=10, device=cfg.device)
    results["CLER"] = m
    log(f"  [Seed {seed}] CLER: NDCG@10={m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")

    torch.cuda.empty_cache()

    # --- CV-CLER (ARFusion collab, pbd=0, fuse=0) ---
    log(f"  [Seed {seed}] Training CV-CLER (ARFusion collab) ...")
    t0 = time.time()
    cvcler_cfg = TrainConfig(**{
        **cfg.__dict__,
        "pbd_weight": 0.0,
        "fuse_weight": 0.0,
    })
    cvcler = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode="collab",
    )
    warm_start_from_cler(cvcler, cler)
    trained, _ = train_model(cvcler, data, cvcler_cfg, "CV-CLER")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["CV-CLER"] = m
    log(f"  [Seed {seed}] CV-CLER: NDCG@10={m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")

    torch.cuda.empty_cache()

    # --- ARFusion-Rec (dual mode, with L_fuse) ---
    log(f"  [Seed {seed}] Training ARFusion-Rec (dual mode) ...")
    t0 = time.time()
    arf = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode=cfg.score_mode,
    )
    warm_start_from_cler(arf, cler)
    trained, _ = train_model(arf, data, cfg, "ARFusion-Rec")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["ARFusion-Rec"] = m
    log(f"  [Seed {seed}] ARFusion-Rec: NDCG@10={m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")

    return results


def main():
    # 清空日志
    with open(PROGRESS_LOG, "w", encoding="utf-8") as f:
        f.write(f"Beliefs 5-seed experiment started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log("=" * 60)
    log("MovieLens Beliefs 5-seed Experiment")
    log("=" * 60)

    # 加载数据
    log("Loading MovieLens Beliefs dataset ...")
    t0 = time.time()
    data = load_movielens_beliefs(use_cache=True)
    log(f"Dataset loaded in {time.time()-t0:.1f}s")
    log(f"  Users: {data.n_users:,}, Items: {data.n_items:,}")
    log(f"  Train/Val/Test: {len(data.train):,}/{len(data.val):,}/{len(data.test):,}")
    log(f"  Test users: {len(data.test_relevant):,}")
    log(f"  Feat dim: {data.user_features.shape[1]}")

    cfg = STRAVL_BEST
    log(f"Config: embed_dim={cfg.embed_dim}, lr={cfg.lr}, batch_size={cfg.batch_size}")
    log(f"  cl_weight={cfg.cl_weight}, cv_weight={cfg.cv_weight}")
    log(f"  max_epochs={cfg.max_epochs}, patience={cfg.patience}")
    log(f"  device={cfg.device}")

    all_results = {}
    completed_seeds = []

    # 检查是否有已完成的结果可以恢复
    if RESULTS_PATH.exists():
        try:
            existing = json.load(open(RESULTS_PATH, encoding="utf-8"))
            all_results = existing.get("per_seed_results", {})
            completed_seeds = [int(s) for s in existing.get("completed_seeds", [])]
            log(f"Resuming: {len(completed_seeds)} seeds already done: {sorted(completed_seeds)}")
        except Exception as e:
            log(f"Failed to load existing results: {e}")

    for seed in SEEDS:
        if seed in completed_seeds:
            log(f"\n{'='*60}\nSeed {seed} (already done, skipping)\n{'='*60}")
            continue
        log(f"\n{'='*60}")
        log(f"Seed {seed} START")
        log(f"{'='*60}")
        t0 = time.time()
        try:
            seed_results = run_single_seed(seed, data, cfg)
            elapsed = time.time() - t0
            log(f"Seed {seed} DONE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
            for method, metrics in seed_results.items():
                log(f"  {method}: NDCG@10={metrics['ndcg@10']:.4f} P@10={metrics['precision@10']:.4f} R@10={metrics['recall@10']:.4f}")
            all_results[str(seed)] = seed_results
            completed_seeds.append(seed)
            completed_seeds.sort()
            save_incremental(all_results, completed_seeds)
        except Exception as e:
            log(f"Seed {seed} FAILED: {e}")
            import traceback
            log(traceback.format_exc())
            save_incremental(all_results, completed_seeds)

    # 计算均值±标准差
    log(f"\n{'='*60}")
    log("Computing summary statistics ...")
    log(f"{'='*60}")
    methods = ["BPR", "Multimodal CF", "CLER", "CV-CLER", "ARFusion-Rec"]
    summary = {}
    for method in methods:
        ndcgs = [all_results[str(s)][method]["ndcg@10"] for s in completed_seeds if str(s) in all_results and method in all_results[str(s)]]
        p10s = [all_results[str(s)][method]["precision@10"] for s in completed_seeds if str(s) in all_results and method in all_results[str(s)]]
        r10s = [all_results[str(s)][method]["recall@10"] for s in completed_seeds if str(s) in all_results and method in all_results[str(s)]]
        if ndcgs:
            summary[method] = {
                "ndcg@10_mean": float(np.mean(ndcgs)),
                "ndcg@10_std": float(np.std(ndcgs)),
                "precision@10_mean": float(np.mean(p10s)),
                "precision@10_std": float(np.std(p10s)),
                "recall@10_mean": float(np.mean(r10s)),
                "recall@10_std": float(np.std(r10s)),
                "n_seeds": len(ndcgs),
                "seeds": {str(s): all_results[str(s)][method] for s in completed_seeds if str(s) in all_results and method in all_results[str(s)]},
            }
            log(f"{method}: NDCG@10={np.mean(ndcgs):.4f}±{np.std(ndcgs):.4f} (n={len(ndcgs)})")

    save_incremental(all_results, completed_seeds, summary)
    log(f"\nExperiment complete! Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
