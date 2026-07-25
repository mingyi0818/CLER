"""Run CV-CLER 5-seed experiment (ARFusion with pbd_weight=0, fuse_weight=0).

CV-CLER = ARFusionModel(score_mode="collab") with:
  - L_BPR + cl_weight * L_UI + cv_weight * L_CV
  - NO L_PBD (pbd_weight=0)
  - NO L_fuse (fuse_weight=0)
  - Graph propagation enabled (same as ARFusion-Rec for fair comparison)

This isolates the contribution of cross-view alignment + score decoupling
from the gated fusion and profile distillation components.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    load_stravl_data, train_model, evaluate_topk,
    TrainConfig, CLERModel, BPRDataset, bpr_loss, set_seed,
)
from arfusion_recommender.arfusion_model import ARFusionModel, warm_start_from_cler
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_PATH = ROOT / "results" / "cvcler_5seed_results.json"


def run_cvcler_single_seed(seed, data, cfg):
    """Train CLER warm-start, then CV-CLER (ARFusion without PBD/fuse)."""
    set_seed(seed)
    feat_dim = data.user_features.shape[1]
    device = torch.device(cfg.device)

    # 1. Train CLER for warm-start (same as ARFusion pipeline)
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

    torch.cuda.empty_cache()

    # 2. CV-CLER = ARFusion with pbd_weight=0, fuse_weight=0
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
    print(f"  [Seed {seed}] CV-CLER: NDCG@10={m['ndcg@10']:.4f}", flush=True)
    return m


def save_incremental(all_results, seeds_done):
    methods = ["CV-CLER"]
    summary = {}
    for method in methods:
        ndcgs = [all_results[str(s)][method]["ndcg@10"] for s in seeds_done]
        p10s = [all_results[str(s)][method]["precision@10"] for s in seeds_done]
        r10s = [all_results[str(s)][method]["recall@10"] for s in seeds_done]
        summary[method] = {
            "ndcg@10_mean": float(np.mean(ndcgs)),
            "ndcg@10_std": float(np.std(ndcgs)),
            "precision@10_mean": float(np.mean(p10s)),
            "precision@10_std": float(np.std(p10s)),
            "recall@10_mean": float(np.mean(r10s)),
            "recall@10_std": float(np.std(r10s)),
            "seeds": {str(s): all_results[str(s)][method] for s in seeds_done},
        }
    output = {
        "experiment": "CV-CLER 5-seed (fixed code, pbd=0, fuse=0)",
        "date": "2026-07-24",
        "seeds": SEEDS,
        "seeds_completed": list(seeds_done),
        "summary": summary,
        "all_results": all_results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  [Incremental save] {len(seeds_done)} seeds completed", flush=True)


def main():
    data = load_stravl_data()
    cfg = STRAVL_BEST
    print(f"Device: {cfg.device}", flush=True)

    all_results = {}
    seeds_done = set()
    if RESULTS_PATH.exists():
        try:
            existing = json.load(open(RESULTS_PATH, encoding="utf-8"))
            all_results = existing.get("all_results", {})
            seeds_done = set(int(s) for s in all_results.keys())
            print(f"Resuming: {len(seeds_done)} seeds already done: {sorted(seeds_done)}", flush=True)
        except Exception as e:
            print(f"Failed to load existing results: {e}", flush=True)

    for seed in SEEDS:
        if seed in seeds_done:
            print(f"\n{'='*60}\nSeed {seed} (already done, skipping)\n{'='*60}", flush=True)
            continue
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}", flush=True)
        t0 = time.time()
        try:
            m = run_cvcler_single_seed(seed, data, cfg)
            all_results[str(seed)] = {"CV-CLER": m}
            seeds_done.add(seed)
            elapsed = time.time() - t0
            print(f"Seed {seed} done in {elapsed:.1f}s", flush=True)
            save_incremental(all_results, sorted(seeds_done))
        except Exception as e:
            print(f"Seed {seed} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            save_incremental(all_results, sorted(seeds_done))
            continue

    print(f"\n{'='*60}\nFinal Summary ({len(seeds_done)}/{len(SEEDS)} seeds)\n{'='*60}", flush=True)
    save_incremental(all_results, sorted(seeds_done))
    ndcgs = [all_results[str(s)]["CV-CLER"]["ndcg@10"] for s in sorted(seeds_done)]
    print(f"CV-CLER: NDCG@10={np.mean(ndcgs):.4f}±{np.std(ndcgs):.4f} (n={len(ndcgs)})", flush=True)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
