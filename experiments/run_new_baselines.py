"""Run XSimGCL / BM3 / DiffRec baselines on Stravl and MovieLens Beliefs with 5 seeds.

三个 2023 强基线在两个数据集上的 5 种子实验，结果格式与 beliefs_5seed_results.json 一致。

特殊处理：
- XSimGCL: 图传播模型，每 batch 清缓存 + 重新传播；BPR + cl_weight * 对比损失
- BM3:     非图模型，需 set_user_features；BPR + cl_weight * 模态对比 + align_weight * 跨模态对齐
- DiffRec: 非图模型；BPR + diff_weight * 扩散 MSE 损失

输出:
  experiments/results/new_baselines_stravl_results.json
  experiments/results/new_baselines_beliefs_results.json
  experiments/results/new_baselines_log.txt
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    load_stravl_data,
    train_model,  # noqa: F401  (任务要求导入，保留以备扩展)
    evaluate_topk,
    TrainConfig,  # noqa: F401
    BPRDataset,
    bpr_loss,
    set_seed,
    build_bipartite_adj,
    user_positive_counts,  # noqa: F401  (任务要求导入)
)
from load_movielens_beliefs import load_movielens_beliefs
from new_baselines import XSimGCLModel, BM3Model, DiffRecModel

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_DIR = ROOT / "experiments" / "results"
STRAVL_RESULTS_PATH = RESULTS_DIR / "new_baselines_stravl_results.json"
BELIEFS_RESULTS_PATH = RESULTS_DIR / "new_baselines_beliefs_results.json"
LOG_PATH = RESULTS_DIR / "new_baselines_log.txt"

# 损失权重（与任务描述一致；BM3/DiffRec 的辅助损失权重不依赖 cfg）
CL_WEIGHT = 0.1       # XSimGCL / BM3 模态内对比
ALIGN_WEIGHT = 0.1    # BM3 跨模态对齐
DIFF_WEIGHT = 0.05    # DiffRec 扩散损失（较小，避免主导 BPR）

METHODS = ["XSimGCL", "BM3", "DiffRec"]


# =============================================================================
# 日志与保存
# =============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def save_incremental(
    all_results: Dict,
    completed_seeds: list,
    results_path: Path,
    dataset_name: str,
    summary: Dict = None,
) -> None:
    """每完成一个种子即保存，防止长时实验数据丢失。"""
    output = {
        "experiment": f"{dataset_name} new-baselines 5-seed",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": METHODS,
        "seeds": SEEDS,
        "completed_seeds": completed_seeds,
        "per_seed_results": all_results,
    }
    if summary:
        output["summary"] = summary
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"[SAVED] {len(completed_seeds)} seeds -> {results_path}")


# =============================================================================
# 自定义损失函数
# =============================================================================
def xsimgcl_loss(model, users, pos, neg, cl_weight: float = CL_WEIGHT):
    s_pos = model.score(users, pos)
    s_neg = model.score(users, neg)
    bpr = bpr_loss(s_pos, s_neg)
    cl = model.contrastive_loss(users, pos)
    return bpr + cl_weight * cl


def bm3_loss(
    model,
    users,
    pos,
    neg,
    cl_weight: float = CL_WEIGHT,
    align_weight: float = ALIGN_WEIGHT,
):
    s_pos = model.score(users, pos)
    s_neg = model.score(users, neg)
    bpr = bpr_loss(s_pos, s_neg)
    cl = model.modality_contrastive_loss(users, pos)
    align = model.cross_modal_align_loss(users, pos)
    return bpr + cl_weight * cl + align_weight * align


def diffrec_loss(model, users, pos, neg, diff_weight: float = DIFF_WEIGHT):
    s_pos = model.score(users, pos)
    s_neg = model.score(users, neg)
    bpr = bpr_loss(s_pos, s_neg)
    diff = model.diffusion_loss(users, pos)
    return bpr + diff_weight * diff


# =============================================================================
# 自定义训练循环（三个基线均有辅助损失，无法直接用 pipeline.train_model）
# =============================================================================
def train_custom(
    model,
    data,
    cfg: TrainConfig,
    method_name: str,
    loss_fn: Callable,
):
    """自定义训练循环，loss_fn(model, users, pos, neg) -> loss。

    对图模型（XSimGCL）每 batch 清缓存 + 重新传播；
    对非图模型（BM3/DiffRec）跳过图传播。
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    if hasattr(model, "set_user_features"):
        model.set_user_features(data.user_features)
    if hasattr(model, "set_adj"):
        # XSimGCL 需要邻接矩阵；BM3/DiffRec 的 set_adj 为空实现
        adj = build_bipartite_adj(data, device)
        model.set_adj(adj)

    train_loader = DataLoader(
        BPRDataset(data, "train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    is_graph = hasattr(model, "adj") and (getattr(model, "adj", None) is not None)

    best_val, patience_ctr, best_state = float("inf"), 0, None
    for epoch in range(cfg.max_epochs):
        model.train()
        # epoch 开头清缓存（图模型）
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        train_losses = []
        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            optimizer.zero_grad()
            # 图模型每 batch 清缓存 + 重新传播（嵌入已更新）
            if is_graph and hasattr(model, "clear_cache"):
                model.clear_cache()
            if is_graph and hasattr(model, "propagate"):
                model.propagate()
            loss = loss_fn(model, users, pos, neg)
            loss.backward()
            optimizer.step()
            # 释放传播缓存显存
            if is_graph and hasattr(model, "clear_cache"):
                model.clear_cache()
            train_losses.append(loss.item())

        model.eval()
        # val 阶段图模型传播一次，所有 val batch 共用缓存
        if is_graph and hasattr(model, "propagate"):
            with torch.no_grad():
                model.propagate()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                users, pos, neg = [x.to(device) for x in batch]
                vl = bpr_loss(model.score(users, pos), model.score(users, neg))
                val_losses.append(vl.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        scheduler.step(val_loss)
        log(f"  [{method_name}] Epoch {epoch + 1}/{cfg.max_epochs} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                log(f"  [{method_name}] Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        # 恢复资源注入（best_state 在 CPU 上 clone，需重新绑定设备相关张量）
        if hasattr(model, "set_user_features"):
            model.set_user_features(data.user_features)
        if is_graph and hasattr(model, "set_adj"):
            model.set_adj(build_bipartite_adj(data, device))

    return model


# =============================================================================
# 单种子实验：3 基线
# =============================================================================
def run_single_seed(seed: int, data, cfg: TrainConfig) -> Dict[str, Dict[str, float]]:
    set_seed(seed)
    feat_dim = data.user_features.shape[1]
    results: Dict[str, Dict[str, float]] = {}

    # --- XSimGCL ---
    log(f"  [Seed {seed}] Training XSimGCL ...")
    t0 = time.time()
    xsim = XSimGCLModel(
        data.n_users,
        data.n_items,
        dim=cfg.embed_dim,
        n_layers=cfg.n_gnn_layers,
        cl_weight=cfg.cl_weight,
        ui_temperature=cfg.ui_temperature,
    )
    trained = train_custom(
        xsim,
        data,
        cfg,
        f"XSimGCL-s{seed}",
        lambda m, u, p, n: xsimgcl_loss(m, u, p, n, CL_WEIGHT),
    )
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["XSimGCL"] = m
    log(
        f"  [Seed {seed}] XSimGCL: NDCG@10={m['ndcg@10']:.4f} "
        f"P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} ({time.time() - t0:.1f}s)"
    )
    del xsim, trained
    torch.cuda.empty_cache()

    # --- BM3 ---
    log(f"  [Seed {seed}] Training BM3 ...")
    t0 = time.time()
    bm3 = BM3Model(
        data.n_users,
        data.n_items,
        feat_dim,
        dim=cfg.embed_dim,
        cl_weight=CL_WEIGHT,
        align_weight=ALIGN_WEIGHT,
    )
    trained = train_custom(
        bm3,
        data,
        cfg,
        f"BM3-s{seed}",
        lambda m, u, p, n: bm3_loss(m, u, p, n, CL_WEIGHT, ALIGN_WEIGHT),
    )
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["BM3"] = m
    log(
        f"  [Seed {seed}] BM3: NDCG@10={m['ndcg@10']:.4f} "
        f"P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} ({time.time() - t0:.1f}s)"
    )
    del bm3, trained
    torch.cuda.empty_cache()

    # --- DiffRec ---
    log(f"  [Seed {seed}] Training DiffRec ...")
    t0 = time.time()
    diff = DiffRecModel(
        data.n_users,
        data.n_items,
        dim=cfg.embed_dim,
        diff_weight=DIFF_WEIGHT,
    )
    trained = train_custom(
        diff,
        data,
        cfg,
        f"DiffRec-s{seed}",
        lambda m, u, p, n: diffrec_loss(m, u, p, n, DIFF_WEIGHT),
    )
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["DiffRec"] = m
    log(
        f"  [Seed {seed}] DiffRec: NDCG@10={m['ndcg@10']:.4f} "
        f"P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} ({time.time() - t0:.1f}s)"
    )
    del diff, trained
    torch.cuda.empty_cache()

    return results


# =============================================================================
# 单数据集实验：3 基线 × 5 种子
# =============================================================================
def run_dataset(dataset_name: str, load_fn: Callable, results_path: Path) -> Dict:
    log("=" * 60)
    log(f"Dataset: {dataset_name}")
    log("=" * 60)

    log(f"Loading {dataset_name} dataset ...")
    t0 = time.time()
    data = load_fn()
    log(f"Dataset loaded in {time.time() - t0:.1f}s")
    log(f"  Users: {data.n_users:,}, Items: {data.n_items:,}")
    log(f"  Train/Val/Test: {len(data.train):,}/{len(data.val):,}/{len(data.test):,}")
    log(f"  Test users: {len(data.test_relevant):,}")
    log(f"  Feat dim: {data.user_features.shape[1]}")

    cfg = STRAVL_BEST
    log(f"Config: embed_dim={cfg.embed_dim}, lr={cfg.lr}, batch_size={cfg.batch_size}")
    log(f"  cl_weight={cfg.cl_weight} (XSimGCL/BM3), align_weight={ALIGN_WEIGHT} (BM3), diff_weight={DIFF_WEIGHT} (DiffRec)")
    log(f"  max_epochs={cfg.max_epochs}, patience={cfg.patience}, device={cfg.device}")

    all_results: Dict[str, Dict] = {}
    completed_seeds: list = []

    # 恢复机制：若已有结果，跳过已完成种子
    if results_path.exists():
        try:
            existing = json.load(open(results_path, encoding="utf-8"))
            all_results = existing.get("per_seed_results", {})
            completed_seeds = [int(s) for s in existing.get("completed_seeds", [])]
            log(f"Resuming: {len(completed_seeds)} seeds already done: {sorted(completed_seeds)}")
        except Exception as e:
            log(f"Failed to load existing results: {e}")

    for seed in SEEDS:
        if seed in completed_seeds:
            log(f"\n{'=' * 60}\n[{dataset_name}] Seed {seed} (already done, skipping)\n{'=' * 60}")
            continue
        log(f"\n{'=' * 60}")
        log(f"[{dataset_name}] Seed {seed} START")
        log(f"{'=' * 60}")
        t0 = time.time()
        try:
            seed_results = run_single_seed(seed, data, cfg)
            elapsed = time.time() - t0
            log(f"[{dataset_name}] Seed {seed} DONE in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
            all_results[str(seed)] = seed_results
            completed_seeds.append(seed)
            completed_seeds.sort()
            save_incremental(all_results, completed_seeds, results_path, dataset_name)
        except Exception as e:
            log(f"[{dataset_name}] Seed {seed} FAILED: {e}")
            log(traceback.format_exc())
            save_incremental(all_results, completed_seeds, results_path, dataset_name)

    # 汇总均值±标准差
    log(f"\n{'=' * 60}")
    log(f"[{dataset_name}] Computing summary statistics ...")
    log(f"{'=' * 60}")
    summary: Dict[str, Dict] = {}
    for method in METHODS:
        ndcgs, p10s, r10s, f1s = [], [], [], []
        for s in completed_seeds:
            key = str(s)
            if key in all_results and method in all_results[key]:
                m = all_results[key][method]
                ndcgs.append(m["ndcg@10"])
                p10s.append(m["precision@10"])
                r10s.append(m["recall@10"])
                f1s.append(m["f1@10"])
        if ndcgs:
            summary[method] = {
                "ndcg@10_mean": float(np.mean(ndcgs)),
                "ndcg@10_std": float(np.std(ndcgs)),
                "precision@10_mean": float(np.mean(p10s)),
                "precision@10_std": float(np.std(p10s)),
                "recall@10_mean": float(np.mean(r10s)),
                "recall@10_std": float(np.std(r10s)),
                "f1@10_mean": float(np.mean(f1s)),
                "f1@10_std": float(np.std(f1s)),
                "n_seeds": len(ndcgs),
                "seeds": {
                    str(s): all_results[str(s)][method]
                    for s in completed_seeds
                    if str(s) in all_results and method in all_results[str(s)]
                },
            }
            log(
                f"  {method}: NDCG@10={np.mean(ndcgs):.4f}±{np.std(ndcgs):.4f} "
                f"P@10={np.mean(p10s):.4f}±{np.std(p10s):.4f} "
                f"R@10={np.mean(r10s):.4f}±{np.std(r10s):.4f} (n={len(ndcgs)})"
            )

    save_incremental(all_results, completed_seeds, results_path, dataset_name, summary)
    log(f"[{dataset_name}] Experiment complete! Results saved to {results_path}")
    return {"per_seed_results": all_results, "summary": summary, "completed_seeds": completed_seeds}


# =============================================================================
# 主入口
# =============================================================================
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"New baselines 5-seed experiment started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log("=" * 60)
    log("New Baselines (XSimGCL / BM3 / DiffRec) 5-seed Experiment")
    log(f"Seeds: {SEEDS}")
    log(f"Loss weights: cl_weight={CL_WEIGHT}, align_weight={ALIGN_WEIGHT}, diff_weight={DIFF_WEIGHT}")
    log("=" * 60)

    # --- Stravl ---
    stravl_out = run_dataset("Stravl", load_stravl_data, STRAVL_RESULTS_PATH)

    # --- MovieLens Beliefs ---
    beliefs_out = run_dataset("Beliefs", load_movielens_beliefs, BELIEFS_RESULTS_PATH)

    log("\n" + "=" * 60)
    log("ALL EXPERIMENTS COMPLETE")
    log("=" * 60)
    log(f"Stravl:  {len(stravl_out['completed_seeds'])}/{len(SEEDS)} seeds -> {STRAVL_RESULTS_PATH}")
    log(f"Beliefs: {len(beliefs_out['completed_seeds'])}/{len(SEEDS)} seeds -> {BELIEFS_RESULTS_PATH}")


if __name__ == "__main__":
    main()
