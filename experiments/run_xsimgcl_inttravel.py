"""XSimGCL on IntTravel (3-shard merge) with 5 seeds.

目的：
  补全表23 IntTravel 行 XSimGCL 列的空缺（原为"—"），完成 P0-1/P1-4 关键对照实验。
  实验设计同 run_new_baselines.py（5种子）+ run_inttravel_cvcler_mmssl.py（IntTravel 加载/分批评估）。

数据集：IntTravel 三分片合并（n_shards=3, min_unique_pois=5, max_users=25000）
  与表23 中 CLER/CV-CLER/BPR/MMCF/ARFusion-Rec 同数据划分。

模型配置说明：
  - IntTravel 上 CV-CLER 用 use_graph=False（INTTRAVEL_BEST.n_gnn_layers=0），因为 CV-CLER
    的图传播在匿名 profile 域反而不利。
  - XSimGCL 本身是图对比学习模型，必须用图传播才能体现其方法特性。为公平对比，这里采用
    n_layers=2（与 Stravl/Beliefs 上 XSimGCL 一致），让 XSimGCL 在其设计适用场景下运行。

输出：
  D:\\tourism\\ARFusion_Research\\experiments\\results\\inttravel_xsimgcl_5seed_results.json
  D:\\tourism\\submission_CLER_CEA_CN\\experiments\\results\\inttravel_xsimgcl_5seed_results.json
  D:\\tourism\\ARFusion_Research\\experiments\\results\\inttravel_xsimgcl_5seed_log.txt
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable, Dict

# 禁用 PyTorch sparse tensor 警告
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from arfusion_recommender.configs import INTTRAVEL_BEST
from arfusion_recommender.pipeline import (
    BPRDataset,
    TrainConfig,
    bpr_loss,
    set_seed,
    build_bipartite_adj,
    load_inttravel_data,
)
from new_baselines import XSimGCLModel
from arfusion_recommender.arfusion_model import info_nce  # noqa: F401  (XSimGCL 内部使用)

SEEDS = [42, 123, 456, 789, 2024]

# XSimGCL 配置：n_layers=2（与 Stravl/Beliefs 一致），cl_weight=0.1（同 STRAVL_BEST）
XSIM_N_LAYERS = 2
XSIM_CL_WEIGHT = 0.1
XSIM_NOISE_SCALE = 0.1
XSIM_UI_TEMPERATURE = 0.2

# 输出路径
RESULTS_DIR_ARFUSION = ROOT / "experiments" / "results"
RESULTS_DIR_CEA = ROOT.parent / "submission_CLER_CEA_CN" / "experiments" / "results"
RESULTS_PATH_ARFUSION = RESULTS_DIR_ARFUSION / "inttravel_xsimgcl_5seed_results.json"
RESULTS_PATH_CEA = RESULTS_DIR_CEA / "inttravel_xsimgcl_5seed_results.json"
LOG_PATH = RESULTS_DIR_ARFUSION / "inttravel_xsimgcl_5seed_log.txt"

# 复用 run_inttravel_cvcler_mmssl.py 的分批评估函数（避免代码重复）
from run_inttravel_cvcler_mmssl import evaluate_topk_batched  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def save_incremental(
    all_results: Dict,
    completed_seeds: list,
    summary: Dict = None,
) -> None:
    """每完成一个种子即保存到两个位置，防止长时实验数据丢失。"""
    output = {
        "experiment": "IntTravel 3-shard XSimGCL 5-seed",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "XSimGCL",
        "model_config": {
            "n_layers": XSIM_N_LAYERS,
            "cl_weight": XSIM_CL_WEIGHT,
            "noise_scale": XSIM_NOISE_SCALE,
            "ui_temperature": XSIM_UI_TEMPERATURE,
            "embed_dim": INTTRAVEL_BEST.embed_dim,
            "lr": INTTRAVEL_BEST.lr,
            "weight_decay": INTTRAVEL_BEST.weight_decay,
            "batch_size": INTTRAVEL_BEST.batch_size,
            "max_epochs": INTTRAVEL_BEST.max_epochs,
            "patience": INTTRAVEL_BEST.patience,
        },
        "data_config": {"n_shards": 3, "min_unique_pois": 5, "max_users": 25000},
        "seeds": SEEDS,
        "completed_seeds": completed_seeds,
        "per_seed_results": all_results,
    }
    if summary:
        output["summary"] = summary

    for path in [RESULTS_PATH_ARFUSION, RESULTS_PATH_CEA]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"[SAVED] {len(completed_seeds)} seeds -> {RESULTS_PATH_ARFUSION}")
    log(f"[SAVED] {len(completed_seeds)} seeds -> {RESULTS_PATH_CEA}")


def xsimgcl_loss(model, users, pos, neg, cl_weight: float = XSIM_CL_WEIGHT):
    """BPR + cl_weight * InfoNCE 对比损失（与 run_new_baselines.py 一致）。"""
    s_pos = model.score(users, pos)
    s_neg = model.score(users, neg)
    bpr = bpr_loss(s_pos, s_neg)
    cl = model.contrastive_loss(users, pos)
    return bpr + cl_weight * cl


def train_xsimgcl_single_seed(
    model: XSimGCLModel,
    data,
    cfg: TrainConfig,
    seed: int,
) -> XSimGCLModel:
    """XSimGCL 单种子训练循环（图模型每 batch 清缓存 + 重新传播）。"""
    device = torch.device(cfg.device)
    model = model.to(device)
    # XSimGCL 需要邻接矩阵
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val, patience_ctr, best_state = float("inf"), 0, None

    for epoch in range(cfg.max_epochs):
        model.train()
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        train_losses = []
        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            optimizer.zero_grad()
            # 图模型每 batch 清缓存 + 重新传播（嵌入已更新）
            if hasattr(model, "clear_cache"):
                model.clear_cache()
            if hasattr(model, "propagate"):
                model.propagate()
            loss = xsimgcl_loss(model, users, pos, neg, XSIM_CL_WEIGHT)
            loss.backward()
            optimizer.step()
            # 释放传播缓存显存
            if hasattr(model, "clear_cache"):
                model.clear_cache()
            train_losses.append(loss.item())

        model.eval()
        # val 阶段图模型传播一次，所有 val batch 共用缓存
        if hasattr(model, "propagate"):
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
        log(
            f"  [XSimGCL s{seed}] Epoch {epoch + 1}/{cfg.max_epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                log(f"  [XSimGCL s{seed}] Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        # 恢复邻接矩阵注入（best_state 在 CPU 上 clone，需重新绑定设备相关张量）
        model.set_adj(build_bipartite_adj(data, device))

    return model


def run_single_seed(seed: int, data, cfg: TrainConfig) -> Dict[str, float]:
    """单种子实验：训练 + 评估。"""
    set_seed(seed)
    log(f"  [Seed {seed}] Training XSimGCL on IntTravel ...")
    t0 = time.time()

    xsim = XSimGCLModel(
        data.n_users,
        data.n_items,
        dim=cfg.embed_dim,
        n_layers=XSIM_N_LAYERS,
        cl_weight=XSIM_CL_WEIGHT,
        noise_scale=XSIM_NOISE_SCALE,
        ui_temperature=XSIM_UI_TEMPERATURE,
    )
    trained = train_xsimgcl_single_seed(xsim, data, cfg, seed)

    # 分批评估（IntTravel 25000 用户 × 87612 物品，需分批避免 OOM）
    log(f"  [Seed {seed}] Evaluating XSimGCL (batched) ...")
    t1 = time.time()
    metrics = evaluate_topk_batched(trained, data, k=10, device=cfg.device, batch_size=500)
    elapsed_train = time.time() - t0
    elapsed_eval = time.time() - t1
    m = metrics
    log(
        f"  [Seed {seed}] XSimGCL: P@10={m['precision@10']:.6f} R@10={m['recall@10']:.6f} "
        f"F1@10={m['f1@10']:.6f} NDCG@10={m['ndcg@10']:.6f} "
        f"(train={elapsed_train:.1f}s eval={elapsed_eval:.1f}s)"
    )

    del xsim, trained
    torch.cuda.empty_cache()

    return {
        "precision@10": float(metrics["precision@10"]),
        "recall@10": float(metrics["recall@10"]),
        "f1@10": float(metrics["f1@10"]),
        "ndcg@10": float(metrics["ndcg@10"]),
        "n_eval_users": int(metrics["n_eval_users"]),
        "train_time_sec": float(elapsed_train),
        "eval_time_sec": float(elapsed_eval),
    }


def main():
    log("=" * 60)
    log("XSimGCL on IntTravel (3-shard merge) — 5-seed experiment")
    log("=" * 60)

    # 加载 IntTravel 三分片数据
    log("Loading IntTravel 3-shard dataset ...")
    t0 = time.time()
    data = load_inttravel_data(n_shards=3, min_unique_pois=5, max_users=25000)
    log(f"Dataset loaded in {time.time() - t0:.1f}s")
    log(f"  Users: {data.n_users:,}, Items: {data.n_items:,}")
    log(f"  Train/Val/Test: {len(data.train):,}/{len(data.val):,}/{len(data.test):,}")
    log(f"  Test users: {len(data.test_relevant):,}")
    log(f"  Feat dim: {data.user_features.shape[1]}")
    log(f"  Stats: {data.stats}")

    cfg = INTTRAVEL_BEST
    log(f"Config (from INTTRAVEL_BEST): embed_dim={cfg.embed_dim}, lr={cfg.lr}, "
        f"batch_size={cfg.batch_size}, max_epochs={cfg.max_epochs}, patience={cfg.patience}")
    log(f"XSimGCL overrides: n_layers={XSIM_N_LAYERS}, cl_weight={XSIM_CL_WEIGHT}, "
        f"noise_scale={XSIM_NOISE_SCALE}, ui_temperature={XSIM_UI_TEMPERATURE}")
    log(f"Device: {cfg.device}")

    all_results: Dict[str, Dict] = {}
    completed_seeds: list = []

    # 恢复机制：若已有结果，跳过已完成种子
    if RESULTS_PATH_ARFUSION.exists():
        try:
            existing = json.load(open(RESULTS_PATH_ARFUSION, encoding="utf-8"))
            all_results = existing.get("per_seed_results", {})
            completed_seeds = [int(s) for s in existing.get("completed_seeds", [])]
            log(f"Resuming: {len(completed_seeds)} seeds already done: {sorted(completed_seeds)}")
        except Exception as e:
            log(f"Failed to load existing results: {e}")

    for seed in SEEDS:
        if seed in completed_seeds:
            log(f"\n{'=' * 60}\n[Seed {seed}] (already done, skipping)\n{'=' * 60}")
            continue
        log(f"\n{'=' * 60}")
        log(f"[Seed {seed}] START")
        log(f"{'=' * 60}")
        t0 = time.time()
        try:
            seed_results = run_single_seed(seed, data, cfg)
            elapsed = time.time() - t0
            log(f"[Seed {seed}] DONE in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
            all_results[str(seed)] = seed_results
            completed_seeds.append(seed)
            completed_seeds.sort()
            save_incremental(all_results, completed_seeds)
        except Exception as e:
            log(f"[Seed {seed}] FAILED: {e}")
            log(traceback.format_exc())
            save_incremental(all_results, completed_seeds)

    # 汇总均值±标准差
    log(f"\n{'=' * 60}")
    log("Computing summary statistics ...")
    log(f"{'=' * 60}")
    summary: Dict[str, Dict] = {}
    ndcgs, p10s, r10s, f1s, times = [], [], [], [], []
    for s in completed_seeds:
        key = str(s)
        if key in all_results:
            m = all_results[key]
            ndcgs.append(m["ndcg@10"])
            p10s.append(m["precision@10"])
            r10s.append(m["recall@10"])
            f1s.append(m["f1@10"])
            times.append(m.get("train_time_sec", 0.0))

    if ndcgs:
        summary = {
            "method": "XSimGCL",
            "n_seeds": len(ndcgs),
            "precision@10": {
                "mean": float(np.mean(p10s)),
                "std": float(np.std(p10s, ddof=1)) if len(p10s) > 1 else 0.0,
            },
            "recall@10": {
                "mean": float(np.mean(r10s)),
                "std": float(np.std(r10s, ddof=1)) if len(r10s) > 1 else 0.0,
            },
            "f1@10": {
                "mean": float(np.mean(f1s)),
                "std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            },
            "ndcg@10": {
                "mean": float(np.mean(ndcgs)),
                "std": float(np.std(ndcgs, ddof=1)) if len(ndcgs) > 1 else 0.0,
            },
            "train_time_sec": {
                "mean": float(np.mean(times)),
                "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            },
        }
        log(f"\nXSimGCL on IntTravel (5-seed summary):")
        log(f"  NDCG@10   = {summary['ndcg@10']['mean']:.6f} ± {summary['ndcg@10']['std']:.6f}")
        log(f"  P@10       = {summary['precision@10']['mean']:.6f} ± {summary['precision@10']['std']:.6f}")
        log(f"  R@10       = {summary['recall@10']['mean']:.6f} ± {summary['recall@10']['std']:.6f}")
        log(f"  F1@10      = {summary['f1@10']['mean']:.6f} ± {summary['f1@10']['std']:.6f}")
        log(f"  Train time = {summary['train_time_sec']['mean']:.1f}s ± {summary['train_time_sec']['std']:.1f}s")

    save_incremental(all_results, completed_seeds, summary)
    log("\nAll done.")


if __name__ == "__main__":
    main()
