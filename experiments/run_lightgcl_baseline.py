"""Run LightGCL baseline on Stravl (Table 6 missing entry).

LightGCL is listed in Table 6 with NDCG@10=0.0077 but has no result file.
This script runs it and saves to experiments/results/lightgcl_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TOURISM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOURISM_ROOT))

from travel_recommender.paper_pipeline import (
    LightGCLModel,
    TrainConfig,
    evaluate_topk,
    evaluate_topk_per_user,
    load_rec_data,
    set_seed,
    train_bpr_model,
)

RESULTS_PATH = ROOT / "experiments" / "results" / "lightgcl_results.json"
SEED = 42


def main():
    set_seed(SEED)
    data = load_rec_data()
    cfg = TrainConfig()
    cfg.lr = 0.01
    cfg.cl_weight = 0.01
    cfg.patience = 5
    cfg.max_epochs = 30
    cfg.device = "cuda"
    cfg.batch_size = 2048
    device = cfg.device
    print(f"Device: {device}", flush=True)
    print(f"Dataset stats: {data.stats}", flush=True)
    print(f"Config: lr={cfg.lr}, cl_weight={cfg.cl_weight}, batch_size={cfg.batch_size}", flush=True)

    print(f"\n{'=' * 60}\nTraining: LightGCL\n{'=' * 60}", flush=True)
    model = LightGCLModel(
        data.n_users,
        data.n_items,
        cfg.embed_dim,
        n_layers=3,
    )
    t0 = time.time()
    trained, history = train_bpr_model(model, data, cfg, method_name="LightGCL")
    metrics = evaluate_topk(trained, data, k=10, device=device)
    elapsed = time.time() - t0
    m = metrics
    print(
        f"  [LightGCL] P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} "
        f"NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s",
        flush=True,
    )

    result = {
        "experiment": "LightGCL baseline (Stravl, seed=42)",
        "date": "2026-07-24",
        "seed": SEED,
        "method": "LightGCL",
        "metrics": {
            "precision@10": m["precision@10"],
            "recall@10": m["recall@10"],
            "f1@10": m["f1@10"],
            "ndcg@10": m["ndcg@10"],
            "n_eval_users": m["n_eval_users"],
        },
        "train_time_sec": elapsed,
        "history": history,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
