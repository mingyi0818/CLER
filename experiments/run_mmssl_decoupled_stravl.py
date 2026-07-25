"""Run MMSSL-decoupled baseline on Stravl (Table 9 missing entry).

MMSSL-decoupled: same training as MMSSL (cross-modal InfoNCE) but inference
uses only behavior embedding (no fusion). This isolates the effect of
inference-time fusion vs training-time alignment.

Saves to experiments/results/mmssl_decoupled_stravl_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOURISM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOURISM_ROOT))

from travel_recommender.paper_pipeline import (
    MMSSLDecoupledModel,
    TrainConfig,
    evaluate_topk,
    evaluate_topk_per_user,
    load_rec_data,
    set_seed,
    train_bpr_model,
)

RESULTS_PATH = ROOT / "experiments" / "results" / "mmssl_decoupled_stravl_results.json"
SEED = 42


def main():
    set_seed(SEED)
    data = load_rec_data()
    cfg = TrainConfig()
    cfg.device = "cuda"
    feat_dim = data.user_features.shape[1]
    device = cfg.device
    print(f"Device: {device}", flush=True)
    print(f"Dataset stats: {data.stats}", flush=True)

    print(f"\n{'=' * 60}\nTraining: MMSSL-decoupled\n{'=' * 60}", flush=True)
    model = MMSSLDecoupledModel(
        data.n_users,
        data.n_items,
        feat_dim,
        cfg.embed_dim,
        n_layers=3,
    )
    t0 = time.time()
    trained, history = train_bpr_model(model, data, cfg, method_name="MMSSL-decoupled")
    metrics = evaluate_topk(trained, data, k=10, device=device)
    elapsed = time.time() - t0
    m = metrics
    print(
        f"  [MMSSL-decoupled] P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} "
        f"NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s",
        flush=True,
    )

    result = {
        "experiment": "MMSSL-decoupled baseline (Stravl, seed=42)",
        "date": "2026-07-24",
        "seed": SEED,
        "method": "MMSSL-decoupled",
        "description": "MMSSL training (cross-modal InfoNCE) but inference uses only behavior embedding",
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
