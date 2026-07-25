"""Cross-dataset experiments (Tables 20-21).

Runs BPR, Multimodal CF, MMSSL, MMSSL-decoupled, CLER, CV-CLER
on MovieLens-1M and Amazon-Electronics with seed=42.

Saves results to experiments/results/cross_dataset_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
TOURISM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOURISM_ROOT))

from travel_recommender.paper_pipeline import (
    CLERModel,
    CVCLERModel,
    MFModel,
    MMSSLDecoupledModel,
    MMSSLModel,
    MultimodalCFModel,
    TrainConfig,
    evaluate_topk,
    evaluate_topk_per_user,
    init_cvcler_from_cler,
    save_results,
    set_seed,
    train_bpr_model,
)
from travel_recommender.public_dataset_loaders import (
    load_amazon_electronics_5core,
    load_movielens_1m,
)

RESULTS_PATH = ROOT / "experiments" / "results" / "cross_dataset_results.json"
SEED = 42


def run_one(name, model, data, cfg, return_model=False, **kwargs):
    print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}", flush=True)
    t0 = time.time()
    trained, history = train_bpr_model(model, data, cfg, method_name=name, **kwargs)
    metrics = evaluate_topk(trained, data, k=10, device=cfg.device)
    per_user = evaluate_topk_per_user(trained, data, k=10, device=cfg.device)
    elapsed = time.time() - t0
    m = metrics
    print(
        f"  [{name}] P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} "
        f"NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s",
        flush=True,
    )
    result = {
        "method": name,
        "metrics": metrics,
        "per_user_ndcg": {str(uid): vals["ndcg@10"] for uid, vals in per_user.items()},
        "history": history,
        "train_time_sec": elapsed,
    }
    if return_model:
        return result, trained
    return result


def paired_significance(a_ndcg: dict, b_ndcg: dict, label: str) -> dict:
    common = sorted(set(a_ndcg) & set(b_ndcg), key=lambda x: int(x))
    diffs = [a_ndcg[u] - b_ndcg[u] for u in common]
    if not diffs or np.std(diffs) == 0:
        return {"comparison": label, "n": len(common), "delta": float(np.mean(diffs)), "wilcoxon_p": 1.0}
    stat, p = wilcoxon(diffs, alternative="greater")
    return {
        "comparison": label,
        "n": len(common),
        "delta": float(np.mean(diffs)),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
    }


def build_cvcler(data, feat_dim, cfg):
    return CVCLERModel(
        data.n_users,
        data.n_items,
        feat_dim,
        cfg.embed_dim,
        use_cross_view=True,
        use_ui_cl=True,
        cv_temperature=cfg.cv_temperature,
        ui_temperature=cfg.ui_temperature,
    )


def run_dataset_suite(data, dataset_name: str) -> dict:
    cfg = TrainConfig()
    feat_dim = data.user_features.shape[1]
    print(f"\n### Dataset: {dataset_name} ###", flush=True)
    print(data.stats, flush=True)

    results = {}
    results["BPR"] = run_one("BPR", MFModel(data.n_users, data.n_items, cfg.embed_dim), data, cfg)
    results["Multimodal CF"] = run_one(
        "Multimodal CF",
        MultimodalCFModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim),
        data,
        cfg,
    )
    results["MMSSL"] = run_one(
        "MMSSL",
        MMSSLModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim, n_layers=3),
        data,
        cfg,
    )
    results["MMSSL-decoupled"] = run_one(
        "MMSSL-decoupled",
        MMSSLDecoupledModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim, n_layers=3),
        data,
        cfg,
    )

    cler_model = CLERModel(
        data.n_users, data.n_items, cfg.embed_dim, use_cl=True, ui_temperature=cfg.ui_temperature
    )
    results["CLER"], cler_trained = run_one(
        "CLER",
        cler_model,
        data,
        cfg,
        return_model=True,
        use_cl=True,
        cl_fn=lambda u, i, m=cler_model: m.contrastive_loss(u, i),
    )
    cvcler = build_cvcler(data, feat_dim, cfg)
    init_cvcler_from_cler(cvcler, cler_trained)
    results["CV-CLER"] = run_one("CV-CLER", cvcler, data, cfg)

    significance = [
        paired_significance(results["CV-CLER"]["per_user_ndcg"], results["BPR"]["per_user_ndcg"], "CV-CLER vs BPR"),
        paired_significance(results["CV-CLER"]["per_user_ndcg"], results["CLER"]["per_user_ndcg"], "CV-CLER vs CLER"),
        paired_significance(results["MMSSL-decoupled"]["per_user_ndcg"], results["MMSSL"]["per_user_ndcg"], "MMSSL-decoupled vs MMSSL"),
    ]

    summary = [
        {
            "method": k,
            "precision@10": round(v["metrics"]["precision@10"], 4),
            "recall@10": round(v["metrics"]["recall@10"], 4),
            "ndcg@10": round(v["metrics"]["ndcg@10"], 4),
            "train_time_sec": round(v["train_time_sec"], 1),
        }
        for k, v in results.items()
    ]

    return {
        "dataset": dataset_name,
        "stats": data.stats,
        "config": {
            "embed_dim": cfg.embed_dim,
            "gnn_layers": 3,
            "split": "per-user 80/10/10",
            "cl_weight": cfg.cl_weight,
            "cv_weight": cfg.cv_weight,
            "device": cfg.device,
        },
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_user_ndcg"} for k, v in results.items()
        },
        "significance": significance,
        "summary_table": summary,
    }


def main():
    set_seed(SEED)
    cross = {}

    print("\n### Loading MovieLens-1M ###", flush=True)
    ml = load_movielens_1m()
    cross["movielens_1m"] = run_dataset_suite(ml, "MovieLens-1M")

    # Save incrementally after MovieLens
    save_results(
        {
            "experiment": "Cross-dataset validation (seed=42)",
            "date": "2026-07-24",
            "seed": SEED,
            "results": cross,
        },
        str(RESULTS_PATH),
    )
    print(f"\n[Saved incremental] {RESULTS_PATH}", flush=True)

    print("\n### Loading Amazon-Electronics ###", flush=True)
    amazon = load_amazon_electronics_5core()
    cross["amazon_electronics"] = run_dataset_suite(amazon, "Amazon-Electronics-5core")

    save_results(
        {
            "experiment": "Cross-dataset validation (seed=42)",
            "date": "2026-07-24",
            "seed": SEED,
            "results": cross,
        },
        str(RESULTS_PATH),
    )
    print(f"\n[Saved] {RESULTS_PATH}", flush=True)

    # Print summary
    print(f"\n{'=' * 70}\nCross-Dataset Summary\n{'=' * 70}", flush=True)
    for ds_name, ds_results in cross.items():
        print(f"\n{ds_name}:", flush=True)
        for row in sorted(ds_results["summary_table"], key=lambda x: -x["ndcg@10"]):
            print(
                f"  {row['method']:20s} NDCG@10={row['ndcg@10']:.4f} P@10={row['precision@10']:.4f} R@10={row['recall@10']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
