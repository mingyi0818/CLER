"""Profile noise injection ablation (Table 13).

Applies random permutation to user profile features at ratio eta in {0, 0.5, 1.0}
and trains Multimodal CF, MMSSL, CV-CLER on Stravl with seed=42.

Saves results to experiments/results/profile_noise_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure both project roots are importable
ROOT = Path(__file__).resolve().parent
TOURISM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOURISM_ROOT))

from travel_recommender.paper_pipeline import (
    CLERModel,
    CVCLERModel,
    MFModel,
    MMSSLModel,
    MultimodalCFModel,
    TrainConfig,
    evaluate_topk,
    evaluate_topk_per_user,
    init_cvcler_from_cler,
    load_rec_data,
    save_results,
    set_seed,
    train_bpr_model,
)

RESULTS_PATH = ROOT / "experiments" / "results" / "profile_noise_results.json"
SEED = 42
NOISE_RATIOS = [0.0, 0.5, 1.0]


def apply_profile_noise(features: np.ndarray, eta: float, rng: np.random.RandomState) -> np.ndarray:
    """Permute eta fraction of users' profile features.

    For each user, with probability eta, replace their features with features
    sampled from a randomly chosen other user (without replacement per draw).
    """
    if eta <= 0.0:
        return features.copy()
    n = features.shape[0]
    noisy = features.copy()
    # Select users to corrupt
    mask = rng.random(n) < eta
    n_corrupt = int(mask.sum())
    if n_corrupt == 0:
        return noisy
    # Permutation source: shuffle indices, avoid self-assignment
    perm = rng.permutation(n)
    # Ensure no fixed points among corrupted users
    for i in np.where(mask)[0]:
        src = perm[i]
        if src == i:
            # Swap with next index
            src = (src + 1) % n
        noisy[i] = features[src]
    return noisy


def run_one(name, model, data, cfg, return_model=False, **kwargs):
    print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}", flush=True)
    t0 = time.time()
    trained, history = train_bpr_model(model, data, cfg, method_name=name, **kwargs)
    metrics = evaluate_topk(trained, data, k=10, device=cfg.device)
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
        "history": history,
        "train_time_sec": elapsed,
    }
    if return_model:
        return result, trained
    return result


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


def main():
    set_seed(SEED)
    data_original = load_rec_data()
    cfg = TrainConfig()
    feat_dim = data_original.user_features.shape[1]
    print(f"Device: {cfg.device}", flush=True)
    print(f"Dataset stats: {data_original.stats}", flush=True)

    all_results = {}

    for eta in NOISE_RATIOS:
        eta_key = f"{eta:.1f}"
        print(f"\n{'#' * 70}\n# Profile noise eta={eta_key}\n{'#' * 70}", flush=True)

        # Create data copy with noisy features
        import copy
        data = copy.deepcopy(data_original)
        rng = np.random.RandomState(SEED)
        data.user_features = apply_profile_noise(data_original.user_features, eta, rng)
        print(f"  Applied noise: {eta*100:.0f}% of users have permuted features", flush=True)

        eta_results = {}

        # Multimodal CF
        set_seed(SEED)
        eta_results["Multimodal CF"] = run_one(
            "Multimodal CF",
            MultimodalCFModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim),
            data,
            cfg,
        )

        # MMSSL
        set_seed(SEED)
        eta_results["MMSSL"] = run_one(
            "MMSSL",
            MMSSLModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim, n_layers=3),
            data,
            cfg,
        )

        # CV-CLER (needs CLER warm start)
        set_seed(SEED)
        cler_model = CLERModel(
            data.n_users, data.n_items, cfg.embed_dim, use_cl=True, ui_temperature=cfg.ui_temperature
        )
        cler_result, cler_trained = run_one(
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
        eta_results["CV-CLER"] = run_one("CV-CLER", cvcler, data, cfg)

        # Store summary
        all_results[eta_key] = {
            "eta": eta,
            "methods": {
                method: {
                    "precision@10": res["metrics"]["precision@10"],
                    "recall@10": res["metrics"]["recall@10"],
                    "f1@10": res["metrics"]["f1@10"],
                    "ndcg@10": res["metrics"]["ndcg@10"],
                    "n_eval_users": res["metrics"]["n_eval_users"],
                    "train_time_sec": res["train_time_sec"],
                }
                for method, res in eta_results.items()
            },
        }

        # Save incrementally
        save_results(
            {
                "experiment": "Profile noise injection ablation (Stravl, seed=42)",
                "date": "2026-07-24",
                "seed": SEED,
                "noise_ratios": NOISE_RATIOS,
                "dataset_stats": data_original.stats,
                "results": all_results,
            },
            str(RESULTS_PATH),
        )
        print(f"  [Saved incremental] {RESULTS_PATH}", flush=True)

    # Print summary table
    print(f"\n{'=' * 70}\nProfile Noise Injection Summary (NDCG@10)\n{'=' * 70}", flush=True)
    print(f"{'eta':<8} {'Multimodal CF':<18} {'MMSSL':<18} {'CV-CLER':<18}", flush=True)
    for eta in NOISE_RATIOS:
        eta_key = f"{eta:.1f}"
        r = all_results[eta_key]["methods"]
        print(
            f"{eta_key:<8} {r['Multimodal CF']['ndcg@10']:<18.4f} "
            f"{r['MMSSL']['ndcg@10']:<18.4f} {r['CV-CLER']['ndcg@10']:<18.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
