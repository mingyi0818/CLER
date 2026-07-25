"""Compute rho_xi_t0 on MovieLens Beliefs dataset (P1-2 of paperadvice.md).

目的：
  补全论文中"Beliefs 上 rho≈0.31"的源文件，使用与 Stravl/MovieLens/Amazon/IntTravel
  完全相同的协议：50k BPR 三元组 + Multimodal CF warm model + bootstrap 95% CI。

输出：
  D:\\tourism\\ARFusion_Research\\experiments\\results\\cross_domain_rho_beliefs.json
  D:\\tourism\\submission_CLER_CEA_CN\\experiments\\cross_domain_rho_beliefs.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from travel_recommender.paper_pipeline import (
    MultimodalCFModel,
    TrainConfig,
    set_seed,
    train_bpr_model,
)
from load_movielens_beliefs import load_movielens_beliefs

N_SAMPLES = 50_000
N_BOOT = 1000
SEED = 42

OUT_ARFUSION = ROOT / "experiments" / "results" / "cross_domain_rho_beliefs.json"
OUT_CEA = ROOT.parent / "submission_CLER_CEA_CN" / "experiments" / "cross_domain_rho_beliefs.json"


def bootstrap_ci(t0: np.ndarray, xi: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(t0)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(t0[idx]) < 1e-12 or np.std(xi[idx]) < 1e-12:
            continue
        rhos.append(float(np.corrcoef(t0[idx], xi[idx])[0, 1]))
    if not rhos:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def compute_rho_beliefs(data, cfg: TrainConfig, n_samples: int = N_SAMPLES) -> dict:
    set_seed(SEED)
    feat_dim = data.user_features.shape[1]
    print(f"Beliefs stats: n_users={data.n_users}, n_items={data.n_items}, "
          f"feat_dim={feat_dim}, train={len(data.train)}, val={len(data.val)}, test={len(data.test)}",
          flush=True)

    print("Training Multimodal CF on Beliefs for rho proxy...", flush=True)
    t0 = time.time()
    model = MultimodalCFModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim)
    trained, _ = train_bpr_model(model, data, cfg, method_name="MultimodalCF-rho-Beliefs")
    device = torch.device(cfg.device)
    trained = trained.to(device)
    trained.eval()
    print(f"Multimodal CF trained in {time.time() - t0:.1f}s", flush=True)

    rng = np.random.RandomState(SEED)
    pos = [(u, i) for u, i, r in data.train if r >= 1.0]
    user_pos = defaultdict(set)
    for u, i, r in data.train:
        if r >= 1.0:
            user_pos[u].add(i)

    print(f"Sampling {n_samples} BPR triples...", flush=True)
    t0_list, xi_list = [], []
    for _ in range(n_samples):
        u, i_pos = pos[rng.randint(len(pos))]
        for _ in range(30):
            i_neg = rng.randint(data.n_items)
            if i_neg not in user_pos[u]:
                break
        else:
            continue
        with torch.no_grad():
            eu = trained.user_cf(torch.tensor([u], device=device))
            ep = trained.item_emb(torch.tensor([i_pos], device=device))
            en = trained.item_emb(torch.tensor([i_neg], device=device))
            form = trained.user_mlp(
                torch.tensor(data.user_features[u:u + 1], dtype=torch.float32, device=device)
            )
            t0_val = (eu * (ep - en)).sum().item()
            xi_val = (form * (ep - en)).sum().item()
        t0_list.append(t0_val)
        xi_list.append(xi_val)

    t0_arr = np.asarray(t0_list)
    xi_arr = np.asarray(xi_list)
    rho = float(np.corrcoef(t0_arr, xi_arr)[0, 1]) if len(t0_arr) >= 100 else float("nan")
    ci_lo, ci_hi = bootstrap_ci(t0_arr, xi_arr)
    print(f"Beliefs rho_xi_t0={rho:.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}] "
          f"(n={len(t0_arr)})", flush=True)

    # 按 t0 分 10 桶，计算每桶 E[xi|t0] 和 95% CI（与 conditional_xi_t0_stravl.json 一致）
    N_BUCKETS = 10
    edges = np.quantile(t0_arr, np.linspace(0, 1, N_BUCKETS + 1))
    buckets = []
    for b in range(N_BUCKETS):
        lo, hi = edges[b], edges[b + 1]
        if b < N_BUCKETS - 1:
            mask = (t0_arr >= lo) & (t0_arr < hi)
        else:
            mask = (t0_arr >= lo) & (t0_arr <= hi)
        if mask.sum() < 20:
            continue
        mean_xi = float(xi_arr[mask].mean())
        se = float(xi_arr[mask].std(ddof=1) / np.sqrt(mask.sum()))
        buckets.append({
            "bucket": b + 1,
            "t0_lo": float(lo),
            "t0_hi": float(hi),
            "n": int(mask.sum()),
            "mean_xi_given_t0": mean_xi,
            "ci95_mean_xi": [mean_xi - 1.96 * se, mean_xi + 1.96 * se],
        })
    max_abs_mean = max(abs(x["mean_xi_given_t0"]) for x in buckets) if buckets else float("nan")

    return {
        "dataset": "MovieLens-Beliefs",
        "n_samples": int(len(t0_arr)),
        "n_users": int(data.n_users),
        "n_items": int(data.n_items),
        "feat_dim": int(feat_dim),
        "rho_xi_t0": rho,
        "ci95": [ci_lo, ci_hi],
        "max_abs_bucket_mean_xi": max_abs_mean,
        "buckets": buckets,
        "protocol": "50k BPR triples after Multimodal CF training; rho=Corr(xi,t0) on profile vs collab margins; bootstrap 95% CI (1000 resamples)",
    }


def main():
    print("=" * 60)
    print("Computing rho_xi_t0 on MovieLens Beliefs dataset")
    print("=" * 60, flush=True)

    cfg = TrainConfig()
    print("Loading Beliefs dataset...", flush=True)
    t0 = time.time()
    data = load_movielens_beliefs()
    print(f"Beliefs loaded in {time.time() - t0:.1f}s", flush=True)

    result = compute_rho_beliefs(data, cfg)

    output = {
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "experiment": "Cross-domain rho_xi_t0 on MovieLens Beliefs (P1-2 补充)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [result],
    }

    for out_path in [OUT_ARFUSION, OUT_CEA]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
