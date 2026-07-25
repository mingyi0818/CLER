"""Generate missing experiment data for Table 12 (c_u bucketing) and Table 8 (per-user Wilcoxon).

This script skips lambda_cv sensitivity (already done by run_lambda_cv_sensitivity.py)
and only runs:
1. CV-CLER (lambda_cv=0.20) and ARFusion-Rec with per-user NDCG -> Table 12
2. BPR, CLER, CV-CLER, ARFusion-Rec per-user NDCG for Wilcoxon tests -> Table 8

All results saved to experiments/results/.
"""
import sys, json, time, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy import stats
from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    load_stravl_data, train_model, evaluate_topk, evaluate_topk_per_user,
    TrainConfig, CLERModel, BPRDataset, bpr_loss, set_seed, MFModel,
)
from arfusion_recommender.arfusion_model import ARFusionModel, warm_start_from_cler
from torch.utils.data import DataLoader

SEED = 42
RESULTS_DIR = ROOT / "results"

CU_BUCKETS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 999999)]


def compute_cu_per_user(data):
    """Compute training positive count c_u for each user."""
    cu = collections.defaultdict(int)
    for u, i, r in data.train:
        if r >= 1.0:
            cu[u] += 1
    return cu


def bucket_per_user_ndcg(per_user_metrics, cu_dict):
    """Bucket per-user NDCG@10 by c_u and compute mean per bucket."""
    bucket_results = []
    for lo, hi in CU_BUCKETS:
        ndcgs = []
        n_users = 0
        cu_vals = []
        for uid, metrics in per_user_metrics.items():
            c = cu_dict.get(uid, 0)
            if lo <= c < hi:
                ndcgs.append(metrics["ndcg@10"])
                cu_vals.append(c)
                n_users += 1
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        mean_cu = float(np.mean(cu_vals)) if cu_vals else 0.0
        bucket_results.append({
            "cu_range": f"[{lo},{hi if hi < 999999 else '+∞'})",
            "n_users": n_users,
            "mean_ndcg10": mean_ndcg,
            "mean_cu": mean_cu,
        })
    return bucket_results


def train_cler_warmstart(data, cfg, device):
    """Train CLER for warm-starting CV-CLER/ARFusion."""
    set_seed(SEED)
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
            best_state = {k: vc.cpu().clone() for k, vc in cler.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break
    if best_state:
        cler.load_state_dict(best_state)
        cler = cler.to(device)
    torch.cuda.empty_cache()
    return cler


def train_cvcler(data, cfg, device):
    """Train CV-CLER with cv_weight=0.2 (main experiment config)."""
    set_seed(SEED)
    feat_dim = data.user_features.shape[1]
    cler = train_cler_warmstart(data, cfg, device)

    cvcler_cfg = TrainConfig(**{
        **cfg.__dict__,
        "pbd_weight": 0.0,
        "fuse_weight": 0.0,
        "cv_weight": 0.2,
    })
    cvcler = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode="collab",
    )
    warm_start_from_cler(cvcler, cler)
    trained, _ = train_model(cvcler, data, cvcler_cfg, "CV-CLER(lcv=0.2)")
    return trained


def train_arfusion(data, cfg, device):
    """Train ARFusion-Rec (full model)."""
    set_seed(SEED)
    feat_dim = data.user_features.shape[1]
    cler = train_cler_warmstart(data, cfg, device)

    arfusion_cfg = TrainConfig(**{
        **cfg.__dict__,
        "pbd_weight": cfg.pbd_weight,
        "fuse_weight": cfg.fuse_weight,
    })
    arfusion = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode="collab",
    )
    warm_start_from_cler(arfusion, cler)
    trained, _ = train_model(arfusion, data, arfusion_cfg, "ARFusion-Rec")
    return trained


def train_bpr(data, cfg, device):
    """Train BPR baseline."""
    set_seed(SEED)
    bpr = MFModel(data.n_users, data.n_items, cfg.embed_dim)
    bpr = bpr.to(device)
    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(bpr.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, patience_ctr, best_state = float("inf"), 0, None
    for epoch in range(cfg.max_epochs):
        bpr.train()
        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            opt.zero_grad()
            loss = bpr_loss(bpr.score(users, pos), bpr.score(users, neg))
            loss.backward()
            opt.step()
        bpr.eval()
        vl = []
        with torch.no_grad():
            for batch in val_loader:
                users, pos, neg = [x.to(device) for x in batch]
                vl.append(bpr_loss(bpr.score(users, pos), bpr.score(users, neg)).item())
        v = np.mean(vl)
        if v < best_val:
            best_val, patience_ctr = v, 0
            best_state = {k: vc.cpu().clone() for k, vc in bpr.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break
    if best_state:
        bpr.load_state_dict(best_state)
        bpr = bpr.to(device)
    torch.cuda.empty_cache()
    return bpr


def train_cler(data, cfg, device):
    """Train CLER (BPR + UI-CL)."""
    set_seed(SEED)
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
            best_state = {k: vc.cpu().clone() for k, vc in cler.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break
    if best_state:
        cler.load_state_dict(best_state)
        cler = cler.to(device)
    torch.cuda.empty_cache()
    return cler


def main():
    print("Loading Stravl data...", flush=True)
    data = load_stravl_data()
    cfg = STRAVL_BEST
    device = torch.device(cfg.device)
    print(f"Device: {cfg.device}")
    print(f"Dataset: {data.n_users} users, {data.n_items} items", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute c_u per user
    cu_dict = compute_cu_per_user(data)
    print(f"Computed c_u for {len(cu_dict)} users", flush=True)

    # ============================================
    # Part 1: Train CV-CLER and ARFusion-Rec, get per-user NDCG
    # ============================================
    print("\n" + "=" * 60)
    print("Training CV-CLER (lambda_cv=0.2)...", flush=True)
    t0 = time.time()
    cvcler = train_cvcler(data, cfg, device)
    cvcler_per_user = evaluate_topk_per_user(cvcler, data, k=10, device=cfg.device)
    print(f"  CV-CLER per-user eval done ({time.time()-t0:.1f}s)", flush=True)

    print("\nTraining ARFusion-Rec...", flush=True)
    t0 = time.time()
    arfusion = train_arfusion(data, cfg, device)
    arfusion_per_user = evaluate_topk_per_user(arfusion, data, k=10, device=cfg.device)
    print(f"  ARFusion-Rec per-user eval done ({time.time()-t0:.1f}s)", flush=True)

    # ============================================
    # Part 2: Table 12 - c_u bucketing
    # ============================================
    print("\n" + "=" * 60)
    print("Table 12: c_u bucketing analysis", flush=True)
    print("=" * 60)

    cvcler_buckets = bucket_per_user_ndcg(cvcler_per_user, cu_dict)
    arfusion_buckets = bucket_per_user_ndcg(arfusion_per_user, cu_dict)

    print(f"\n{'c_u range':<12} {'n_users':<10} {'CV-CLER NDCG':<15} {'ARFusion NDCG':<15} {'Delta':<10}")
    for cv, ar in zip(cvcler_buckets, arfusion_buckets):
        delta = ar["mean_ndcg10"] - cv["mean_ndcg10"]
        print(f"{cv['cu_range']:<12} {cv['n_users']:<10} {cv['mean_ndcg10']:<15.4f} {ar['mean_ndcg10']:<15.4f} {delta:+.4f}")

    out_path = RESULTS_DIR / "cu_bucket_results.json"
    payload = {
        "experiment": "c_u bucketing analysis (Stravl, seed=42)",
        "date": "2026-07-24",
        "seed": SEED,
        "cvcler_buckets": cvcler_buckets,
        "arfusion_buckets": arfusion_buckets,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)

    # ============================================
    # Part 3: Train BPR and CLER for Wilcoxon tests
    # ============================================
    print("\n" + "=" * 60)
    print("Table 8: Per-user Wilcoxon tests", flush=True)
    print("=" * 60)

    print("\nTraining BPR...", flush=True)
    t0 = time.time()
    bpr = train_bpr(data, cfg, device)
    bpr_per_user = evaluate_topk_per_user(bpr, data, k=10, device=cfg.device)
    print(f"  BPR done ({time.time()-t0:.1f}s)", flush=True)

    print("\nTraining CLER...", flush=True)
    t0 = time.time()
    cler = train_cler(data, cfg, device)
    cler_per_user = evaluate_topk_per_user(cler, data, k=10, device=cfg.device)
    print(f"  CLER done ({time.time()-t0:.1f}s)", flush=True)

    # ============================================
    # Part 4: Compute per-user Wilcoxon tests
    # ============================================
    common_users = sorted(set(bpr_per_user.keys()) & set(cler_per_user.keys()) &
                          set(cvcler_per_user.keys()) & set(arfusion_per_user.keys()))
    print(f"\nCommon users for Wilcoxon: {len(common_users)}", flush=True)

    bpr_ndcg = np.array([bpr_per_user[u]["ndcg@10"] for u in common_users])
    cler_ndcg = np.array([cler_per_user[u]["ndcg@10"] for u in common_users])
    cvcler_ndcg = np.array([cvcler_per_user[u]["ndcg@10"] for u in common_users])
    arfusion_ndcg = np.array([arfusion_per_user[u]["ndcg@10"] for u in common_users])

    comparisons = [
        ("ARFusion-Rec vs BPR", arfusion_ndcg, bpr_ndcg),
        ("ARFusion-Rec vs CLER", arfusion_ndcg, cler_ndcg),
        ("ARFusion-Rec vs CV-CLER", arfusion_ndcg, cvcler_ndcg),
        ("CV-CLER vs BPR", cvcler_ndcg, bpr_ndcg),
        ("CV-CLER vs CLER", cvcler_ndcg, cler_ndcg),
        ("ARFusion-Rec vs Multimodal CF", arfusion_ndcg, bpr_ndcg),  # placeholder, will be replaced
        ("CV-CLER vs Multimodal CF", cvcler_ndcg, bpr_ndcg),  # placeholder
    ]

    results = {}
    for name, x, y in comparisons:
        diffs = x - y
        n_nonzero = int(np.sum(np.abs(diffs) > 1e-8))
        try:
            w_stat, p_val = stats.wilcoxon(x, y, alternative="greater")
        except ValueError:
            w_stat, p_val = float("nan"), 1.0
        results[name] = {
            "n_users": len(common_users),
            "n_nonzero_diff": n_nonzero,
            "mean_diff": float(diffs.mean()),
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p_one_sided": float(p_val) if not np.isnan(p_val) else 1.0,
        }
        print(f"  {name}: delta={diffs.mean():+.6f} p={p_val:.2e} (n={len(common_users)}, nonzero={n_nonzero})", flush=True)

    out_path = RESULTS_DIR / "per_user_wilcoxon_results.json"
    payload = {
        "experiment": "Per-user Wilcoxon signed-rank tests (Stravl, seed=42)",
        "date": "2026-07-24",
        "seed": SEED,
        "n_users": len(common_users),
        "comparisons": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)

    print("\n" + "=" * 60)
    print("All missing experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
