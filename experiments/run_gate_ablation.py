"""Gate component ablation (Table 14).

Trains ARFusion-Rec with different gate_mode variants on Stravl
in collab scoring mode, seed=42 single run.

Gate variants:
  - prior_only: lambda = sigma(alpha * log(1+c_u) - beta)
  - mlp_only:   lambda = sigma(MLP([e_u; h_prof; log(1+c_u)]))
  - full:       lambda = eta * mlp + (1-eta) * prior  (mixed)

Saves results to experiments/results/gate_ablation_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    BPRDataset,
    CLERModel,
    bpr_loss,
    evaluate_topk,
    load_stravl_data,
    set_seed,
    train_model,
)
from arfusion_recommender.arfusion_model import ARFusionModel, warm_start_from_cler

RESULTS_PATH = ROOT / "experiments" / "results" / "gate_ablation_results.json"
SEED = 42
GATE_VARIANTS = ["prior_only", "mlp_only", "full"]


def train_cler_for_warmstart(data, cfg, device):
    """Train CLER from scratch for warm-starting ARFusion."""
    print("\nTraining CLER for warm start...", flush=True)
    set_seed(SEED)
    cler = CLERModel(data.n_users, data.n_items, cfg.embed_dim, ui_temperature=cfg.ui_temperature)
    cler = cler.to(device)
    train_loader = DataLoader(
        BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
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
        v = float(np.mean(vl))
        if v < best_val:
            best_val, patience_ctr = v, 0
            best_state = {k: vc.cpu().clone() for k, vc in cler.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"  CLER early stop at epoch {epoch + 1}, best_val={best_val:.4f}", flush=True)
                break
    if best_state:
        cler.load_state_dict(best_state)
    print(f"  CLER trained, best_val={best_val:.4f}", flush=True)
    return cler


def main():
    set_seed(SEED)
    torch.cuda.empty_cache()
    data = load_stravl_data()
    cfg = STRAVL_BEST
    feat_dim = data.user_features.shape[1]
    device = torch.device(cfg.device)
    print(f"Device: {cfg.device}", flush=True)
    print(f"Dataset stats: {data.stats}", flush=True)
    print(f"Score mode: {cfg.score_mode}", flush=True)

    # Train CLER once for warm start (reused across variants)
    cler = train_cler_for_warmstart(data, cfg, device)

    all_results = {}

    for gate_mode in GATE_VARIANTS:
        print(f"\n{'=' * 70}\nGate variant: {gate_mode}\n{'=' * 70}", flush=True)
        set_seed(SEED)
        torch.cuda.empty_cache()

        arf = ARFusionModel(
            data.n_users,
            data.n_items,
            feat_dim,
            dim=cfg.embed_dim,
            n_gnn_layers=cfg.n_gnn_layers,
            ui_temperature=cfg.ui_temperature,
            cv_temperature=cfg.cv_temperature,
            use_graph=cfg.use_graph,
            score_mode=cfg.score_mode,  # collab mode for Stravl
            gate_mode=gate_mode,
        )
        warm_start_from_cler(arf, cler)

        t0 = time.time()
        trained, history = train_model(arf, data, cfg, method_name=f"ARFusion-{gate_mode}")
        elapsed = time.time() - t0

        metrics = evaluate_topk(trained, data, k=10, device=cfg.device)
        m = metrics
        print(
            f"  [ARFusion-{gate_mode}] P@10={m['precision@10']:.6f} "
            f"R@10={m['recall@10']:.6f} NDCG@10={m['ndcg@10']:.6f} time={elapsed:.1f}s",
            flush=True,
        )

        all_results[gate_mode] = {
            "gate_mode": gate_mode,
            "score_mode": cfg.score_mode,
            "metrics": {
                "precision@10": m["precision@10"],
                "recall@10": m["recall@10"],
                "f1@10": m["f1@10"],
                "ndcg@10": m["ndcg@10"],
                "n_eval_users": m["n_eval_users"],
            },
            "train_time_sec": elapsed,
            "epochs": history.get("epochs", -1),
            "best_val_loss": history.get("best_val_loss", -1),
        }

        # Save incrementally
        save_payload = {
            "experiment": "Gate component ablation (Stravl, collab mode, seed=42)",
            "date": "2026-07-24",
            "seed": SEED,
            "config": {
                "embed_dim": cfg.embed_dim,
                "n_gnn_layers": cfg.n_gnn_layers,
                "score_mode": cfg.score_mode,
                "cv_weight": cfg.cv_weight,
                "cl_weight": cfg.cl_weight,
                "pbd_weight": cfg.pbd_weight,
                "fuse_weight": cfg.fuse_weight,
                "max_epochs": cfg.max_epochs,
                "patience": cfg.patience,
            },
            "results": all_results,
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=2, ensure_ascii=False)
        print(f"  [Saved incremental] {RESULTS_PATH}", flush=True)

    # Print summary
    print(f"\n{'=' * 70}\nGate Component Ablation Summary\n{'=' * 70}", flush=True)
    print(f"{'Variant':<20} {'NDCG@10':<15} {'P@10':<15} {'R@10':<15}", flush=True)
    for gm in GATE_VARIANTS:
        r = all_results[gm]["metrics"]
        print(
            f"ARFusion-{gm:<12} {r['ndcg@10']:<15.4f} {r['precision@10']:<15.4f} {r['recall@10']:<15.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
