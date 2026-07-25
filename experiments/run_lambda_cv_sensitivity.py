"""Run lambda_cv sensitivity scan for Table 18."""
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

SEED = 42
RESULTS_DIR = ROOT / "results"
LAMBDA_CV_VALUES = [0.05, 0.10, 0.20, 0.30]


def train_cler_warmstart(data, cfg, device):
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
    print(f"Device: {cfg.device}, Dataset: {data.n_users} users, {data.n_items} items", flush=True)

    results = {}
    for lcv in LAMBDA_CV_VALUES:
        t0 = time.time()
        print(f"\n--- Training CV-CLER with lambda_cv={lcv} ---", flush=True)
        set_seed(SEED)
        feat_dim = data.user_features.shape[1]
        cler = train_cler_warmstart(data, cfg, device)
        print(f"  CLER warmstart done ({time.time()-t0:.1f}s)", flush=True)

        cvcler_cfg = TrainConfig(**{
            **cfg.__dict__,
            "pbd_weight": 0.0,
            "fuse_weight": 0.0,
            "cv_weight": lcv,
        })
        cvcler = ARFusionModel(
            data.n_users, data.n_items, feat_dim,
            dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
            ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
            use_graph=cfg.use_graph, score_mode="collab",
        )
        warm_start_from_cler(cvcler, cler)
        t1 = time.time()
        trained, _ = train_model(cvcler, data, cvcler_cfg, f"CV-CLER(lcv={lcv})")
        print(f"  CV-CLER training done ({time.time()-t1:.1f}s)", flush=True)

        m = evaluate_topk(trained, data, k=10, device=cfg.device)
        elapsed = time.time() - t0
        results[f"lambda_cv_{lcv}"] = {
            "lambda_cv": lcv,
            "metrics": {k: float(v) for k, v in m.items()},
            "train_time_sec": elapsed,
        }
        print(f"  lambda_cv={lcv}: P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} NDCG@10={m['ndcg@10']:.4f} ({elapsed:.1f}s)", flush=True)

        # Save incrementally
        out_path = RESULTS_DIR / "cv_lambda_sensitivity_results.json"
        payload = {
            "experiment": "CV-CLER lambda_cv sensitivity (Stravl, seed=42)",
            "date": "2026-07-24",
            "seed": SEED,
            "lambda_cv_values": LAMBDA_CV_VALUES,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  [Saved] {out_path}", flush=True)

    print("\n=== lambda_cv sensitivity scan complete ===", flush=True)


if __name__ == "__main__":
    main()
