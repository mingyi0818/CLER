"""Equal-budget ablation: CV-CLER-scratch vs ARFusion-scratch vs warm-started variants.

Addresses reviewer concern: progressive training causes training budget mixing.
All variants use the SAME total number of gradient steps for fair comparison.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    load_stravl_data, train_model, evaluate_topk,
    TrainConfig, MFModel, CLERModel,
    BPRDataset, bpr_loss, set_seed,
)
from arfusion_recommender.arfusion_model import ARFusionModel, warm_start_from_cler
from torch.utils.data import DataLoader

RESULTS_PATH = str(ROOT / "results" / "equal_budget_ablation.json")


def train_cler_from_scratch(data, cfg, total_epochs):
    """Train CLER from random init for total_epochs (no warm start, no early stop)."""
    device = torch.device(cfg.device)
    cler = CLERModel(data.n_users, data.n_items, cfg.embed_dim, ui_temperature=cfg.ui_temperature).to(device)
    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(cler.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_state = float("inf"), None
    # Force total_epochs by setting patience very high; use val to pick best model
    cfg_forced = TrainConfig(**{**cfg.__dict__, "max_epochs": total_epochs, "patience": total_epochs + 10})
    for epoch in range(total_epochs):
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
            best_val = v
            best_state = {k: v_cpu.clone() for k, v_cpu in cler.state_dict().items()}
    if best_state:
        cler.load_state_dict(best_state)
        cler = cler.to(device)
    return cler


def train_arfusion_from_scratch(data, cfg, total_epochs):
    """Train ARFusion from random init for total_epochs (no CLER warm start)."""
    feat_dim = data.user_features.shape[1]
    arf = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode=cfg.score_mode,
    )
    # Random init (no warm start)
    cfg_forced = TrainConfig(**{**cfg.__dict__, "max_epochs": total_epochs, "patience": total_epochs + 10})
    trained, _ = train_model(arf, data, cfg_forced, "ARFusion_scratch")
    return trained


def save_incremental(results, total_epochs):
    """Save results incrementally to avoid data loss on interruption."""
    output = {
        "experiment": "Equal budget ablation (Stravl)",
        "date": "2026-07-24",
        "total_epochs_per_variant": total_epochs,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  [Incremental save] {list(results.keys())}", flush=True)


def main():
    set_seed(42)
    data = load_stravl_data()
    cfg = STRAVL_BEST
    print(f"Device: {cfg.device}", flush=True)

    TOTAL_EPOCHS = 32
    results = {}

    # Load existing partial results
    if Path(RESULTS_PATH).exists():
        try:
            prev = json.load(open(RESULTS_PATH, encoding="utf-8"))
            results = prev.get("results", {})
            print(f"Loaded {len(results)} previous results: {list(results.keys())}", flush=True)
        except Exception:
            pass

    # Variant 1: CV-CLER scratch (32 epochs from random init)
    if "CV-CLER_scratch" not in results:
        print(f"\n{'='*60}\n[EqualBudget] CV-CLER scratch ({TOTAL_EPOCHS} epochs)\n{'='*60}", flush=True)
        t0 = time.time()
        cler_scratch = train_cler_from_scratch(data, cfg, TOTAL_EPOCHS)
        m = evaluate_topk(cler_scratch, data, k=10, device=cfg.device)
        elapsed = time.time() - t0
        results["CV-CLER_scratch"] = {"metrics": m, "train_time_sec": elapsed}
        print(f"  NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s", flush=True)
        save_incremental(results, TOTAL_EPOCHS)

    # Variant 2: ARFusion scratch (32 epochs from random init)
    if "ARFusion_scratch" not in results:
        print(f"\n{'='*60}\n[EqualBudget] ARFusion scratch ({TOTAL_EPOCHS} epochs)\n{'='*60}", flush=True)
        t0 = time.time()
        arf_scratch = train_arfusion_from_scratch(data, cfg, TOTAL_EPOCHS)
        m = evaluate_topk(arf_scratch, data, k=10, device=cfg.device)
        elapsed = time.time() - t0
        results["ARFusion_scratch"] = {"metrics": m, "train_time_sec": elapsed}
        print(f"  NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s", flush=True)
        save_incremental(results, TOTAL_EPOCHS)

    # Variant 3: CLER warm-start + ARFusion (original progressive, total budget = 32)
    if "ARFusion_progressive" not in results:
        print(f"\n{'='*60}\n[EqualBudget] Progressive (CLER 13 + ARFusion 19 = 32)\n{'='*60}", flush=True)
        t0 = time.time()
        cler_warm = train_cler_from_scratch(data, cfg, 13)
        feat_dim = data.user_features.shape[1]
        arf_warm = ARFusionModel(
            data.n_users, data.n_items, feat_dim,
            dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
            ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
            use_graph=cfg.use_graph, score_mode=cfg.score_mode,
        )
        warm_start_from_cler(arf_warm, cler_warm)
        cfg_arf = TrainConfig(**{**cfg.__dict__, "max_epochs": 19, "patience": 19 + 10})
        arf_warm_trained, _ = train_model(arf_warm, data, cfg_arf, "ARFusion_warmstart")
        m = evaluate_topk(arf_warm_trained, data, k=10, device=cfg.device)
        elapsed = time.time() - t0
        results["ARFusion_progressive"] = {"metrics": m, "train_time_sec": elapsed}
        print(f"  NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s", flush=True)
        save_incremental(results, TOTAL_EPOCHS)

    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
