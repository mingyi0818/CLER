"""Static global lambda ablation: fixed lambda in dual mode vs learned gate."""
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

RESULTS_PATH = str(ROOT / "results" / "static_lambda_ablation.json")

def run_static_lambda(data, cfg, cler_model, lam_value):
    """Run ARFusion with fixed global lambda in dual mode."""
    feat_dim = data.user_features.shape[1]
    arf = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode="dual",
    )
    warm_start_from_cler(arf, cler_model)

    # Override reliability to return constant lambda
    def constant_reliability(users):
        n = users.shape[0]
        return torch.full((n, 1), lam_value, device=users.device)

    arf.reliability = constant_reliability

    cfg_dual = TrainConfig(**{**cfg.__dict__, "score_mode": "dual"})
    trained, _ = train_model(arf, data, cfg_dual, f"static_lambda_{lam_value}")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    return m

def main():
    set_seed(42)
    data = load_stravl_data()
    cfg = STRAVL_BEST
    device = torch.device(cfg.device)

    # Train CLER first (needed for warm start)
    print("Training CLER for warm start...", flush=True)
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
            best_state = {k: v.cpu().clone() for k, v in cler.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break
    if best_state:
        cler.load_state_dict(best_state)
        cler = cler.to(device)
    print("CLER trained.", flush=True)

    # Static lambda sweep
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    # Load existing partial results if any
    if Path(RESULTS_PATH).exists():
        try:
            prev = json.load(open(RESULTS_PATH, encoding="utf-8"))
            results = prev.get("results", {})
            print(f"Loaded {len(results)} previous results: {list(results.keys())}", flush=True)
        except Exception:
            pass

    for lam in lambdas:
        key = str(lam)
        if key in results:
            print(f"\nStatic lambda = {lam} (cached: NDCG@10={results[key]['ndcg@10']:.4f})", flush=True)
            continue
        print(f"\nStatic lambda = {lam}", flush=True)
        m = run_static_lambda(data, cfg, cler, lam)
        results[key] = m
        print(f"  NDCG@10={m['ndcg@10']:.4f}", flush=True)
        # Incremental save
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"experiment": "Static lambda ablation (Stravl)", "date": "2026-07-24", "results": results}, f, indent=2, ensure_ascii=False)

    # Learned gate (collab mode = original ARFusion)
    if "learned_collab" not in results:
        print("\nLearned gate (collab mode)...", flush=True)
        arf = ARFusionModel(
            data.n_users, data.n_items, data.user_features.shape[1],
            dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
            ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
            use_graph=cfg.use_graph, score_mode="collab",
        )
        warm_start_from_cler(arf, cler)
        trained, _ = train_model(arf, data, cfg, "ARFusion_learned_collab")
        m = evaluate_topk(trained, data, k=10, device=cfg.device)
        results["learned_collab"] = m
        print(f"  NDCG@10={m['ndcg@10']:.4f}", flush=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"experiment": "Static lambda ablation (Stravl)", "date": "2026-07-24", "results": results}, f, indent=2, ensure_ascii=False)

    # Learned gate (dual mode)
    if "learned_dual" not in results:
        print("\nLearned gate (dual mode)...", flush=True)
        arf2 = ARFusionModel(
            data.n_users, data.n_items, data.user_features.shape[1],
            dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
            ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
            use_graph=cfg.use_graph, score_mode="dual",
        )
        warm_start_from_cler(arf2, cler)
        cfg_dual = TrainConfig(**{**cfg.__dict__, "score_mode": "dual"})
        trained2, _ = train_model(arf2, data, cfg_dual, "ARFusion_learned_dual")
        m2 = evaluate_topk(trained2, data, k=10, device=cfg.device)
        results["learned_dual"] = m2
        print(f"  NDCG@10={m2['ndcg@10']:.4f}", flush=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"experiment": "Static lambda ablation (Stravl)", "date": "2026-07-24", "results": results}, f, indent=2, ensure_ascii=False)

    output = {
        "experiment": "Static lambda ablation (Stravl)",
        "date": "2026-07-24",
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
