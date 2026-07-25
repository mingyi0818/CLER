"""Run Stravl experiments with 5 seeds and report mean±std."""
import sys, json, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    load_stravl_data, train_model, evaluate_topk,
    evaluate_topk_per_user, RecData, TrainConfig,
    MFModel, MultimodalCFModel, CLERModel,
    BPRDataset, bpr_loss, set_seed,
)
from arfusion_recommender.arfusion_model import (
    ARFusionModel, warm_start_from_cler,
)
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_PATH = str(ROOT / "results" / "stravl_5seed_results.json")

def run_single_seed(seed, data, cfg):
    set_seed(seed)
    feat_dim = data.user_features.shape[1]
    results = {}

    # BPR
    bpr = MFModel(data.n_users, data.n_items, cfg.embed_dim)
    trained, _ = train_model(bpr, data, cfg, "BPR")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["BPR"] = m

    # Multimodal CF
    mmcf = MultimodalCFModel(data.n_users, data.n_items, feat_dim, cfg.embed_dim)
    trained, _ = train_model(mmcf, data, cfg, "Multimodal CF")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["Multimodal CF"] = m

    # CLER
    cler = CLERModel(data.n_users, data.n_items, cfg.embed_dim, ui_temperature=cfg.ui_temperature)
    device = torch.device(cfg.device)
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
    m = evaluate_topk(cler, data, k=10, device=cfg.device)
    results["CLER"] = m

    # ARFusion-Rec with L_fuse fix
    arf = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode=cfg.score_mode,
    )
    warm_start_from_cler(arf, cler)
    trained, _ = train_model(arf, data, cfg, "ARFusion-Rec")
    m = evaluate_topk(trained, data, k=10, device=cfg.device)
    results["ARFusion-Rec"] = m

    return results

def main():
    data = load_stravl_data()
    cfg = STRAVL_BEST
    print(f"Device: {cfg.device}", flush=True)

    all_results = {}
    for seed in SEEDS:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}", flush=True)
        t0 = time.time()
        seed_results = run_single_seed(seed, data, cfg)
        elapsed = time.time() - t0
        print(f"Seed {seed} done in {elapsed:.1f}s", flush=True)
        for method, metrics in seed_results.items():
            print(f"  {method}: NDCG@10={metrics['ndcg@10']:.4f}", flush=True)
        all_results[str(seed)] = seed_results

    # Compute mean±std
    methods = ["BPR", "Multimodal CF", "CLER", "ARFusion-Rec"]
    summary = {}
    for method in methods:
        ndcgs = [all_results[str(s)][method]["ndcg@10"] for s in SEEDS]
        p10s = [all_results[str(s)][method]["precision@10"] for s in SEEDS]
        r10s = [all_results[str(s)][method]["recall@10"] for s in SEEDS]
        summary[method] = {
            "ndcg@10_mean": float(np.mean(ndcgs)),
            "ndcg@10_std": float(np.std(ndcgs)),
            "precision@10_mean": float(np.mean(p10s)),
            "precision@10_std": float(np.std(p10s)),
            "recall@10_mean": float(np.mean(r10s)),
            "recall@10_std": float(np.std(r10s)),
            "seeds": {str(s): all_results[str(s)][method] for s in SEEDS},
        }
        print(f"{method}: NDCG@10={np.mean(ndcgs):.4f}±{np.std(ndcgs):.4f}", flush=True)

    output = {
        "experiment": "Stravl 5-seed with L_fuse fix",
        "date": "2026-07-24",
        "seeds": SEEDS,
        "summary": summary,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)

if __name__ == "__main__":
    main()
