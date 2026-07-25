"""补全 IntTravel 三分片上的 CV-CLER 和 MMSSL 基线完整 metrics。

目的：
1. 填补表19 IntTravel 行 MMSSL 列的空缺（原为"—"）
2. 填补表22 CV-CLER 行 P@10/R@10 的空缺（原为"—"）

数据集：IntTravel 三分片合并（n_shards=3, min_unique_pois=5, max_users=25000）
与表22 中 CLER/BPR/MMCF/ARFusion-Rec 同数据划分。

输出：
- D:\\tourism\\ARFusion_Research\\experiments\\results\\inttravel_cvcler_mmssl_results.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

# 禁用 PyTorch sparse tensor 警告
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from arfusion_recommender.arfusion_model import ARFusionModel, warm_start_from_cler
from arfusion_recommender.configs import INTTRAVEL_BEST
from arfusion_recommender.pipeline import (
    BPRDataset,
    CLERModel,
    TrainConfig,
    bpr_loss,
    set_seed,
    train_model,
    _prepare_model,
    load_inttravel_data,
    build_bipartite_adj,
)
from travel_recommender.paper_pipeline import MMSSLModel

SEED = 42
RESULTS_PATH = ROOT / "experiments" / "results" / "inttravel_cvcler_mmssl_results.json"


def evaluate_topk_batched(model, data, k=10, device="cuda", batch_size=500):
    """分批评估 Top-K 指标，避免一次性计算 [n_users, n_items] 矩阵导致内存爆炸。

    IntTravel 有 25000 用户 × 87612 物品，全量矩阵 8.76GB，需分批。
    """
    device_t = torch.device(device)
    model = _prepare_model(model, data, device_t)
    model.eval()

    test_users = list(data.test_relevant.items())
    n_users = len(test_users)
    user_ids = [uid for uid, _ in test_users]

    all_prec, all_rec, all_f1, all_ndcg = [], [], [], []

    with torch.no_grad():
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_user_ids = user_ids[start:end]
            batch_users_tensor = torch.tensor(batch_user_ids, device=device_t)

            # 计算当前批次的分数矩阵 [B, M]
            if hasattr(model, "all_scores_batch"):
                scores = model.all_scores_batch(batch_user_ids)
            elif isinstance(model, ARFusionModel):
                # ARFusionModel 的 collab 模式
                if model.score_mode == "collab":
                    u = model.collaborative_repr(batch_users_tensor)
                    v = model.item_emb.weight
                    scores = u @ v.T
                else:
                    # dual/additive 模式
                    lam = model.reliability(batch_users_tensor).squeeze(-1)
                    u_collab = model.collaborative_repr(batch_users_tensor)
                    u_prof = model.profile_repr(batch_users_tensor)
                    v = model.item_emb.weight
                    s_c = u_collab @ v.T
                    s_p = u_prof @ v.T
                    scores = lam.unsqueeze(-1) * s_c + (1.0 - lam).unsqueeze(-1) * s_p
            elif hasattr(model, "user_emb") and hasattr(model, "item_emb"):
                # CLERModel / MFModel
                u = model.user_emb(batch_users_tensor)
                v = model.item_emb.weight
                scores = u @ v.T
            elif hasattr(model, "user_repr") and hasattr(model, "item_emb"):
                # MultimodalCFModel / MMSSLModel
                u = model.user_repr(batch_users_tensor)
                v = model.item_emb.weight
                scores = u @ v.T
            else:
                # Fallback: per-user
                scores_list = []
                for uid in batch_user_ids:
                    scores_list.append(model.all_scores(uid))
                scores = torch.stack(scores_list)

            scores_np = scores.detach().cpu().numpy()

            for idx_in_batch, (global_idx, (user_id, relevant)) in enumerate(
                zip(range(start, end), test_users[start:end])
            ):
                s = scores_np[idx_in_batch].copy()
                mask_items = data.train_items.get(user_id, set()) | data.val_items.get(user_id, set())
                for item_id in mask_items:
                    s[item_id] = -1e9
                top_k = np.argpartition(-s, k)[:k]
                top_k = top_k[np.argsort(-s[top_k])]
                hits = len(set(top_k) & relevant)
                prec = hits / k
                rec = hits / len(relevant) if relevant else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                all_prec.append(prec)
                all_rec.append(rec)
                all_f1.append(f1)
                all_ndcg.append(ndcg)

            if (start // batch_size) % 5 == 0:
                print(f"  Eval progress: {end}/{n_users} users", flush=True)
                torch.cuda.empty_cache()

    return {
        "precision@10": float(np.mean(all_prec)),
        "recall@10": float(np.mean(all_rec)),
        "f1@10": float(np.mean(all_f1)),
        "ndcg@10": float(np.mean(all_ndcg)),
        "n_eval_users": n_users,
    }


def run_cvcler_inttravel(data, cfg, device):
    """CV-CLER = ARFusion(score_mode=collab) with pbd_weight=0, fuse_weight=0."""
    print(f"\n{'=' * 60}\n[IntTravel 3-shard] Training: CV-CLER\n{'=' * 60}", flush=True)
    set_seed(SEED)
    feat_dim = data.user_features.shape[1]

    # 1. CLER warm-start
    print("Step 1: CLER warm-start...", flush=True)
    cler = CLERModel(data.n_users, data.n_items, cfg.embed_dim, ui_temperature=cfg.ui_temperature).to(device)
    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(cler.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, patience_ctr, best_state = float("inf"), 0, None
    t0 = time.time()
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
                print(f"  CLER early stop at epoch {epoch+1}, val={v:.4f}", flush=True)
                break
        if (epoch + 1) % 5 == 0:
            print(f"  CLER epoch {epoch+1}: val={v:.4f} time={time.time()-t0:.1f}s", flush=True)
    if best_state:
        cler.load_state_dict(best_state)
        cler = cler.to(device)
    print(f"CLER warm-start done in {time.time()-t0:.1f}s, best_val={best_val:.4f}", flush=True)
    torch.cuda.empty_cache()

    # 2. CV-CLER = ARFusion with pbd=0, fuse=0
    print("Step 2: CV-CLER training...", flush=True)
    cvcler_cfg = TrainConfig(**{**cfg.__dict__, "pbd_weight": 0.0, "fuse_weight": 0.0})
    cvcler = ARFusionModel(
        data.n_users, data.n_items, feat_dim,
        dim=cfg.embed_dim, n_gnn_layers=cfg.n_gnn_layers,
        ui_temperature=cfg.ui_temperature, cv_temperature=cfg.cv_temperature,
        use_graph=cfg.use_graph, score_mode="collab",
    )
    warm_start_from_cler(cvcler, cler)
    t1 = time.time()
    trained, history = train_model(cvcler, data, cvcler_cfg, "CV-CLER")
    print(f"CV-CLER training done in {time.time()-t1:.1f}s, epochs={history.get('epochs')}", flush=True)

    # 3. Evaluate (batched)
    print("Step 3: Evaluating CV-CLER (batched)...", flush=True)
    t2 = time.time()
    metrics = evaluate_topk_batched(trained, data, k=10, device=device, batch_size=500)
    print(f"Evaluation done in {time.time()-t2:.1f}s", flush=True)
    m = metrics
    print(
        f"  [CV-CLER] P@10={m['precision@10']:.6f} R@10={m['recall@10']:.6f} "
        f"F1@10={m['f1@10']:.6f} NDCG@10={m['ndcg@10']:.6f}",
        flush=True,
    )
    return {
        "method": "CV-CLER",
        "metrics": metrics,
        "history": history,
        "train_time_sec": time.time() - t0,
    }


def run_mmssl_inttravel(data, cfg, device):
    """MMSSL baseline on IntTravel."""
    print(f"\n{'=' * 60}\n[IntTravel 3-shard] Training: MMSSL\n{'=' * 60}", flush=True)
    set_seed(SEED)
    feat_dim = data.user_features.shape[1]
    mmssl = MMSSLModel(
        data.n_users, data.n_items, feat_dim, cfg.embed_dim, n_layers=3,
    )
    mmssl = mmssl.to(device)
    mmssl.set_user_features(data.user_features)
    # MMSSL 继承自 LightGCNModel，需要设置邻接矩阵才能 propagate
    mmssl.set_adj(build_bipartite_adj(data, device))

    t0 = time.time()
    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(mmssl.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val, patience_ctr, best_state = float("inf"), 0, None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        mmssl.train()
        mmssl.clear_cache()
        train_losses = []
        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            optimizer.zero_grad()
            mmssl.propagate()
            loss = bpr_loss(mmssl.score(users, pos), mmssl.score(users, neg))
            loss = loss + 0.1 * mmssl.cross_modal_loss(users)
            loss.backward()
            optimizer.step()
            mmssl.clear_cache()
            train_losses.append(loss.item())

        mmssl.eval()
        with torch.no_grad():
            mmssl.propagate()
            val_losses = []
            for batch in val_loader:
                users, pos, neg = [x.to(device) for x in batch]
                val_losses.append(bpr_loss(mmssl.score(users, pos), mmssl.score(users, neg)).item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        print(f"  [MMSSL] Epoch {epoch+1}/{cfg.max_epochs} train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            best_state = {k: vc.cpu().clone() for k, vc in mmssl.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"  [MMSSL] Early stop at epoch {epoch+1}", flush=True)
                break

    if best_state is not None:
        mmssl.load_state_dict(best_state)
        mmssl = mmssl.to(device)
        mmssl.set_user_features(data.user_features)

    print("Evaluating MMSSL (batched)...", flush=True)
    t2 = time.time()
    with torch.no_grad():
        mmssl.propagate()
    metrics = evaluate_topk_batched(mmssl, data, k=10, device=device, batch_size=500)
    print(f"Evaluation done in {time.time()-t2:.1f}s", flush=True)
    elapsed = time.time() - t0
    m = metrics
    print(
        f"  [MMSSL] P@10={m['precision@10']:.6f} R@10={m['recall@10']:.6f} "
        f"F1@10={m['f1@10']:.6f} NDCG@10={m['ndcg@10']:.6f} time={elapsed:.1f}s",
        flush=True,
    )
    history["epochs"] = len(history["train_loss"])
    history["best_val_loss"] = best_val
    return {
        "method": "MMSSL",
        "metrics": metrics,
        "history": history,
        "train_time_sec": elapsed,
    }


def main():
    set_seed(SEED)
    data = load_inttravel_data(n_shards=3, min_unique_pois=5, max_users=25000)
    cfg = INTTRAVEL_BEST
    device = torch.device(cfg.device)
    print(f"Device: {cfg.device}", flush=True)
    print(f"IntTravel 3-shard stats: {data.stats}", flush=True)

    results = {}

    # 1. CV-CLER
    try:
        results["CV-CLER"] = run_cvcler_inttravel(data, cfg, device)
    except Exception as e:
        import traceback
        results["CV-CLER"] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"  [CV-CLER ERROR] {e}", flush=True)

    torch.cuda.empty_cache()

    # 2. MMSSL
    try:
        results["MMSSL"] = run_mmssl_inttravel(data, cfg, device)
    except Exception as e:
        import traceback
        results["MMSSL"] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"  [MMSSL ERROR] {e}", flush=True)

    output = {
        "experiment": "IntTravel 3-shard CV-CLER and MMSSL baselines (fill Table 19 & 22 gaps)",
        "date": "2026-07-24",
        "seed": SEED,
        "dataset": "IntTravel",
        "data_config": {"n_shards": 3, "min_unique_pois": 5, "max_users": 25000},
        "config": cfg.__dict__,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
