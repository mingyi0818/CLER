"""DropoutNet baseline: dropout-based profile fusion for cold-start recommendation.

Reference: Volkovs et al. 2017, "DropoutNet: Addressing Cold Start in Recommender Systems"
Adapted for warm-start Stravl scenario with profile features.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from arfusion_recommender.configs import STRAVL_BEST
from arfusion_recommender.pipeline import (
    load_stravl_data, evaluate_topk,
    TrainConfig, BPRDataset, bpr_loss, set_seed, RESULTS_DIR,
)

RESULTS_PATH = str(ROOT / "results" / "dropoutnet_results.json")


class DropoutNet(nn.Module):
    """DropoutNet:MF + MLP with input dropout on profile and behavior embeddings.

    During training, randomly drop the entire profile or behavior embedding
    to simulate cold-start and force the model to handle missing modalities.
    """
    def __init__(self, n_users, n_items, feat_dim, embed_dim=64, dropout_p=0.5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.profile_encoder = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Fusion MLP: concatenate [user_emb, profile_emb] -> score projection
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dropout_p = dropout_p
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def set_user_features(self, features):
        device = self.user_emb.weight.device
        self._user_features = torch.tensor(features, dtype=torch.float32, device=device)

    def forward(self, users, items, training=True):
        e_u = self.user_emb(users)
        p_u = self.profile_encoder(self._user_features[users])
        if training:
            # Dropout: randomly zero out entire modality
            mask_p = (torch.rand(users.shape[0], 1, device=users.device) > self.dropout_p).float()
            mask_e = (torch.rand(users.shape[0], 1, device=users.device) > self.dropout_p).float()
            p_u = p_u * mask_p
            e_u = e_u * mask_e
        h_u = self.fusion(torch.cat([e_u, p_u], dim=1))
        v_i = self.item_emb(items)
        return (h_u * v_i).sum(dim=1)

    def score(self, users, items):
        return self.forward(users, items, training=False)

    def user_repr(self, users):
        """Return fused user representation for batched evaluation."""
        e_u = self.user_emb(users)
        p_u = self.profile_encoder(self._user_features[users])
        return self.fusion(torch.cat([e_u, p_u], dim=1))


def train_dropoutnet(model, data, cfg):
    """Train DropoutNet with BPR loss + modality dropout."""
    device = torch.device(cfg.device)
    model = model.to(device)
    model.set_user_features(data.user_features)
    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, patience_ctr, best_state = float("inf"), 0, None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            opt.zero_grad()
            s_pos = model.forward(users, pos, training=True)
            s_neg = model.forward(users, neg, training=True)
            loss = bpr_loss(s_pos, s_neg)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                users, pos, neg = [x.to(device) for x in batch]
                val_losses.append(bpr_loss(model.score(users, pos), model.score(users, neg)).item())
        v = float(np.mean(val_losses)) if val_losses else 0.0
        history["val_loss"].append(v)
        print(f"  [DropoutNet] Epoch {epoch+1}/{cfg.max_epochs} train={np.mean(train_losses):.4f} val={v:.4f}", flush=True)
        if v < best_val:
            best_val = v
            patience_ctr = 0
            best_state = {k: v_cpu.cpu().clone() for k, v_cpu in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"  [DropoutNet] Early stop at epoch {epoch+1}", flush=True)
                break
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model, history


def main():
    set_seed(42)
    data = load_stravl_data()
    cfg = STRAVL_BEST
    feat_dim = data.user_features.shape[1]
    print(f"Device: {cfg.device}", flush=True)

    results = {}
    # Try multiple dropout rates
    for drop_p in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}\n[DropoutNet] dropout_p={drop_p}\n{'='*60}", flush=True)
        t0 = time.time()
        model = DropoutNet(data.n_users, data.n_items, feat_dim, cfg.embed_dim, dropout_p=drop_p)
        trained, history = train_dropoutnet(model, data, cfg)
        m = evaluate_topk(trained, data, k=10, device=cfg.device)
        elapsed = time.time() - t0
        results[f"dropout_{drop_p}"] = {
            "metrics": m,
            "train_time_sec": elapsed,
            "history": history,
        }
        print(f"  P@10={m['precision@10']:.4f} R@10={m['recall@10']:.4f} "
              f"NDCG@10={m['ndcg@10']:.4f} time={elapsed:.1f}s", flush=True)

    # Pick best by NDCG@10
    best_drop = max(results.keys(), key=lambda k: results[k]["metrics"]["ndcg@10"])
    print(f"\nBest dropout: {best_drop} NDCG@10={results[best_drop]['metrics']['ndcg@10']:.4f}", flush=True)

    output = {
        "experiment": "DropoutNet baseline (Stravl)",
        "date": "2026-07-24",
        "best_variant": best_drop,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
