"""Unified data loading, models, training and evaluation for paper experiments."""

from __future__ import annotations

import ast
import json
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Portable path setup: data/ at repo root, cache/ under experiments/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = str(_REPO_ROOT / "data" / "Stravl_Travel_Preference_Data.csv")
OUTPUT_DIR = str(_REPO_ROOT / "experiments" / "cache")
CACHE_PATH = str(_REPO_ROOT / "experiments" / "cache" / "rec_data_cache_v2.pkl")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_list_string(value) -> List:
    if pd.isna(value) or value == "[]":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


@dataclass
class RecData:
    n_users: int
    n_items: int
    train: List[Tuple[int, int, float]]
    val: List[Tuple[int, int, float]]
    test: List[Tuple[int, int, float]]
    user_features: np.ndarray
    train_items: Dict[int, Set[int]]
    val_items: Dict[int, Set[int]]
    test_relevant: Dict[int, Set[int]]
    stats: Dict = field(default_factory=dict)


def load_rec_data(data_path: str = DATA_PATH, use_cache: bool = True) -> RecData:
    if use_cache and os.path.exists(CACHE_PATH):
        mtime_data = os.path.getmtime(data_path)
        mtime_cache = os.path.getmtime(CACHE_PATH)
        if mtime_cache >= mtime_data:
            _log("Loading cached RecData...")
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)

    _log("Parsing CSV (first run, will cache)...")
    df = pd.read_csv(data_path)
    n_users = len(df)

    yes_lists = df["yes_swipes"].map(parse_list_string).tolist()
    maybe_lists = df["maybe_swipes"].map(parse_list_string).tolist()
    no_lists = df["no_swipes"].map(parse_list_string).tolist()

    interactions: List[Tuple[int, int, float]] = []
    dest_set: Set[int] = set()

    for user_idx in range(n_users):
        for dest_id in yes_lists[user_idx]:
            interactions.append((user_idx, dest_id, 1.0))
            dest_set.add(dest_id)
        for dest_id in maybe_lists[user_idx]:
            interactions.append((user_idx, dest_id, 0.5))
            dest_set.add(dest_id)
        for dest_id in no_lists[user_idx]:
            interactions.append((user_idx, dest_id, 0.0))
            dest_set.add(dest_id)

    dest_to_idx = {d: i for i, d in enumerate(sorted(dest_set))}
    n_items = len(dest_to_idx)

    mapped: List[Tuple[int, int, float]] = [
        (u, dest_to_idx[d], r) for u, d, r in interactions if d in dest_to_idx
    ]

    user_features = _extract_form_features(df)

    by_user: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for u, i, r in mapped:
        by_user[u].append((u, i, r))

    rng = np.random.RandomState(SEED)
    train, val, test = [], [], []

    for user_id, user_inters in by_user.items():
        rng.shuffle(user_inters)
        n = len(user_inters)
        if n == 1:
            train.extend(user_inters)
            continue
        n_train = max(1, int(n * 0.8))
        n_val = max(0, int(n * 0.1))
        n_test = n - n_train - n_val
        if n_test == 0 and n_val > 0:
            n_test = 1
            n_val -= 1
        elif n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)

        train.extend(user_inters[:n_train])
        val.extend(user_inters[n_train : n_train + n_val])
        test.extend(user_inters[n_train + n_val :])

    train_items = _items_by_user(train)
    val_items = _items_by_user(val)
    test_relevant = {
        u: {i for uu, i, r in test if uu == u and r >= 1.0}
        for u in set(u for u, _, _ in test)
    }
    test_relevant = {u: items for u, items in test_relevant.items() if items}

    pos_train = sum(1 for _, _, r in train if r >= 1.0)
    pos_test = sum(len(v) for v in test_relevant.values())

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": len(mapped),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_positive_train": pos_train,
        "n_positive_test": pos_test,
        "n_test_users": len(test_relevant),
    }

    result = RecData(
        n_users=n_users,
        n_items=n_items,
        train=train,
        val=val,
        test=test,
        user_features=user_features,
        train_items=train_items,
        val_items=val_items,
        test_relevant=test_relevant,
        stats=stats,
    )

    if use_cache:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(result, f)
        _log(f"RecData cached to {CACHE_PATH}")

    return result


def corrupt_user_features(features: np.ndarray, noise_ratio: float, seed: int = SEED) -> np.ndarray:
    """Destroy profile–behavior alignment by permuting a fraction of user feature rows."""
    if noise_ratio <= 0.0:
        return features.copy()
    out = features.copy()
    n = len(out)
    n_swap = max(1, int(round(n * noise_ratio)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    out[:n_swap] = features[perm[:n_swap]]
    return out


def with_corrupted_features(data: RecData, noise_ratio: float, seed: int = SEED) -> RecData:
    """Return a shallow copy of RecData with permuted user_features."""
    import copy

    corrupted = copy.copy(data)
    corrupted.user_features = corrupt_user_features(data.user_features, noise_ratio, seed=seed)
    return corrupted


def _items_by_user(rows: List[Tuple[int, int, float]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for u, i, _ in rows:
        out[u].add(i)
    return dict(out)


FORM_FIELD_SIZES = {
    "form_a": 4,
    "form_b": 4,
    "form_c": 4,
    "form_f": 8,
    "form_g": 8,
    "form_h": 3,
    "form_i": 3,
    "form_j": 3,
    "form_k": 8,
}


def _parse_form_multihot(value, num_classes: int) -> np.ndarray:
    out = np.zeros(num_classes, dtype=np.float32)
    if pd.isna(value) or value == "[]":
        return out
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(parsed, list):
            return out
        for item in parsed:
            idx = int(item)
            if 0 <= idx < num_classes:
                out[idx] = 1.0
    except (ValueError, SyntaxError, TypeError):
        pass
    return out


def _extract_form_features(df: pd.DataFrame) -> np.ndarray:
    """Multi-hot encoded user form features (declarative preference view)."""
    rows = []
    for _, row in df.iterrows():
        parts = [_parse_form_multihot(row[col], size) for col, size in FORM_FIELD_SIZES.items()]
        rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=np.float32)


class BPRDataset(Dataset):
    def __init__(self, data: RecData, split: str = "train", num_neg: int = 1):
        self.data = data
        self.num_neg = num_neg
        rows = {"train": data.train, "val": data.val, "test": data.test}[split]
        self.pos_pairs = [(u, i) for u, i, r in rows if r >= 1.0]
        self.all_items = list(range(data.n_items))
        self.user_pos = defaultdict(set)
        for u, i, r in rows:
            if r >= 1.0:
                self.user_pos[u].add(i)

    def __len__(self) -> int:
        return max(len(self.pos_pairs), 1)

    def __getitem__(self, idx: int):
        u, i = self.pos_pairs[idx % len(self.pos_pairs)]
        for _ in range(20):
            j = random.choice(self.all_items)
            if j not in self.user_pos[u]:
                return u, i, j
        j = random.choice(self.all_items)
        return u, i, j


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def embedding_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    if not x.requires_grad:
        return x
    mask = (torch.rand_like(x) > drop_prob).float()
    return x * mask / (1.0 - drop_prob)


def build_bipartite_adj(data: RecData, device: torch.device) -> torch.Tensor:
    """Normalized user-item adjacency matrix (n_users x n_items)."""
    rows, cols, vals = [], [], []
    for u, i, r in data.train:
        if r >= 1.0:
            rows.append(u)
            cols.append(i)
            vals.append(1.0)

    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0),
            (data.n_users, data.n_items),
            device=device,
        ).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (data.n_users, data.n_items), device=device).coalesce()

    user_deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    item_deg = torch.sparse.sum(adj, dim=0).to_dense().clamp(min=1.0)
    row, col = adj.indices()
    norm_vals = adj.values() * user_deg[row].pow(-0.5) * item_deg[col].pow(-0.5)
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape, device=device).coalesce()


def build_train_adj(data: RecData, device: torch.device) -> torch.Tensor:
    return build_bipartite_adj(data, device)


def sparse_mm(adj: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(adj, emb)


class MFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int) -> torch.Tensor:
        u = self.user_emb.weight[user_id]
        return self.item_emb.weight @ u


class LightGCNModel(nn.Module):
    """LightGCN-style propagation on bipartite user-item graph."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.adj: Optional[torch.Tensor] = None
        self._cached_user = None
        self._cached_item = None

    def set_adj(self, adj: torch.Tensor) -> None:
        self.adj = adj

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.adj is not None
        user_layers = [self.user_emb.weight]
        item_layers = [self.item_emb.weight]
        u, v = self.user_emb.weight, self.item_emb.weight
        for _ in range(self.n_layers):
            u = sparse_mm(self.adj, v)
            v = sparse_mm(self.adj.transpose(0, 1), u)
            user_layers.append(u)
            item_layers.append(v)
        user_out = torch.stack(user_layers, dim=0).mean(dim=0)
        item_out = torch.stack(item_layers, dim=0).mean(dim=0)
        self._cached_user = user_out
        self._cached_item = item_out
        return user_out, item_out

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_cached_user") and self._cached_user is not None:
            return (self._cached_user[users] * self._cached_item[items]).sum(dim=1)
        user_out, item_out = self.propagate()
        return (user_out[users] * item_out[items]).sum(dim=1)

    def all_scores(self, user_id: int) -> torch.Tensor:
        if hasattr(self, "_cached_user") and self._cached_user is not None:
            return self._cached_item @ self._cached_user[user_id]
        user_out, item_out = self.propagate()
        return item_out @ user_out[user_id]

    def clear_cache(self) -> None:
        self._cached_user = None
        self._cached_item = None


class SimGCLModel(LightGCNModel):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3, noise_scale: float = 0.1):
        super().__init__(n_users, n_items, dim, n_layers)
        self.noise_scale = noise_scale

    def contrastive_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_out, item_out = self.propagate()
        batch_emb = torch.cat([user_out[users], item_out[items]], dim=0)
        view1 = batch_emb + torch.randn_like(batch_emb) * self.noise_scale
        view2 = batch_emb + torch.randn_like(batch_emb) * self.noise_scale
        return info_nce(view1, view2)


class LightGCLModel(LightGCNModel):
    """SVD-based graph augmentation contrastive learning (Cai et al., ICLR 2023)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        n_layers: int = 3,
        svd_rank: int = 64,
        svd_noise_scale: float = 0.1,
        temperature: float = 0.2,
    ):
        super().__init__(n_users, n_items, dim, n_layers)
        self.svd_rank = min(svd_rank, min(n_users, n_items))
        self.svd_noise_scale = svd_noise_scale
        self.temperature = temperature
        self.adj_aug1: Optional[torch.Tensor] = None
        self.adj_aug2: Optional[torch.Tensor] = None

    @staticmethod
    def _normalize_ui_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = adj.coalesce()
        user_deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        item_deg = torch.sparse.sum(adj, dim=0).to_dense().clamp(min=1.0)
        row, col = adj.indices()
        norm_vals = adj.values() * user_deg[row].pow(-0.5) * item_deg[col].pow(-0.5)
        return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape, device=adj.device).coalesce()

    def _svd_augment_adj(self, adj: torch.Tensor, seed: int) -> torch.Tensor:
        adj = adj.coalesce()
        ui_dense = adj.to_dense()
        q = min(self.svd_rank, min(ui_dense.shape) - 1)
        if q < 2:
            return adj
        torch.manual_seed(seed)
        U, S, V = torch.svd_lowrank(ui_dense, q=q, niter=4)
        scale = 1.0 + self.svd_noise_scale * torch.randn(S.size(0), device=S.device)
        R_aug = (U * (S * scale).unsqueeze(0)) @ V.T
        support = ui_dense > 0
        R_aug = R_aug.abs() * support
        rows, cols = support.nonzero(as_tuple=True)
        vals = R_aug[rows, cols]
        aug = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            adj.shape,
            device=adj.device,
        ).coalesce()
        return self._normalize_ui_adj(aug)

    def set_adj(self, adj: torch.Tensor) -> None:
        self.adj = adj
        self.adj_aug1 = self._svd_augment_adj(adj, SEED)
        self.adj_aug2 = self._svd_augment_adj(adj, SEED + 1)

    def _propagate_with_adj(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self.user_emb.weight]
        item_layers = [self.item_emb.weight]
        u, v = self.user_emb.weight, self.item_emb.weight
        for _ in range(self.n_layers):
            u = sparse_mm(adj, v)
            v = sparse_mm(adj.transpose(0, 1), u)
            user_layers.append(u)
            item_layers.append(v)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(dim=0)

    def contrastive_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        assert self.adj_aug1 is not None and self.adj_aug2 is not None
        u1, i1 = self._propagate_with_adj(self.adj_aug1)
        u2, i2 = self._propagate_with_adj(self.adj_aug2)
        view1 = torch.cat([u1[users], i1[items]], dim=0)
        view2 = torch.cat([u2[users], i2[items]], dim=0)
        return info_nce(view1, view2, self.temperature)


def _dropout_adj(adj: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
    """Random edge dropout for subgraph contrastive views (SGL-style)."""
    if adj._nnz() == 0:
        return adj
    mask = torch.rand(adj.values().size(0), device=adj.device) > drop_rate
    if mask.sum() == 0:
        return adj
    return torch.sparse_coo_tensor(
        adj.indices()[:, mask],
        adj.values()[mask],
        adj.shape,
        device=adj.device,
    ).coalesce()


class SGLModel(LightGCNModel):
    """Subgraph contrastive learning on bipartite graph (Wu et al., WWW 2021)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        n_layers: int = 3,
        drop_rate: float = 0.1,
        temperature: float = 0.2,
    ):
        super().__init__(n_users, n_items, dim, n_layers)
        self.drop_rate = drop_rate
        self.temperature = temperature

    def _propagate_with_adj(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self.user_emb.weight]
        item_layers = [self.item_emb.weight]
        u, v = self.user_emb.weight, self.item_emb.weight
        for _ in range(self.n_layers):
            u = sparse_mm(adj, v)
            v = sparse_mm(adj.transpose(0, 1), u)
            user_layers.append(u)
            item_layers.append(v)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(dim=0)

    def contrastive_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        assert self.adj is not None
        u1, i1 = self._propagate_with_adj(_dropout_adj(self.adj, self.drop_rate))
        u2, i2 = self._propagate_with_adj(_dropout_adj(self.adj, self.drop_rate))
        view1 = torch.cat([u1[users], i1[items]], dim=0)
        view2 = torch.cat([u2[users], i2[items]], dim=0)
        return info_nce(view1, view2, self.temperature)


class CL4SRecModel(MFModel):
    """Augmented contrastive learning adapted from CL4SRec (Liu et al., SIGIR 2021)."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, temperature: float = 0.2):
        super().__init__(n_users, n_items, dim)
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def contrastive_loss(self, users: torch.Tensor, items: torch.Tensor, drop_prob: float = 0.2) -> torch.Tensor:
        u = self.user_emb(users)
        v = self.item_emb(items)
        z_u1 = self.projector(embedding_dropout(u, drop_prob))
        z_u2 = self.projector(embedding_dropout(u, drop_prob))
        z_i1 = self.projector(embedding_dropout(v, drop_prob))
        z_i2 = self.projector(embedding_dropout(v, drop_prob))
        return 0.5 * (info_nce(z_u1, z_u2, self.temperature) + info_nce(z_i1, z_i2, self.temperature))


class MMSSLModel(LightGCNModel):
    """Multi-modal self-supervised learning with graph behavior + declarative form (Wei et al., WWW 2023)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        feat_dim: int,
        dim: int = 64,
        n_layers: int = 3,
        temperature: float = 0.2,
    ):
        super().__init__(n_users, n_items, dim, n_layers)
        self.temperature = temperature
        self.form_encoder = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim),
        )
        self.form_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.behavior_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.register_buffer("user_features", torch.zeros(n_users, feat_dim))

    def set_user_features(self, features: np.ndarray) -> None:
        device = self.user_emb.weight.device
        self.user_features = torch.tensor(features, dtype=torch.float32, device=device)

    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        self.propagate()
        assert self._cached_user is not None
        form = self.form_encoder(self.user_features[users.to(self.user_emb.weight.device)])
        return self._cached_user[users] + form

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_repr(users) * self._cached_item[items]).sum(dim=1)

    def all_scores(self, user_id: int) -> torch.Tensor:
        self.propagate()
        device = self.user_emb.weight.device
        u = self.user_repr(torch.tensor([user_id], device=device))[0]
        return self._cached_item @ u

    def cross_modal_loss(self, users: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        self.propagate()
        assert self._cached_user is not None
        z_form = self.form_projector(embedding_dropout(self.form_encoder(self.user_features[users]), drop_prob))
        z_beh = self.behavior_projector(embedding_dropout(self._cached_user[users], drop_prob))
        return 0.5 * (
            info_nce(z_form, z_beh, self.temperature) + info_nce(z_beh, z_form, self.temperature)
        )


class MMSSLDecoupledModel(MMSSLModel):
    """MMSSL-style cross-modal training with behavior-only scoring at inference (ablation)."""

    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        self.propagate()
        assert self._cached_user is not None
        return self._cached_user[users]


class MultimodalCFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, feat_dim: int, dim: int = 64):
        super().__init__()
        self.user_cf = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, dim),
        )
        nn.init.normal_(self.user_cf.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.register_buffer("user_features", torch.zeros(n_users, feat_dim))

    def set_user_features(self, features: np.ndarray) -> None:
        device = self.user_cf.weight.device
        self.user_features = torch.tensor(features, dtype=torch.float32, device=device)

    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        users = users.to(self.user_cf.weight.device)
        return self.user_cf(users) + self.user_mlp(self.user_features[users])

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_repr(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int) -> torch.Tensor:
        device = self.user_cf.weight.device
        u = self.user_repr(torch.tensor([user_id], device=device))[0]
        return self.item_emb.weight @ u


class CLERModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, use_cl: bool = True, ui_temperature: float = 0.2):
        super().__init__()
        self.use_cl = use_cl
        self.ui_temperature = ui_temperature
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int) -> torch.Tensor:
        u = self.user_emb.weight[user_id]
        return self.item_emb.weight @ u

    def contrastive_loss(self, users: torch.Tensor, items: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        if not self.use_cl:
            return torch.tensor(0.0, device=users.device)
        u = self.user_emb(users)
        v = self.item_emb(items)
        z1 = self.projection(embedding_dropout(u, drop_prob))
        z2 = self.projection(embedding_dropout(v, drop_prob))
        return info_nce(z1, z2, self.ui_temperature)


class CVCLERModel(nn.Module):
    """Cross-view CLER: form view (declarative) vs behavior view (interaction)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        feat_dim: int,
        dim: int = 64,
        use_cross_view: bool = True,
        use_ui_cl: bool = True,
        cv_temperature: float = 0.2,
        ui_temperature: float = 0.2,
    ):
        super().__init__()
        self.use_cross_view = use_cross_view
        self.use_ui_cl = use_ui_cl
        self.cv_temperature = cv_temperature
        self.ui_temperature = ui_temperature
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.form_encoder = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim),
        )
        self.form_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.behavior_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.ui_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.register_buffer("user_features", torch.zeros(n_users, feat_dim))

    def set_user_features(self, features: np.ndarray) -> None:
        device = self.user_emb.weight.device
        self.user_features = torch.tensor(features, dtype=torch.float32, device=device)

    def form_view(self, users: torch.Tensor) -> torch.Tensor:
        return self.form_encoder(self.user_features[users.to(self.user_emb.weight.device)])

    def behavior_view(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_emb(users)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.behavior_view(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int) -> torch.Tensor:
        u = self.user_emb.weight[user_id]
        return self.item_emb.weight @ u

    def cross_view_loss(self, users: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        if not self.use_cross_view:
            return torch.tensor(0.0, device=users.device)
        z_form = self.form_projector(embedding_dropout(self.form_view(users), drop_prob))
        z_beh = self.behavior_projector(embedding_dropout(self.behavior_view(users), drop_prob))
        return 0.5 * (
            info_nce(z_form, z_beh, self.cv_temperature) + info_nce(z_beh, z_form, self.cv_temperature)
        )

    def ui_contrastive_loss(self, users: torch.Tensor, items: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        if not self.use_ui_cl:
            return torch.tensor(0.0, device=users.device)
        u = self.behavior_view(users)
        v = self.item_emb(items)
        z1 = self.ui_projector(embedding_dropout(u, drop_prob))
        z2 = self.ui_projector(embedding_dropout(v, drop_prob))
        return info_nce(z1, z2, self.ui_temperature)


@dataclass
class TrainConfig:
    embed_dim: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 2048
    max_epochs: int = 30
    patience: int = 5
    cl_weight: float = 0.1
    cv_weight: float = 0.2
    cv_temperature: float = 0.2
    ui_temperature: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_bpr_model(
    model: nn.Module,
    data: RecData,
    cfg: TrainConfig,
    use_cl: bool = False,
    cl_fn=None,
    method_name: str = "model",
) -> Tuple[nn.Module, Dict]:
    device = torch.device(cfg.device)
    model = model.to(device)

    if isinstance(model, LightGCNModel):
        model.set_adj(build_train_adj(data, device))
    if isinstance(model, MultimodalCFModel):
        model.set_user_features(data.user_features)
        model = model.to(device)
    if isinstance(model, (CVCLERModel, MMSSLModel)):
        model.set_user_features(data.user_features)
        model = model.to(device)

    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    is_gnn = isinstance(model, LightGCNModel)

    for epoch in range(cfg.max_epochs):
        model.train()
        if is_gnn:
            model.clear_cache()
        train_losses = []

        if is_gnn:
            for batch in train_loader:
                users, pos, neg = [x.to(device) for x in batch]
                optimizer.zero_grad()
                model.clear_cache()
                model.propagate()
                loss = bpr_loss(model.score(users, pos), model.score(users, neg))
                if isinstance(model, SimGCLModel):
                    loss = loss + cfg.cl_weight * model.contrastive_loss(users, pos)
                if isinstance(model, LightGCLModel):
                    loss = loss + cfg.cl_weight * model.contrastive_loss(users, pos)
                if isinstance(model, SGLModel):
                    loss = loss + cfg.cl_weight * model.contrastive_loss(users, pos)
                if isinstance(model, MMSSLModel):
                    loss = loss + cfg.cv_weight * model.cross_modal_loss(users)
                loss.backward()
                optimizer.step()
                model.clear_cache()
                train_losses.append(loss.item())
        else:
            for batch in train_loader:
                users, pos, neg = [x.to(device) for x in batch]
                optimizer.zero_grad()
                pos_scores = model.score(users, pos)
                neg_scores = model.score(users, neg)
                loss = bpr_loss(pos_scores, neg_scores)

                if use_cl and cl_fn is not None:
                    loss = loss + cfg.cl_weight * cl_fn(users, pos)
                if hasattr(model, "cross_view_loss") and getattr(model, "use_cross_view", False):
                    cv_w = getattr(cfg, "cv_weight", cfg.cl_weight)
                    loss = loss + cv_w * model.cross_view_loss(users)
                if hasattr(model, "ui_contrastive_loss") and getattr(model, "use_ui_cl", False):
                    loss = loss + cfg.cl_weight * model.ui_contrastive_loss(users, pos)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        model.eval()
        if is_gnn:
            with torch.no_grad():
                model.propagate()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                users, pos, neg = [x.to(device) for x in batch]
                val_losses.append(bpr_loss(model.score(users, pos), model.score(users, neg)).item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 1 == 0:
            _log(f"  [{method_name}] Epoch {epoch + 1}/{cfg.max_epochs} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            _log(f"  [{method_name}] Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        if isinstance(model, LightGCNModel):
            model.set_adj(build_train_adj(data, device))

    history["epochs"] = len(history["train_loss"])
    history["best_val_loss"] = best_val
    return model, history


def evaluate_topk(model: nn.Module, data: RecData, k: int = 10, device: str = "cuda") -> Dict[str, float]:
    per_user = evaluate_topk_per_user(model, data, k=k, device=device)
    if not per_user:
        return {
            "precision@10": 0.0,
            "recall@10": 0.0,
            "f1@10": 0.0,
            "ndcg@10": 0.0,
            "n_eval_users": 0,
        }
    precisions = [m["precision@10"] for m in per_user.values()]
    recalls = [m["recall@10"] for m in per_user.values()]
    f1s = [m["f1@10"] for m in per_user.values()]
    ndcgs = [m["ndcg@10"] for m in per_user.values()]
    return {
        "precision@10": float(np.mean(precisions)),
        "recall@10": float(np.mean(recalls)),
        "f1@10": float(np.mean(f1s)),
        "ndcg@10": float(np.mean(ndcgs)),
        "n_eval_users": len(per_user),
    }


def evaluate_topk_per_user(
    model: nn.Module, data: RecData, k: int = 10, device: str = "cuda"
) -> Dict[int, Dict[str, float]]:
    """Return per-user Top-K metrics keyed by user_id."""
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    if isinstance(model, LightGCNModel):
        model.set_adj(build_train_adj(data, device_t))
    if isinstance(model, MultimodalCFModel):
        model.set_user_features(data.user_features)
    if isinstance(model, (CVCLERModel, MMSSLModel)):
        model.set_user_features(data.user_features)

    out: Dict[int, Dict[str, float]] = {}

    with torch.no_grad():
        if isinstance(model, LightGCNModel):
            model.propagate()
        test_users = list(data.test_relevant.items())
        total = len(test_users)
        for idx, (user_id, relevant) in enumerate(test_users):
            scores = model.all_scores(user_id).detach().cpu().numpy()
            if idx > 0 and idx % 5000 == 0:
                _log(f"  Eval progress: {idx}/{total} users")

            mask_items = data.train_items.get(user_id, set()) | data.val_items.get(user_id, set())
            for item_id in mask_items:
                scores[item_id] = -1e9

            top_k = np.argsort(-scores)[:k]
            hits = len(set(top_k) & relevant)

            prec = hits / k
            rec = hits / len(relevant) if relevant else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            out[user_id] = {
                "precision@10": float(prec),
                "recall@10": float(rec),
                "f1@10": float(f1),
                "ndcg@10": float(ndcg),
            }

    return out


def init_cvcler_from_cler(cv_model: CVCLERModel, cler_model: CLERModel) -> None:
    """Warm-start CV-CLER behavior branch from a trained CLER model."""
    cv_model.user_emb.load_state_dict(cler_model.user_emb.state_dict())
    cv_model.item_emb.load_state_dict(cler_model.item_emb.state_dict())
    cv_model.ui_projector.load_state_dict(cler_model.projection.state_dict())


def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
