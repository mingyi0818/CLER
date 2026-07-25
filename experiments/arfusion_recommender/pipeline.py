"""Data loading, baselines, training and evaluation for ARFusion research."""

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

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "dataset"
STRAVL_PATH = DATA_DIR / "Stravl_Travel_Preference_Data.csv"
INTTRAVEL_DIR = DATA_DIR / "IntTravel" / "full"
EXPERIMENTS_DIR = ROOT / "experiments"
CACHE_DIR = EXPERIMENTS_DIR / "cache"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
SEED = 42

FORM_FIELD_SIZES = {
    "form_a": 4, "form_b": 4, "form_c": 4,
    "form_f": 8, "form_g": 8,
    "form_h": 3, "form_i": 3, "form_j": 3, "form_k": 8,
}

PROFILE_FIELD_SIZES = {
    "profile_feature_1": 4, "profile_feature_2": 2, "profile_feature_3": 9,
    "profile_feature_4": 18, "profile_feature_5": 10, "profile_feature_6": 2,
}

POSITIVE_ACTION_TYPES = {2, 3}
USER_INFO_URL = "https://huggingface.co/datasets/GD-ML/IntTravel_dataset/resolve/main/user_info.csv"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def _items_by_user(rows: List[Tuple[int, int, float]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for u, i, _ in rows:
        out[u].add(i)
    return dict(out)


def parse_list_string(value) -> List:
    if pd.isna(value) or value == "[]":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


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
    rows = []
    for _, row in df.iterrows():
        parts = [_parse_form_multihot(row[col], size) for col, size in FORM_FIELD_SIZES.items()]
        rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=np.float32)


def load_stravl_data(use_cache: bool = True) -> RecData:
    cache_path = CACHE_DIR / "stravl_rec_data.pkl"
    if use_cache and cache_path.exists():
        mtime_data = STRAVL_PATH.stat().st_mtime
        if cache_path.stat().st_mtime >= mtime_data:
            _log("Loading cached Stravl RecData...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    _log("Parsing Stravl CSV...")
    df = pd.read_csv(STRAVL_PATH)
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
    mapped = [(u, dest_to_idx[d], r) for u, d, r in interactions if d in dest_to_idx]
    user_features = _extract_form_features(df)

    by_user: Dict[int, List] = defaultdict(list)
    for u, i, r in mapped:
        by_user[u].append((u, i, r))

    rng = np.random.RandomState(SEED)
    train, val, test = [], [], []
    for user_inters in by_user.values():
        rng.shuffle(user_inters)
        n = len(user_inters)
        if n == 1:
            train.extend(user_inters)
            continue
        n_train = max(1, int(n * 0.8))
        n_val = max(0, int(n * 0.1))
        n_test = n - n_train - n_val
        if n_test == 0 and n_val > 0:
            n_test, n_val = 1, n_val - 1
        elif n_test == 0:
            n_test, n_train = 1, max(1, n_train - 1)
        train.extend(user_inters[:n_train])
        val.extend(user_inters[n_train : n_train + n_val])
        test.extend(user_inters[n_train + n_val :])

    train_items = _items_by_user(train)
    val_items = _items_by_user(val)
    test_relevant = {
        u: {i for uu, i, r in test if uu == u and r >= 1.0}
        for u in set(u for u, _, _ in test)
    }
    test_relevant = {u: s for u, s in test_relevant.items() if s}

    result = RecData(
        n_users=n_users,
        n_items=len(dest_to_idx),
        train=train, val=val, test=test,
        user_features=user_features,
        train_items=train_items, val_items=val_items,
        test_relevant=test_relevant,
        stats={
            "dataset": "Stravl",
            "n_users": n_users,
            "n_items": len(dest_to_idx),
            "n_interactions": len(mapped),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_positive_train": sum(1 for _, _, r in train if r >= 1.0),
            "n_positive_test": sum(len(v) for v in test_relevant.values()),
            "n_test_users": len(test_relevant),
        },
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def _encode_profile_row(row: pd.Series) -> np.ndarray:
    parts = []
    for col, size in PROFILE_FIELD_SIZES.items():
        vec = np.zeros(size, dtype=np.float32)
        val = row.get(col, np.nan)
        if pd.notna(val):
            idx = int(val)
            if 0 <= idx < size:
                vec[idx] = 1.0
        parts.append(vec)
    return np.concatenate(parts)


def _load_user_profiles(user_ids: Set[int], max_retries: int = 3) -> Dict[int, np.ndarray]:
    profiles: Dict[int, np.ndarray] = {}
    _log(f"Streaming user profiles for {len(user_ids):,} users...")
    last_err = None
    for attempt in range(max_retries):
        try:
            profiles.clear()
            for chunk in pd.read_csv(USER_INFO_URL, sep="\t", chunksize=500_000):
                sub = chunk[chunk["user_id"].isin(user_ids)]
                for _, row in sub.iterrows():
                    profiles[int(row["user_id"])] = _encode_profile_row(row)
                if len(profiles) >= len(user_ids):
                    break
            _log(f"Loaded profiles for {len(profiles):,}/{len(user_ids):,} users")
            return profiles
        except Exception as exc:
            last_err = exc
            _log(f"Profile download attempt {attempt + 1}/{max_retries} failed: {exc}")
    raise RuntimeError(f"Failed to load user profiles after {max_retries} attempts") from last_err


def _iter_inttravel_shards(paths: List[str]):
    for path in paths:
        _log(f"  reading {os.path.basename(path)} ...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=2_000_000, usecols=["user_id", "poi_id", "action_type"]):
            pos = chunk[chunk["action_type"].isin(POSITIVE_ACTION_TYPES)]
            if not pos.empty:
                yield pos


def _recdata_from_any(obj) -> RecData:
    """Rebuild RecData from parent-module pickle or local instance."""
    if isinstance(obj, RecData):
        return obj
    return RecData(
        n_users=obj.n_users,
        n_items=obj.n_items,
        train=obj.train,
        val=obj.val,
        test=obj.test,
        user_features=obj.user_features,
        train_items=obj.train_items,
        val_items=obj.val_items,
        test_relevant=obj.test_relevant,
        stats=dict(getattr(obj, "stats", {})),
    )


def _try_load_paper_cache(shard_tag: str, min_unique_pois: int, max_users: int) -> RecData | None:
    paper_cache = PROJECT_ROOT / "paper_experiments" / f"inttravel_{shard_tag}_min{min_unique_pois}_u{max_users}.pkl"
    if not paper_cache.exists():
        return None
    _log(f"Loading IntTravel RecData from paper cache: {paper_cache.name}")
    import sys
    parent = str(PROJECT_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    with open(paper_cache, "rb") as f:
        cached = pickle.load(f)
    data = cached["data"] if isinstance(cached, dict) and "data" in cached else cached
    return _recdata_from_any(data)


def _split_inttravel_interactions(
    by_user: Dict[int, List[Tuple[int, int, float]]],
) -> Tuple[List, List, List]:
    """Match travel_recommender/inttravel_loader.py exactly."""
    rng_np = np.random.RandomState(SEED)
    train, val, test = [], [], []
    for user_inters in by_user.values():
        rng_np.shuffle(user_inters)
        n = len(user_inters)
        if n == 1:
            train.extend(user_inters)
            continue
        if n == 2:
            train.extend(user_inters[:1])
            test.extend(user_inters[1:])
            continue
        if n == 3:
            train.extend(user_inters[:1])
            val.extend(user_inters[1:2])
            test.extend(user_inters[2:])
            continue
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)
        train.extend(user_inters[:n_train])
        val.extend(user_inters[n_train : n_train + n_val])
        test.extend(user_inters[n_train + n_val :])
    return train, val, test


def _count_user_pois(paths: List[str]) -> Dict[int, Set[int]]:
    user_pois: Dict[int, Set[int]] = defaultdict(set)
    for pos in _iter_inttravel_shards(paths):
        for u, p in zip(pos["user_id"].values, pos["poi_id"].values):
            user_pois[int(u)].add(int(p))
    return user_pois


def _collect_user_pairs(paths: List[str], selected_set: Set[int]) -> Dict[int, Set[int]]:
    raw_pairs: Dict[int, Set[int]] = defaultdict(set)
    for pos in _iter_inttravel_shards(paths):
        sub = pos[pos["user_id"].isin(selected_set)]
        for u, p in zip(sub["user_id"].values, sub["poi_id"].values):
            raw_pairs[int(u)].add(int(p))
    return raw_pairs


def load_inttravel_data(
    n_shards: int = 1,
    min_unique_pois: int = 3,
    max_users: int = 25000,
    use_cache: bool = True,
) -> RecData:
    """Aligned with travel_recommender/inttravel_loader.py (sorted user subset, two-pass)."""
    paths = [str(INTTRAVEL_DIR / f"interaction_{i}.csv") for i in range(1, n_shards + 1)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing shard: {p}")

    shard_tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in paths)
    cache_key = {
        "interaction_paths": [os.path.basename(p) for p in paths],
        "min_unique_pois": min_unique_pois,
        "max_users": max_users,
        "positive_action_types": sorted(POSITIVE_ACTION_TYPES),
        "loader_version": "aligned_v2",
    }
    cache_path = CACHE_DIR / f"inttravel_{shard_tag}_min{min_unique_pois}_u{max_users}_v2.pkl"

    if use_cache and cache_path.exists():
        oldest_mtime = min(os.path.getmtime(p) for p in paths)
        if cache_path.stat().st_mtime >= oldest_mtime:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                _log("Loading cached IntTravel RecData (aligned loader)...")
                return cached["data"]

    paper_data = _try_load_paper_cache(shard_tag, min_unique_pois, max_users)
    if paper_data is not None and use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": paper_data}, f)
        _log(f"Mirrored paper cache to {cache_path}")
        return paper_data

    _log(f"Pass 1: counting unique positive POIs across {len(paths)} shard(s)...")
    user_pois = _count_user_pois(paths)
    eligible = [u for u, pois in user_pois.items() if len(pois) >= min_unique_pois]
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    selected_raw = sorted(eligible[:max_users])
    selected_set = set(selected_raw)
    _log(f"Eligible users (>={min_unique_pois} POIs): {len(eligible):,}; using {len(selected_raw):,}")

    _log("Pass 2: collecting interactions for selected users...")
    raw_pairs = _collect_user_pairs(paths, selected_set)
    profiles = _load_user_profiles(selected_set)
    kept_raw_users = [
        u for u in selected_raw if u in profiles and len(raw_pairs.get(u, set())) >= min_unique_pois
    ]

    raw_to_idx = {u: i for i, u in enumerate(kept_raw_users)}
    poi_set: Set[int] = set()
    for u in kept_raw_users:
        poi_set.update(raw_pairs[u])
    poi_to_idx = {p: i for i, p in enumerate(sorted(poi_set))}

    mapped: List[Tuple[int, int, float]] = []
    for u in kept_raw_users:
        ui = raw_to_idx[u]
        for p in raw_pairs[u]:
            mapped.append((ui, poi_to_idx[p], 1.0))

    feat_dim = sum(PROFILE_FIELD_SIZES.values())
    user_features = np.zeros((len(kept_raw_users), feat_dim), dtype=np.float32)
    for u in kept_raw_users:
        user_features[raw_to_idx[u]] = profiles[u]

    by_user: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for triplet in mapped:
        by_user[triplet[0]].append(triplet)

    train, val, test = _split_inttravel_interactions(by_user)
    train_items = _items_by_user(train)
    val_items = _items_by_user(val)
    test_relevant = {
        u: {i for uu, i, r in test if uu == u and r >= 1.0}
        for u in set(u for u, _, _ in test)
    }
    test_relevant = {u: items for u, items in test_relevant.items() if items}

    poi_counts = [len(raw_pairs[u]) for u in kept_raw_users]
    result = RecData(
        n_users=len(kept_raw_users),
        n_items=len(poi_to_idx),
        train=train,
        val=val,
        test=test,
        user_features=user_features,
        train_items=train_items,
        val_items=val_items,
        test_relevant=test_relevant,
        stats={
            "dataset": "IntTravel",
            "n_shards": n_shards,
            "shards": [os.path.basename(p) for p in paths],
            "min_unique_pois": min_unique_pois,
            "max_users_cap": max_users,
            "n_users": len(kept_raw_users),
            "n_items": len(poi_to_idx),
            "n_interactions": len(mapped),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_positive_train": len(train),
            "n_positive_test": sum(len(v) for v in test_relevant.values()),
            "n_test_users": len(test_relevant),
            "mean_unique_pois_per_user": float(np.mean(poi_counts)),
            "median_unique_pois_per_user": float(np.median(poi_counts)),
            "split": "1/1/1 for 3-POI users; 8/1/1 otherwise on deduplicated positive POIs",
            "loader": "aligned_with_inttravel_loader",
        },
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"cache_key": cache_key, "data": result}, f)
    _log(f"IntTravel RecData cached to {cache_path}")
    return result


class BPRDataset(Dataset):
    """BPR triplet sampler.

    For ``split="val"`` the negative-sampling exclusion set is built from the
    union of train and val positives, so validation negatives never collide
    with known training positives. Validation negatives are also drawn from a
    fixed RNG seeded by ``(user, pos)`` so that the validation loss is stable
    across epochs (previous implementation re-sampled randomly every epoch,
    making early-stopping noisy). Training negatives remain stochastic.
    """

    def __init__(self, data: RecData, split: str = "train"):
        rows = {"train": data.train, "val": data.val, "test": data.test}[split]
        self.pos_pairs = [(u, i) for u, i, r in rows if r >= 1.0]
        self.all_items = list(range(data.n_items))
        self.split = split
        self.user_pos = defaultdict(set)
        if split == "val":
            # Exclude all known positives (train + val) so val negatives are
            # never items the user actually interacted with during training.
            for u, i, r in data.train:
                if r >= 1.0:
                    self.user_pos[u].add(i)
            for u, i, r in data.val:
                if r >= 1.0:
                    self.user_pos[u].add(i)
        else:
            for u, i, r in rows:
                if r >= 1.0:
                    self.user_pos[u].add(i)
        # Pre-sample deterministic negatives for validation triples
        if split == "val":
            self._val_negs = self._sample_fixed_negatives()
        else:
            self._val_negs = None

    def _sample_fixed_negatives(self) -> List[int]:
        rng = random.Random(0xBEEF)  # fixed seed for reproducibility
        negs = []
        for u, _ in self.pos_pairs:
            for _ in range(20):
                j = rng.choice(self.all_items)
                if j not in self.user_pos[u]:
                    negs.append(j)
                    break
            else:
                negs.append(rng.choice(self.all_items))
        return negs

    def __len__(self) -> int:
        return max(len(self.pos_pairs), 1)

    def __getitem__(self, idx: int):
        u, i = self.pos_pairs[idx % len(self.pos_pairs)]
        if self._val_negs is not None:
            return u, i, self._val_negs[idx % len(self._val_negs)]
        for _ in range(20):
            j = random.choice(self.all_items)
            if j not in self.user_pos[u]:
                return u, i, j
        return u, i, random.choice(self.all_items)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def build_bipartite_adj(data: RecData, device: torch.device) -> torch.Tensor:
    rows, cols = [], []
    for u, i, r in data.train:
        if r >= 1.0:
            rows.append(u)
            cols.append(i)
    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0),
            (data.n_users, data.n_items), device=device,
        ).coalesce()
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (data.n_users, data.n_items), device=device).coalesce()
    user_deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    item_deg = torch.sparse.sum(adj, dim=0).to_dense().clamp(min=1.0)
    row, col = adj.indices()
    norm_vals = adj.values() * user_deg[row].pow(-0.5) * item_deg[col].pow(-0.5)
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape, device=device).coalesce()


def user_positive_counts(data: RecData) -> np.ndarray:
    counts = np.zeros(data.n_users, dtype=np.float32)
    for u, _, r in data.train:
        if r >= 1.0:
            counts[u] += 1
    return counts


class MFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, users, items):
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int):
        return self.item_emb.weight @ self.user_emb.weight[user_id]


class CLERModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, ui_temperature: float = 0.2):
        super().__init__()
        self.ui_temperature = ui_temperature
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, users, items):
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int):
        return self.item_emb.weight @ self.user_emb.weight[user_id]

    def contrastive_loss(self, users, items, drop_prob: float = 0.1):
        from .arfusion_model import embedding_dropout, info_nce
        u = self.user_emb(users)
        v = self.item_emb(items)
        z1 = self.projection(embedding_dropout(u, drop_prob))
        z2 = self.projection(embedding_dropout(v, drop_prob))
        # Multi-positive mask: same user at multiple batch positions implies
        # their respective positive items are also positives for that user.
        pos_mask = (users.unsqueeze(0) == users.unsqueeze(1)).float()
        return info_nce(z1, z2, self.ui_temperature, pos_mask=pos_mask)


class MultimodalCFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, feat_dim: int, dim: int = 64):
        super().__init__()
        self.user_cf = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_mlp = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, dim))
        nn.init.normal_(self.user_cf.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.register_buffer("user_features", torch.zeros(n_users, feat_dim))

    def set_user_features(self, features: np.ndarray) -> None:
        self.user_features = torch.tensor(features, dtype=torch.float32, device=self.user_cf.weight.device)

    def user_repr(self, users):
        return self.user_cf(users) + self.user_mlp(self.user_features[users])

    def score(self, users, items):
        return (self.user_repr(users) * self.item_emb(items)).sum(dim=1)

    def all_scores(self, user_id: int):
        device = self.user_cf.weight.device
        u = self.user_repr(torch.tensor([user_id], device=device))[0]
        return self.item_emb.weight @ u


@dataclass
class TrainConfig:
    embed_dim: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 2048
    max_epochs: int = 40
    patience: int = 8
    cl_weight: float = 0.1
    cv_weight: float = 0.2
    pbd_weight: float = 0.05
    fuse_weight: float = 0.1
    cv_temperature: float = 0.2
    ui_temperature: float = 0.2
    n_gnn_layers: int = 2
    use_graph: bool = True
    score_mode: str = "dual"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_model(model: nn.Module, data: RecData, device: torch.device):
    from .arfusion_model import ARFusionModel

    model = model.to(device)
    # Any model that exposes set_user_features (MultimodalCFModel, ARFusionModel,
    # DropoutNet, etc.) gets fresh features bound to the target device.
    if hasattr(model, "set_user_features"):
        try:
            model.set_user_features(data.user_features)
        except TypeError:
            # Method signature mismatch — skip silently (model already configured).
            pass
    if isinstance(model, ARFusionModel):
        model.set_user_log_counts(user_positive_counts(data))
        model.set_adj(build_bipartite_adj(data, device))
    return model


def train_model(
    model: nn.Module,
    data: RecData,
    cfg: TrainConfig,
    method_name: str = "model",
) -> Tuple[nn.Module, Dict]:
    from .arfusion_model import ARFusionModel

    device = torch.device(cfg.device)
    model = _prepare_model(model, data, device)
    is_arfusion = isinstance(model, ARFusionModel)

    train_loader = DataLoader(BPRDataset(data, "train"), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BPRDataset(data, "val"), batch_size=cfg.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        if is_arfusion:
            model.clear_cache()
        train_losses = []

        for batch in train_loader:
            users, pos, neg = [x.to(device) for x in batch]
            optimizer.zero_grad()
            if is_arfusion and model.use_graph:
                model.clear_cache()
                model.propagate()
            loss = bpr_loss(model.score(users, pos), model.score(users, neg))
            if is_arfusion:
                loss = loss + cfg.cl_weight * model.ui_contrastive_loss(users, pos)
                loss = loss + cfg.cv_weight * model.cross_view_loss(users)
                loss = loss + cfg.pbd_weight * model.profile_distillation_loss(users)
                loss = loss + cfg.fuse_weight * model.fusion_loss(users, pos, neg)
            loss.backward()
            optimizer.step()
            if is_arfusion and model.use_graph:
                model.clear_cache()
            train_losses.append(loss.item())

        model.eval()
        if is_arfusion and model.use_graph:
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
        model = _prepare_model(model, data, device)

    history["epochs"] = len(history["train_loss"])
    history["best_val_loss"] = best_val
    return model, history


def evaluate_topk_per_user(model: nn.Module, data: RecData, k: int = 10, device: str = "cuda") -> Dict[int, Dict[str, float]]:
    """Evaluate Top-K metrics per user.

    Uses batched scoring when possible (all users at once via matrix multiply)
    to avoid the slow per-user Python loop. Falls back to per-user scoring
    only when the model does not support batched ``all_scores_batch``.
    """
    from .arfusion_model import ARFusionModel

    device_t = torch.device(device)
    model = _prepare_model(model, data, device_t)
    model.eval()
    out: Dict[int, Dict[str, float]] = {}

    test_users = list(data.test_relevant.items())
    n_users = len(test_users)
    user_ids = [uid for uid, _ in test_users]

    with torch.no_grad():
        # Try batched scoring: compute [n_test_users, n_items] score matrix
        all_scores = None
        if hasattr(model, "all_scores_batch"):
            all_scores = model.all_scores_batch(user_ids)
        elif isinstance(model, ARFusionModel):
            all_scores = _arfusion_all_scores_batch(model, user_ids)
        elif hasattr(model, "user_emb") and hasattr(model, "item_emb"):
            # MFModel / CLERModel: simple dot product
            u = model.user_emb(torch.tensor(user_ids, device=device_t))
            v = model.item_emb.weight
            all_scores = u @ v.T
        elif hasattr(model, "user_repr") and hasattr(model, "item_emb"):
            # MultimodalCFModel: user_cf + user_mlp(features) dot item_emb
            u = model.user_repr(torch.tensor(user_ids, device=device_t))
            v = model.item_emb.weight
            all_scores = u @ v.T

        if all_scores is not None:
            all_scores = all_scores.detach().cpu().numpy()
            for idx, (user_id, relevant) in enumerate(test_users):
                if idx > 0 and idx % 5000 == 0:
                    _log(f"  Eval progress: {idx}/{n_users} users")
                scores = all_scores[idx].copy()
                mask_items = data.train_items.get(user_id, set()) | data.val_items.get(user_id, set())
                for item_id in mask_items:
                    scores[item_id] = -1e9
                top_k = np.argpartition(-scores, k)[:k]
                top_k = top_k[np.argsort(-scores[top_k])]
                hits = len(set(top_k) & relevant)
                prec = hits / k
                rec = hits / len(relevant) if relevant else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                out[user_id] = {
                    "precision@10": float(prec), "recall@10": float(rec),
                    "f1@10": float(f1), "ndcg@10": float(ndcg),
                }
            return out

        # Fallback: per-user scoring (slow)
        if isinstance(model, ARFusionModel) and model.use_graph:
            model.propagate()
        for idx, (user_id, relevant) in enumerate(test_users):
            scores = model.all_scores(user_id).detach().cpu().numpy()
            if idx > 0 and idx % 5000 == 0:
                _log(f"  Eval progress: {idx}/{n_users} users")
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
                "precision@10": float(prec), "recall@10": float(rec),
                "f1@10": float(f1), "ndcg@10": float(ndcg),
            }
    return out


def _arfusion_all_scores_batch(model, user_ids):
    """Batched scoring for ARFusionModel: compute scores for all users at once.

    Uses matrix-multiplication formulation ``u @ v.T`` to avoid the
    memory blowup of a [B, M, D] intermediate tensor in dual mode.
    """
    device = model.user_beh.weight.device
    users = torch.tensor(user_ids, device=device)

    # Ensure graph propagation cache exists when graph is enabled.
    if model.use_graph and model._cached_user_graph is None and model.adj is not None:
        model.propagate()

    # Full item-side weight for the collaborative stream.
    if model.use_graph and model._cached_item_graph is not None:
        mix = torch.sigmoid(model.graph_mix)
        v_collab_full = (1.0 - mix) * model.item_emb.weight + mix * model._cached_item_graph
    else:
        v_collab_full = model.item_emb.weight
    v_prof_full = model.item_emb.weight

    if model.score_mode == "collab":
        u = model.collaborative_repr(users)  # [B, D]
        return u @ v_collab_full.T  # [B, M]
    if model.score_mode == "additive":
        lam = model.reliability(users).squeeze(-1)  # [B]
        u = model.collaborative_repr(users) + (1.0 - lam).unsqueeze(-1) * model.profile_repr(users)  # [B, D]
        return u @ v_prof_full.T  # [B, M]
    # dual mode: compute s_c and s_p as [B, M] via matmul, then fuse.
    lam = model.reliability(users).squeeze(-1)  # [B]
    u_collab = model.collaborative_repr(users)  # [B, D]
    u_prof = model.profile_repr(users)  # [B, D]
    s_c = u_collab @ v_collab_full.T  # [B, M]
    s_p = u_prof @ v_prof_full.T  # [B, M]
    return lam.unsqueeze(-1) * s_c + (1.0 - lam).unsqueeze(-1) * s_p


def evaluate_topk(model: nn.Module, data: RecData, k: int = 10, device: str = "cuda") -> Dict[str, float]:
    per_user = evaluate_topk_per_user(model, data, k=k, device=device)
    if not per_user:
        return {"precision@10": 0.0, "recall@10": 0.0, "f1@10": 0.0, "ndcg@10": 0.0, "n_eval_users": 0}
    return {
        "precision@10": float(np.mean([m["precision@10"] for m in per_user.values()])),
        "recall@10": float(np.mean([m["recall@10"] for m in per_user.values()])),
        "f1@10": float(np.mean([m["f1@10"] for m in per_user.values()])),
        "ndcg@10": float(np.mean([m["ndcg@10"] for m in per_user.values()])),
        "n_eval_users": len(per_user),
    }


def save_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
