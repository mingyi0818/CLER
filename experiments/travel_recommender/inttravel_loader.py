"""Load IntTravel interaction shard(s) for cross-dataset CV-CLER validation."""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from travel_recommender.paper_pipeline import RecData, SEED, _items_by_user, _log

# Portable path setup: data/ at repo root, cache/ under experiments/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INTTRAVEL_DIR = str(_REPO_ROOT / "data" / "IntTravel" / "full")
DEFAULT_SHARD_PATHS = [
    os.path.join(INTTRAVEL_DIR, f"interaction_{i}.csv") for i in (1, 2, 3)
]
USER_INFO_URL = "https://huggingface.co/datasets/GD-ML/IntTravel_dataset/resolve/main/user_info.csv"
_CACHE_DIR = _REPO_ROOT / "experiments" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = str(_CACHE_DIR / "inttravel_rec_data_cache.pkl")
MULTI_CACHE_PATH = str(_CACHE_DIR / "inttravel_multishard_cache.pkl")

POSITIVE_ACTION_TYPES = {2, 3}

PROFILE_FIELD_SIZES = {
    "profile_feature_1": 4,
    "profile_feature_2": 2,
    "profile_feature_3": 9,
    "profile_feature_4": 18,
    "profile_feature_5": 10,
    "profile_feature_6": 2,
}


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


def _load_user_profiles(user_ids: Set[int]) -> Dict[int, np.ndarray]:
    profiles: Dict[int, np.ndarray] = {}
    _log(f"Streaming user profiles for {len(user_ids):,} users...")
    for chunk in pd.read_csv(USER_INFO_URL, sep="\t", chunksize=500_000):
        sub = chunk[chunk["user_id"].isin(user_ids)]
        for _, row in sub.iterrows():
            profiles[int(row["user_id"])] = _encode_profile_row(row)
        if len(profiles) >= len(user_ids):
            break
    _log(f"Loaded profiles for {len(profiles):,}/{len(user_ids):,} users")
    return profiles


def _iter_shards(interaction_paths: Sequence[str]):
    for path in interaction_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing IntTravel shard: {path}")
        _log(f"  reading {os.path.basename(path)} ...")
        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=2_000_000,
            usecols=["user_id", "poi_id", "action_type"],
        ):
            pos = chunk[chunk["action_type"].isin(POSITIVE_ACTION_TYPES)]
            if not pos.empty:
                yield pos


def _count_user_pois(interaction_paths: Sequence[str]) -> Dict[int, Set[int]]:
    user_pois: Dict[int, Set[int]] = defaultdict(set)
    for pos in _iter_shards(interaction_paths):
        for u, p in zip(pos["user_id"].values, pos["poi_id"].values):
            user_pois[int(u)].add(int(p))
    return user_pois


def _collect_user_pairs(
    interaction_paths: Sequence[str], selected_set: Set[int]
) -> Dict[int, Set[int]]:
    raw_pairs: Dict[int, Set[int]] = defaultdict(set)
    for pos in _iter_shards(interaction_paths):
        sub = pos[pos["user_id"].isin(selected_set)]
        for u, p in zip(sub["user_id"].values, sub["poi_id"].values):
            raw_pairs[int(u)].add(int(p))
    return raw_pairs


def _split_interactions(
    by_user: Dict[int, List[Tuple[int, int, float]]],
) -> Tuple[List, List, List]:
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


def _build_rec_data(
    kept_raw_users: List[int],
    raw_pairs: Dict[int, Set[int]],
    profiles: Dict[int, np.ndarray],
    stats_extra: Dict,
) -> RecData:
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

    train, val, test = _split_interactions(by_user)
    train_items = _items_by_user(train)
    val_items = _items_by_user(val)
    test_relevant = {
        u: {i for uu, i, r in test if uu == u and r >= 1.0}
        for u in set(u for u, _, _ in test)
    }
    test_relevant = {u: items for u, items in test_relevant.items() if items}

    poi_counts = [len(raw_pairs[u]) for u in kept_raw_users]
    stats = {
        "dataset": "IntTravel",
        "n_users": len(kept_raw_users),
        "n_items": len(poi_to_idx),
        "n_interactions": len(mapped),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_positive_train": len(train),
        "n_positive_test": sum(len(v) for v in test_relevant.values()),
        "n_test_users": len(test_relevant),
        "min_unique_pois": stats_extra["min_unique_pois"],
        "max_users_cap": stats_extra["max_users_cap"],
        "positive_action_types": sorted(POSITIVE_ACTION_TYPES),
        "profile_feat_dim": int(feat_dim),
        "split": "1/1/1 for 3-POI users; 8/1/1 otherwise on deduplicated positive POIs",
        "mean_unique_pois_per_user": float(np.mean(poi_counts)),
        "median_unique_pois_per_user": float(np.median(poi_counts)),
        **{k: v for k, v in stats_extra.items() if k not in ("min_unique_pois", "max_users_cap")},
    }

    return RecData(
        n_users=len(kept_raw_users),
        n_items=len(poi_to_idx),
        train=train,
        val=val,
        test=test,
        user_features=user_features,
        train_items=train_items,
        val_items=val_items,
        test_relevant=test_relevant,
        stats=stats,
    )


def load_inttravel_data(
    interaction_paths: Union[str, Sequence[str], None] = None,
    min_unique_pois: int = 3,
    max_users: int = 25000,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> RecData:
    if interaction_paths is None:
        paths = [os.path.join(INTTRAVEL_DIR, "interaction_1.csv")]
    elif isinstance(interaction_paths, str):
        paths = [interaction_paths]
    else:
        paths = list(interaction_paths)

    if cache_path is None:
        shard_tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in paths)
        cache_path = os.path.join(
            os.path.dirname(CACHE_PATH),
            f"inttravel_{shard_tag}_min{min_unique_pois}_u{max_users}.pkl",
        )

    cache_key = {
        "interaction_paths": [os.path.basename(p) for p in paths],
        "min_unique_pois": min_unique_pois,
        "max_users": max_users,
        "positive_action_types": sorted(POSITIVE_ACTION_TYPES),
    }

    if use_cache and os.path.exists(cache_path):
        oldest_mtime = min(os.path.getmtime(p) for p in paths)
        mtime_cache = os.path.getmtime(cache_path)
        if mtime_cache >= oldest_mtime:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                _log("Loading cached IntTravel RecData...")
                return cached["data"]

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
    kept_raw_users = [u for u in selected_raw if u in profiles and len(raw_pairs[u]) >= min_unique_pois]

    stats_extra = {
        "min_unique_pois": min_unique_pois,
        "max_users_cap": max_users,
        "shards": [os.path.basename(p) for p in paths],
        "n_shards": len(paths),
    }
    if len(paths) == 1:
        stats_extra["shard"] = os.path.basename(paths[0])

    result = _build_rec_data(kept_raw_users, raw_pairs, profiles, stats_extra)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"cache_key": cache_key, "data": result}, f)
    _log(f"IntTravel RecData cached to {cache_path}")
    return result
