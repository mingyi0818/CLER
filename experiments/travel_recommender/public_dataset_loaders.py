"""Load MovieLens-1M and Amazon 5-core subsets for cross-domain SCIS experiments."""

from __future__ import annotations

import gzip
import json
import os
import pickle
import zipfile
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from travel_recommender.paper_pipeline import OUTPUT_DIR, SEED, RecData, _log

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
AMAZON_ELECTRONICS_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
)

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "public")


def _per_user_split(
    interactions: List[Tuple[int, int, float]],
    positive_threshold: float = 1.0,
) -> Tuple[List, List, List, Dict, Dict, Dict]:
    by_user: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for u, i, r in interactions:
        by_user[u].append((u, i, r))

    rng = np.random.RandomState(SEED)
    train, val, test = [], [], []
    train_items: Dict[int, Set[int]] = defaultdict(set)
    val_items: Dict[int, Set[int]] = defaultdict(set)
    test_relevant: Dict[int, Set[int]] = defaultdict(set)

    for user_id, user_inters in by_user.items():
        rng.shuffle(user_inters)
        n = len(user_inters)
        if n == 1:
            train.extend(user_inters)
            if user_inters[0][2] >= positive_threshold:
                train_items[user_id].add(user_inters[0][1])
            continue
        n_train = max(1, int(n * 0.8))
        n_val = max(0, int(n * 0.1))
        n_test = n - n_train - n_val
        if n_test == 0 and n_val > 0:
            n_test = 1
            n_val -= 1
        elif n_test == 0 and n_val == 0 and n_train > 1:
            n_train -= 1
            n_test = 1

        for split_name, chunk in (
            ("train", user_inters[:n_train]),
            ("val", user_inters[n_train : n_train + n_val]),
            ("test", user_inters[n_train + n_val :]),
        ):
            for u, i, r in chunk:
                if split_name == "train":
                    train.append((u, i, r))
                    if r >= positive_threshold:
                        train_items[u].add(i)
                elif split_name == "val":
                    val.append((u, i, r))
                    if r >= positive_threshold:
                        val_items[u].add(i)
                else:
                    test.append((u, i, r))
                    if r >= positive_threshold:
                        test_relevant[u].add(i)

    return train, val, test, train_items, val_items, test_relevant


def _ensure_download(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    _log(f"Downloading {url} ...")
    urlretrieve(url, dest_path)
    _log(f"Saved {dest_path}")


def load_movielens_1m(
    min_user_ratings: int = 20,
    positive_rating: int = 4,
    use_cache: bool = True,
) -> RecData:
    """MovieLens-1M: declarative demographics + implicit feedback (rating >= 4)."""
    cache_path = os.path.join(OUTPUT_DIR, f"movielens_1m_pos{positive_rating}_min{min_user_ratings}.pkl")
    zip_path = os.path.join(DATASET_ROOT, "ml-1m.zip")
    extract_dir = os.path.join(DATASET_ROOT, "ml-1m")

    if use_cache and os.path.exists(cache_path):
        _log("Loading cached MovieLens-1M RecData...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    _ensure_download(ML1M_URL, zip_path)
    if not os.path.exists(os.path.join(extract_dir, "ratings.dat")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATASET_ROOT)

    users = pd.read_csv(
        os.path.join(extract_dir, "users.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    ratings = pd.read_csv(
        os.path.join(extract_dir, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    user_counts = ratings.groupby("user_id").size()
    keep_users = set(user_counts[user_counts >= min_user_ratings].index)
    ratings = ratings[ratings["user_id"].isin(keep_users)]
    ratings = ratings[ratings["rating"] >= positive_rating]

    raw_users = sorted(ratings["user_id"].unique())
    raw_items = sorted(ratings["movie_id"].unique())
    user_map = {u: i for i, u in enumerate(raw_users)}
    item_map = {m: i for i, m in enumerate(raw_items)}

    interactions = [
        (user_map[u], item_map[m], 1.0)
        for u, m in zip(ratings["user_id"], ratings["movie_id"])
    ]

    users = users[users["user_id"].isin(raw_users)].set_index("user_id").loc[raw_users]
    gender_map = {"M": 0, "F": 1}
    age_bins = sorted(users["age"].unique())
    occ_bins = sorted(users["occupation"].unique())
    age_to_idx = {a: i for i, a in enumerate(age_bins)}
    occ_to_idx = {o: i for i, o in enumerate(occ_bins)}
    feat_dim = 2 + len(age_bins) + len(occ_bins)
    user_features = np.zeros((len(raw_users), feat_dim), dtype=np.float32)
    for idx, row in enumerate(users.itertuples()):
        user_features[idx, gender_map.get(row.gender, 0)] = 1.0
        user_features[idx, 2 + age_to_idx[row.age]] = 1.0
        user_features[idx, 2 + len(age_bins) + occ_to_idx[row.occupation]] = 1.0

    train, val, test, train_items, val_items, test_relevant = _per_user_split(interactions)

    data = RecData(
        n_users=len(raw_users),
        n_items=len(raw_items),
        train=train,
        val=val,
        test=test,
        user_features=user_features,
        train_items=dict(train_items),
        val_items=dict(val_items),
        test_relevant=dict(test_relevant),
        stats={
            "dataset": "MovieLens-1M",
            "n_users": len(raw_users),
            "n_items": len(raw_items),
            "n_interactions": len(interactions),
            "positive_rule": f"rating>={positive_rating}",
            "min_user_ratings": min_user_ratings,
            "feat_dim": feat_dim,
            "split": "per-user 80/10/10",
        },
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    _log(f"MovieLens-1M cached: {data.stats}")
    return data


def load_amazon_electronics_5core(
    max_users: int = 25000,
    min_user_interactions: int = 10,
    use_cache: bool = True,
) -> RecData:
    """Amazon-Electronics 5-core: large item catalog; user profile from train-split activity stats."""
    cache_path = os.path.join(
        OUTPUT_DIR, f"amazon_electronics_5core_u{max_users}_min{min_user_interactions}.pkl"
    )
    gz_path = os.path.join(DATASET_ROOT, "reviews_Electronics_5.json.gz")

    if use_cache and os.path.exists(cache_path):
        _log("Loading cached Amazon-Electronics RecData...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    _ensure_download(AMAZON_ELECTRONICS_URL, gz_path)

    by_user: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    item_set: Set[str] = set()
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = row["reviewerID"]
            asin = row["asin"]
            rating = float(row["overall"])
            if rating >= 4.0:
                by_user[uid].append((uid, asin, 1.0))
                item_set.add(asin)

    eligible = [
        (uid, inters)
        for uid, inters in by_user.items()
        if len(inters) >= min_user_interactions
    ]
    eligible.sort(key=lambda x: -len(x[1]))
    eligible = eligible[:max_users]

    raw_users = [uid for uid, _ in eligible]
    used_items: Set[str] = set()
    for _, inters in eligible:
        for _, asin, _ in inters:
            used_items.add(asin)
    raw_items = sorted(used_items)
    user_map = {u: i for i, u in enumerate(raw_users)}
    item_map = {a: i for i, a in enumerate(raw_items)}

    interactions = [
        (user_map[u], item_map[a], r) for u, inters in eligible for u, a, r in inters if a in item_map
    ]

    # Declarative profile from TRAIN interactions only (computed after split).
    train, val, test, train_items, val_items, test_relevant = _per_user_split(interactions)
    feat_dim = 20
    user_features = np.zeros((len(raw_users), feat_dim), dtype=np.float32)
    train_by_user: Dict[int, List[float]] = defaultdict(list)
    for u, _, _ in train:
        train_by_user[u].append(1.0)
    for u in range(len(raw_users)):
        n_train = len(train_by_user.get(u, []))
        if n_train == 0:
            user_features[u, 0] = 1.0
            continue
        activity_bin = min(9, n_train // 5)
        user_features[u, activity_bin] = 1.0
        user_features[u, 10 + min(9, n_train // 10)] = 1.0

    data = RecData(
        n_users=len(raw_users),
        n_items=len(raw_items),
        train=train,
        val=val,
        test=test,
        user_features=user_features,
        train_items=dict(train_items),
        val_items=dict(val_items),
        test_relevant=dict(test_relevant),
        stats={
            "dataset": "Amazon-Electronics-5core",
            "n_users": len(raw_users),
            "n_items": len(raw_items),
            "n_interactions": len(interactions),
            "positive_rule": "rating>=4",
            "min_user_interactions": min_user_interactions,
            "max_users": max_users,
            "feat_dim": feat_dim,
            "split": "per-user 80/10/10",
        },
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    _log(f"Amazon-Electronics cached: {data.stats}")
    return data
