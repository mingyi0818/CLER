"""MovieLens Beliefs Dataset loader for cross-domain CLER/CV-CLER/ARFusion experiments.

声明式 profile (说): userPredictRating (对未看电影的预期评分) → 按电影类型聚合成偏好向量
行为反馈 (做): userElicitRating (实际评分) + user_rating_history → rating>=4 为正反馈

数据集路径: D:\\datasets\\tourism\\ml_belief_2024_data_release_2\\data_release\\
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# 复用 arfusion_recommender 的接口 (与 run_stravl_5seed.py 一致)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arfusion_recommender.pipeline import RecData, _log, set_seed, SEED

# 数据集路径
BELIEFS_DIR = Path(r"D:\datasets\tourism\ml_belief_2024_data_release_2\data_release")
BELIEFS_CSV = BELIEFS_DIR / "belief_data.csv"
MOVIES_CSV = BELIEFS_DIR / "movies.csv"
RATING_HIST_CSV = BELIEFS_DIR / "user_rating_history.csv"
EXTRA_RATINGS_CSV = BELIEFS_DIR / "ratings_for_additional_users.csv"

# 缓存目录 (与 arfusion_recommender 一致)
CACHE_DIR = ROOT / "experiments" / "cache"

# 18种电影类型
MOVIE_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _parse_genres(genre_str: str) -> List[str]:
    if pd.isna(genre_str) or genre_str == "(no genres listed)":
        return []
    return genre_str.split("|")


def _build_movie_genre_matrix(movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    movie_genres = {}
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        genres = _parse_genres(row["genres"])
        vec = np.zeros(len(MOVIE_GENRES), dtype=np.float32)
        for g in genres:
            if g in MOVIE_GENRES:
                vec[MOVIE_GENRES.index(g)] = 1.0
        movie_genres[mid] = vec
    return movie_genres


def _per_user_split(
    interactions: List[Tuple[int, int, float]],
    positive_threshold: float = 1.0,
) -> Tuple[List, List, List, Dict, Dict, Dict]:
    """Per-user 80/10/10 split (与 travel_recommender.public_dataset_loaders 一致)。"""
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


def load_movielens_beliefs(
    min_user_beliefs: int = 3,
    min_user_ratings: int = 5,
    min_movie_ratings: int = 5,
    max_users: int = 25000,
    positive_rating: float = 4.0,
    use_cache: bool = True,
) -> RecData:
    """MovieLens Beliefs: 预期评分(说) + 实际评分(做)。

    声明式 profile: userPredictRating 按电影类型聚合成 40 维偏好向量
    行为反馈: userElicitRating + user_rating_history + extra, rating>=4 为正反馈
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"movielens_beliefs_minB{min_user_beliefs}_minR{min_user_ratings}.pkl"

    if use_cache and cache_path.exists():
        _log("Loading cached MovieLens-Beliefs RecData...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    t_start = time.time()
    _log("=" * 60)
    _log("Loading MovieLens Beliefs Dataset")
    _log("=" * 60)

    # 1. 读取电影信息
    _log(f"[1/5] Reading movies.csv ...")
    movies_df = pd.read_csv(MOVIES_CSV)
    movie_genres = _build_movie_genre_matrix(movies_df)
    _log(f"  Movies: {len(movies_df)}")

    # 2. 读取信念数据，分离预期评分和实际评分
    _log(f"[2/5] Reading belief_data.csv (this may take a while)...")
    belief_df = pd.read_csv(BELIEFS_CSV)
    _log(f"  Total belief rows: {len(belief_df):,}")

    # 预期评分记录 (isSeen=0, userPredictRating有值)
    predict_df = belief_df[
        (belief_df["isSeen"] == 0) & (belief_df["userPredictRating"] > 0)
    ].copy()
    _log(f"  Predict (说) records: {len(predict_df):,}")

    # 实际评分记录 (isSeen=1, userElicitRating有值)
    elicit_df = belief_df[
        (belief_df["isSeen"] == 1) & (belief_df["userElicitRating"] > 0)
    ].copy()
    _log(f"  Elicit (做) records: {len(elicit_df):,}")

    # 3. 读取历史评分
    _log(f"[3/5] Reading user_rating_history.csv ...")
    hist_df = pd.read_csv(RATING_HIST_CSV)
    hist_df = hist_df[hist_df["rating"] >= positive_rating]
    _log(f"  Positive history ratings (>=4): {len(hist_df):,}")

    # 额外评分（Release 2）
    _log(f"[3b/5] Reading ratings_for_additional_users.csv ...")
    extra_df = pd.read_csv(EXTRA_RATINGS_CSV)
    extra_df = extra_df[extra_df["rating"] >= positive_rating]
    _log(f"  Positive extra ratings (>=4): {len(extra_df):,}")

    # 4. 筛选用户：同时有足够的预期评分和实际评分
    _log(f"[4/5] Filtering users ...")
    predict_counts = predict_df.groupby("userId").size()
    elicit_counts = elicit_df.groupby("userId").size()
    hist_counts = hist_df.groupby("userId").size()
    extra_counts = extra_df.groupby("userId").size()

    # 合并实际评分数量（elicit + history + extra）
    all_rating_counts = defaultdict(int)
    for uid, c in elicit_counts.items():
        all_rating_counts[uid] += c
    for uid, c in hist_counts.items():
        all_rating_counts[uid] += c
    for uid, c in extra_counts.items():
        all_rating_counts[uid] += c

    # 筛选：同时有足够的预期评分和实际评分
    eligible_users = []
    for uid in predict_counts.index:
        if predict_counts[uid] >= min_user_beliefs and all_rating_counts.get(uid, 0) >= min_user_ratings:
            eligible_users.append(uid)

    _log(f"  Eligible users (beliefs>={min_user_beliefs} & ratings>={min_user_ratings}): {len(eligible_users):,}")

    # 按实际评分数量降序，取 top max_users
    eligible_users.sort(key=lambda u: -all_rating_counts[u])
    if len(eligible_users) > max_users:
        eligible_users = eligible_users[:max_users]
        _log(f"  Capped to {max_users} users")

    user_set = set(eligible_users)
    user_map = {u: i for i, u in enumerate(eligible_users)}

    # 5. 构建行为反馈 interactions（实际评分 >= 4 为正反馈）
    _log(f"[5/5] Building interactions and profiles ...")
    interactions: List[Tuple[int, int, float]] = []
    item_set: Set[int] = set()

    for uid in eligible_users:
        uidx = user_map[uid]
        # elicit 实际评分
        sub = elicit_df[elicit_df["userId"] == uid]
        for _, row in sub.iterrows():
            if row["userElicitRating"] >= positive_rating:
                mid = int(row["movieId"])
                interactions.append((uidx, mid, 1.0))
                item_set.add(mid)
        # history 评分
        sub = hist_df[hist_df["userId"] == uid]
        for _, row in sub.iterrows():
            mid = int(row["movieId"])
            interactions.append((uidx, mid, 1.0))
            item_set.add(mid)
        # extra 评分
        sub = extra_df[extra_df["userId"] == uid]
        for _, row in sub.iterrows():
            mid = int(row["movieId"])
            interactions.append((uidx, mid, 1.0))
            item_set.add(mid)

    # 筛选物品：至少被 min_movie_ratings 个用户评分
    item_counts = defaultdict(int)
    for _, mid, _ in interactions:
        item_counts[mid] += 1
    keep_items = {mid for mid, c in item_counts.items() if c >= min_movie_ratings}
    item_map = {mid: i for i, mid in enumerate(sorted(keep_items))}

    # 重新映射 interactions
    interactions = [(u, item_map[m], r) for u, m, r in interactions if m in item_map]
    _log(f"  Final items (ratings>={min_movie_ratings}): {len(item_map):,}")
    _log(f"  Final interactions: {len(interactions):,}")

    # 6. 构建声明式 profile（预期评分按电影类型聚合）
    # 40维：18类型×2统计量(平均预期评分归一化/数量占比) + 全局4统计量
    feat_dim = 18 * 2 + 4  # 40维
    user_features = np.zeros((len(eligible_users), feat_dim), dtype=np.float32)

    for uid in eligible_users:
        uidx = user_map[uid]
        user_predicts = predict_df[predict_df["userId"] == uid]
        if user_predicts.empty:
            user_features[uidx, -1] = 1.0  # 无预期评分标记
            continue

        # 按类型聚合
        genre_ratings = defaultdict(list)
        genre_certainty = defaultdict(list)
        for _, row in user_predicts.iterrows():
            mid = int(row["movieId"])
            if mid not in movie_genres:
                continue
            gvec = movie_genres[mid]
            rating = float(row["userPredictRating"])
            certainty = float(row.get("userCertainty", 3.0))
            for gi, gv in enumerate(gvec):
                if gv > 0:
                    genre_ratings[gi].append(rating)
                    genre_certainty[gi].append(certainty)

        # 18类型 × 2统计量
        for gi in range(18):
            ratings_g = genre_ratings.get(gi, [])
            if ratings_g:
                # 平均预期评分归一化到 [0,1] (除以5)
                user_features[uidx, gi] = np.mean(ratings_g) / 5.0
                # 数量占比 (该类型预期评分数量 / 总预期评分数量)
                user_features[uidx, 18 + gi] = len(ratings_g) / len(user_predicts)

        # 全局统计量
        all_ratings = [float(r) for r in user_predicts["userPredictRating"]]
        user_features[uidx, 36] = np.mean(all_ratings) / 5.0 if all_ratings else 0
        user_features[uidx, 37] = np.std(all_ratings) / 5.0 if len(all_ratings) > 1 else 0
        all_certainty = [float(c) for c in user_predicts["userCertainty"] if c > 0]
        user_features[uidx, 38] = np.mean(all_certainty) / 5.0 if all_certainty else 0
        user_features[uidx, 39] = len(all_ratings) / 100.0  # 预期评分数量归一化

    # 7. 划分 train/val/test (per-user 80/10/10)
    train, val, test, train_items, val_items, test_relevant = _per_user_split(interactions)

    elapsed = time.time() - t_start
    data = RecData(
        n_users=len(eligible_users),
        n_items=len(item_map),
        train=train,
        val=val,
        test=test,
        user_features=user_features,
        train_items=dict(train_items),
        val_items=dict(val_items),
        test_relevant=dict(test_relevant),
        stats={
            "dataset": "MovieLens-Beliefs",
            "n_users": len(eligible_users),
            "n_items": len(item_map),
            "n_interactions": len(interactions),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_positive_train": sum(1 for _, _, r in train if r >= 1.0),
            "n_positive_test": sum(len(v) for v in test_relevant.values()),
            "n_test_users": len(test_relevant),
            "positive_rule": f"rating>={positive_rating}",
            "feat_dim": feat_dim,
            "profile_source": "userPredictRating aggregated by genre",
            "split": "per-user 80/10/10",
            "load_time_sec": round(elapsed, 1),
        },
    )

    os.makedirs(str(CACHE_DIR), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    _log("=" * 60)
    _log(f"MovieLens-Beliefs loaded in {elapsed:.1f}s")
    _log(f"  Users: {data.n_users:,}")
    _log(f"  Items: {data.n_items:,}")
    _log(f"  Interactions: {len(interactions):,}")
    _log(f"  Train/Val/Test: {len(train):,}/{len(val):,}/{len(test):,}")
    _log(f"  Test users (with positive): {len(test_relevant):,}")
    _log(f"  Profile dim: {feat_dim}")
    _log(f"  Cached to: {cache_path}")
    _log("=" * 60)
    return data


if __name__ == "__main__":
    data = load_movielens_beliefs(use_cache=False)
    print(f"\nDone. Dataset stats: {data.stats}")
