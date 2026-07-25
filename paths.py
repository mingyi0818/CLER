"""Path configuration for experiment reproduction.

审稿人请根据本地环境修改 DATA_ROOT 与 OUTPUT_ROOT。
默认布局：将数据集放入 ./data/ 目录，结果输出到 ./experiments/results/。

If you place datasets under ./data/ (recommended), no modification is needed.
Otherwise, update DATA_ROOT to point to your dataset directory.
"""
from __future__ import annotations
import os
from pathlib import Path

# 仓库根目录（本文件所在目录）
REPO_ROOT = Path(__file__).resolve().parent

# 数据集根目录：审稿人可改为本地数据集所在路径
# 默认：仓库根目录下的 data/ 子目录
DATA_ROOT = Path(os.environ.get("CLER_DATA_ROOT", REPO_ROOT / "data"))

# 实验结果输出目录
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# 图表输出目录
FIGURES_DIR = REPO_ROOT / "figures"

# 各数据集具体路径
STRAVL_CSV = DATA_ROOT / "Stravl_Travel_Preference_Data.csv"
MOVIELENS_DIR = DATA_ROOT / "ml-1m"
AMAZON_ELECTRONICS_PATH = DATA_ROOT / "reviews_Electronics_5.json.gz"
INTTRAVEL_DIR = DATA_ROOT / "IntTravel" / "full"
INTTRAVEL_SAMPLE_DIR = DATA_ROOT / "IntTravel" / "github_sample"

# 缓存目录（用于加速数据加载）
CACHE_DIR = REPO_ROOT / "experiments" / "cache"

# 确保目录存在
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_result_path(filename: str) -> Path:
    """返回结果文件的完整路径。"""
    return RESULTS_DIR / filename


def ensure_data_ready() -> dict:
    """检查数据集是否就绪，返回各数据集的存在状态。"""
    status = {
        "Stravl-Data": STRAVL_CSV.exists(),
        "MovieLens-1M": MOVIELENS_DIR.exists(),
        "Amazon-Electronics": AMAZON_ELECTRONICS_PATH.exists(),
        "IntTravel": INTTRAVEL_DIR.exists() or INTTRAVEL_SAMPLE_DIR.exists(),
    }
    return status


if __name__ == "__main__":
    print(f"REPO_ROOT = {REPO_ROOT}")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"RESULTS_DIR = {RESULTS_DIR}")
    print()
    print("Dataset availability:")
    for name, ready in ensure_data_ready().items():
        flag = "[OK]" if ready else "[MISSING]"
        print(f"  {flag} {name}")
