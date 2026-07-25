"""下载 MovieLens-1M 数据集。

来源：https://grouplens.org/datasets/movielens/1m/
文件：ml-1m.zip (解压后含 ratings.dat, users.dat, movies.dat)

论文中用于：跨数据集验证（表20-21）、Beliefs 5种子实验
"""
from __future__ import annotations
import sys
import zipfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH = DATA_ROOT / "ml-1m.zip"
DEST_DIR = DATA_ROOT / "ml-1m"

if DEST_DIR.exists() and any(DEST_DIR.iterdir()):
    print(f"已存在: {DEST_DIR}")
    sys.exit(0)

print(f"下载 MovieLens-1M ...")
print(f"  URL: {URL}")
print(f"  目标: {DEST_DIR}")
try:
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print(f"  [OK] 下载完成，正在解压 ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    ZIP_PATH.unlink()
    print(f"  [OK] 解压完成: {DEST_DIR}")
except Exception as e:
    print(f"  [FAIL] {e}")
    print("  请手动下载：")
    print(f"    1. 访问 https://grouplens.org/datasets/movielens/1m/")
    print(f"    2. 下载 ml-1m.zip")
    print(f"    3. 解压至 {DEST_DIR}")
    sys.exit(1)
