"""下载 Amazon-Electronics 5-core 数据集。

来源：https://jmcauley.ucsd.edu/data/amazon/
文件：reviews_Electronics_5.json.gz

论文中用于：跨数据集验证（表20-21）
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Electronics_5.json.gz"
DEST = DATA_ROOT / "reviews_Electronics_5.json.gz"

if DEST.exists():
    print(f"已存在: {DEST}")
    sys.exit(0)

print(f"下载 Amazon-Electronics 5-core ...")
print(f"  URL: {URL}")
print(f"  目标: {DEST}")
try:
    urllib.request.urlretrieve(URL, DEST)
    size_mb = DEST.stat().st_size / (1024 * 1024)
    print(f"  [OK] 下载完成 ({size_mb:.2f} MB)")
except Exception as e:
    print(f"  [FAIL] {e}")
    print("  请手动下载：")
    print(f"    1. 访问 https://jmcauley.ucsd.edu/data/amazon/")
    print(f"    2. 选择 'Per-category files' 下的 Electronics_5.json.gz")
    print(f"    3. 重命名为 reviews_Electronics_5.json.gz 并放至 {DEST}")
    sys.exit(1)
