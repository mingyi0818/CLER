"""下载 IntTravel 数据集。

来源：https://github.com/haoy2000/IntTravel
文件：interaction_1.csv, interaction_2.csv, interaction_3.csv (full/)

论文中用于：IntTravel 三分片实验（表19, 表22-23）、XSimGCL 对比
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DEST_DIR = DATA_ROOT / "IntTravel" / "full"
DEST_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/haoy2000/IntTravel/main/data/full"
FILES = ["interaction_1.csv", "interaction_2.csv", "interaction_3.csv"]

all_ok = True
for fname in FILES:
    dest = DEST_DIR / fname
    if dest.exists():
        print(f"已存在: {dest}")
        continue

    url = f"{BASE_URL}/{fname}"
    print(f"下载 {fname} ...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"  [OK] ({size_kb:.2f} KB)")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_ok = False

if not all_ok:
    print()
    print("部分文件下载失败，请手动下载：")
    print(f"    1. 访问 https://github.com/haoy2000/IntTravel")
    print(f"    2. 下载 data/full/ 下的三个 interaction_*.csv 文件")
    print(f"    3. 放至 {DEST_DIR}")
    sys.exit(1)
else:
    print(f"\n[OK] IntTravel 数据集就绪: {DEST_DIR}")
