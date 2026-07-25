"""下载 Stravl-Data 数据集。

来源：https://github.com/Stravl/Stravl-Data
文件：Stravl_Travel_Preference_Data.csv

论文中用于：主实验、5种子稳定性、消融实验、profile噪声实验、cu分桶实验
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/Stravl/Stravl-Data/raw/main/Stravl_Travel_Preference_Data.csv"
DEST = DATA_ROOT / "Stravl_Travel_Preference_Data.csv"

if DEST.exists():
    print(f"已存在: {DEST}")
    sys.exit(0)

print(f"下载 Stravl-Data ...")
print(f"  URL: {URL}")
print(f"  目标: {DEST}")
try:
    urllib.request.urlretrieve(URL, DEST)
    size_mb = DEST.stat().st_size / (1024 * 1024)
    print(f"  [OK] 下载完成 ({size_mb:.2f} MB)")
except Exception as e:
    print(f"  [FAIL] {e}")
    print("  请手动下载：")
    print(f"    1. 访问 https://github.com/Stravl/Stravl-Data")
    print(f"    2. 下载 Stravl_Travel_Preference_Data.csv")
    print(f"    3. 放至 {DEST}")
    sys.exit(1)
