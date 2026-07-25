"""数据集下载脚本统一入口。

运行：
    python data_scripts/download_all.py

将在仓库根目录下创建 data/ 子目录并下载所有数据集。
每个数据集也有独立下载脚本（download_stravl.py 等）。
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print(f"数据集将下载至: {DATA_ROOT}")
print()

scripts = [
    ("Stravl-Data", "download_stravl.py"),
    ("MovieLens-1M", "download_movielens.py"),
    ("Amazon-Electronics", "download_amazon.py"),
    ("IntTravel", "download_inttravel.py"),
]

for name, script in scripts:
    print(f"[1/4] 下载 {name} ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / script)],
        cwd=str(REPO_ROOT)
    )
    if result.returncode != 0:
        print(f"  [WARNING] {name} 下载失败，请参考 README.md 手动下载")
    print()

print("下载流程结束。如部分数据集下载失败，请参考 README.md 中的手动下载说明。")
