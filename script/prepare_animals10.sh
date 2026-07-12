#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SRC_DIR="${1:-data/animal_classification/raw-img}"
DST_DIR="${2:-data/animals10/recognition}"
VAL_RATIO="${3:-0.2}"

python - <<'PY' "${SRC_DIR}" "${DST_DIR}" "${VAL_RATIO}"
import random
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
val_ratio = float(sys.argv[3])

name_map = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
    "dog": "dog",
    "horse": "horse",
    "elephant": "elephant",
    "butterfly": "butterfly",
    "chicken": "chicken",
    "cat": "cat",
    "cow": "cow",
    "sheep": "sheep",
    "spider": "spider",
    "squirrel": "squirrel",
}

if not src.exists():
    raise SystemExit(f"[ERROR] 源目录不存在: {src}\n请确认 Animals-10 raw-img 解压位置，或传入源目录：bash script/prepare_animals10.sh /path/to/raw-img")

suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
class_dirs = [p for p in src.iterdir() if p.is_dir()]
if not class_dirs:
    raise SystemExit(f"[ERROR] 源目录下没有类别文件夹: {src}")

random.seed(42)
summary = []
for class_dir in sorted(class_dirs):
    mapped = name_map.get(class_dir.name)
    if mapped is None:
        print(f"[WARN] 跳过未知类别目录: {class_dir.name}")
        continue
    images = sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in suffixes])
    if not images:
        print(f"[WARN] 类别 {class_dir.name} 没有图片")
        continue
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
    val_set = set(images[:val_count])
    for img in images:
        split = "val" if img in val_set else "train"
        out_dir = dst / split / mapped
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img.name
        if out_path.exists():
            stem = img.stem
            out_path = out_dir / f"{stem}_{abs(hash(str(img))) % 100000}{img.suffix.lower()}"
        shutil.copy2(img, out_path)
    summary.append((mapped, len(images) - val_count, val_count))

if not summary:
    raise SystemExit("[ERROR] 没有成功整理任何类别，请检查源目录。")

print(f"[OK] Animals-10 已整理到: {dst}")
for cls, train_n, val_n in summary:
    print(f"  {cls:12s} train={train_n:5d} val={val_n:5d}")
print("\n下一步训练：")
print("  bash script/run_train_recognition.sh openclip configs/default.yaml")
PY

# 使用方法
# bash script/prepare_animals10.sh data/animal_classification/raw-img data/animals10/recognition 0.2

