import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔参数。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("参数不能为空")
    return items


def count_recognition(root: Path, splits: Sequence[str], class_names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """统计按类别文件夹组织的识别数据分布。"""
    counts: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_counts = {}
        for class_name in class_names:
            class_dir = root / split / class_name
            if not class_dir.exists():
                split_counts[class_name] = 0
                continue
            split_counts[class_name] = sum(1 for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        counts[split] = split_counts
    return counts


def count_detection(root: Path, splits: Sequence[str], class_names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """统计 YOLO 检测标签里的 bbox 类别分布。"""
    counts: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_counts = {class_name: 0 for class_name in class_names}
        label_dir = root / split / "labels"
        if not label_dir.exists():
            counts[split] = split_counts
            continue
        for label_path in sorted(label_dir.rglob("*.txt")):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(class_names):
                    split_counts[class_names[class_id]] += 1
        counts[split] = split_counts
    return counts


def totals_by_split(counts: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """统计每个 split 的总数。"""
    return {split: sum(class_counts.values()) for split, class_counts in counts.items()}


def totals_by_class(counts: Dict[str, Dict[str, int]], class_names: Sequence[str]) -> Dict[str, int]:
    """统计每个类别跨 split 的总数。"""
    totals = {class_name: 0 for class_name in class_names}
    for class_counts in counts.values():
        for class_name in class_names:
            totals[class_name] += class_counts.get(class_name, 0)
    return totals


def save_distribution_chart(counts: Dict[str, Dict[str, int]], class_names: Sequence[str], output_path: Path, title: str) -> None:
    """保存按 split 分组的类别分布柱状图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits = list(counts.keys())
    x = list(range(len(class_names)))
    width = 0.8 / max(len(splits), 1)

    plt.figure(figsize=(max(12, len(class_names) * 1.1), 7))
    for split_index, split in enumerate(splits):
        values = [counts[split].get(class_name, 0) for class_name in class_names]
        offsets = [idx - 0.4 + width / 2 + split_index * width for idx in x]
        plt.bar(offsets, values, width=width, label=split)
        for bar_x, value in zip(offsets, values):
            if value > 0:
                plt.text(bar_x, value, str(value), ha="center", va="bottom", fontsize=8, rotation=90)

    plt.xticks(x, class_names, rotation=35, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_total_chart(totals: Dict[str, int], output_path: Path, title: str) -> None:
    """保存总类别分布柱状图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(totals.keys())
    values = list(totals.values())
    plt.figure(figsize=(max(12, len(names) * 1.1), 7))
    plt.bar(names, values)
    for idx, value in enumerate(values):
        if value > 0:
            plt.text(idx, value, str(value), ha="center", va="bottom", fontsize=8, rotation=90)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_summary(counts: Dict[str, Dict[str, int]], class_names: Sequence[str], output_path: Path, dataset_type: str, root: Path) -> None:
    """保存统计摘要 JSON。"""
    payload = {
        "type": dataset_type,
        "root": str(root),
        "class_names": list(class_names),
        "counts": counts,
        "totals_by_split": totals_by_split(counts),
        "totals_by_class": totals_by_class(counts, class_names),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_default_root(dataset_type: str) -> str:
    """按数据类型返回默认根目录。"""
    if dataset_type == "recognition":
        return "data/coco_animals_10cls/recognition"
    return "data/coco_animals_10cls/detection"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stat and plot dataset class distribution")
    parser.add_argument("--type", choices=["recognition", "detection"], default="recognition", help="统计识别分类目录或 YOLO 检测标签")
    parser.add_argument("--root", default=None, help="数据根目录。recognition 指向 recognition 根目录，detection 指向 detection 根目录")
    parser.add_argument("--classes", default="bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe", help="逗号分隔类别列表，顺序必须与标签 class_id 一致")
    parser.add_argument("--splits", default="train,val", help="逗号分隔 split 列表")
    parser.add_argument("--output-dir", default="outputs/dataset_stats", help="统计图片和 JSON 输出目录")
    parser.add_argument("--name", default="coco_animals_10cls", help="输出文件名前缀")
    args = parser.parse_args()

    dataset_root = Path(args.root or infer_default_root(args.type))
    class_names = parse_csv(args.classes)
    splits = parse_csv(args.splits)
    output_dir = Path(args.output_dir)

    if args.type == "recognition":
        counts = count_recognition(dataset_root, splits, class_names)
        title = f"Recognition Class Distribution: {args.name}"
    else:
        counts = count_detection(dataset_root, splits, class_names)
        title = f"Detection Box Distribution: {args.name}"

    grouped_chart = output_dir / f"{args.name}_{args.type}_by_split.png"
    total_chart = output_dir / f"{args.name}_{args.type}_total.png"
    summary_json = output_dir / f"{args.name}_{args.type}_summary.json"

    save_distribution_chart(counts, class_names, grouped_chart, title)
    save_total_chart(totals_by_class(counts, class_names), total_chart, f"Total {title}")
    write_summary(counts, class_names, summary_json, args.type, dataset_root)

    print(f"[INFO] saved grouped chart: {grouped_chart}")
    print(f"[INFO] saved total chart: {total_chart}")
    print(f"[INFO] saved summary json: {summary_json}")
    print(f"[INFO] totals by split: {totals_by_split(counts)}")


if __name__ == "__main__":
    main()
