import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from src.utils.logger import setup_logger


ANIMAL_ALIASES: Dict[str, str] = {
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "cow": "cow",
    "sheep": "sheep",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
}


def iter_yolo_labels(label_path: Path) -> Iterable[List[float]]:
    """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
    
    Args:
        label_path: 文件路径参数，函数会读取该文件或将结果写入该位置。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([float(x) for x in line.split()[:5]])
    return rows


def split_existing_yolo(source_root: str, output_root: str, val_ratio: float, seed: int) -> None:
    """把公开下载并转换好的 YOLO 数据重新整理为本工程目录结构。"""
    logger = setup_logger("prepare_dataset")
    source = Path(source_root)
    output = Path(output_root)
    images = sorted([p for p in (source / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.Random(seed).shuffle(images)
    val_count = int(len(images) * val_ratio)
    splits = {"val": images[:val_count], "train": images[val_count:]}

    for split, split_images in splits.items():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            label_path = source / "labels" / f"{image_path.stem}.txt"
            shutil.copy2(image_path, output / "images" / split / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, output / "labels" / split / label_path.name)
            else:
                (output / "labels" / split / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
        logger.info("%s: %d 张图片", split, len(split_images))


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="Prepare public animal dataset already converted to YOLO format")
    parser.add_argument("--source-root", required=True, help="包含 images/ 和 labels/ 的公开数据集目录")
    parser.add_argument("--output-root", default="data/animal_detection")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_existing_yolo(args.source_root, args.output_root, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
