import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔参数。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("参数不能为空")
    return items


def reset_dir(path: Path, overwrite: bool) -> None:
    """创建输出目录，必要时先清空。"""
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    """复制或软链接单个文件。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def iter_class_images(dataset_root: Path, split: str, class_name: str) -> Iterable[Path]:
    """遍历某个分类数据集 split/class 下的图片。"""
    class_dir = dataset_root / split / class_name
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def unique_output_name(dataset_index: int, src: Path) -> str:
    """生成稳定且不易冲突的输出文件名。"""
    stem = src.stem.replace(" ", "_")
    return f"ds{dataset_index:02d}_{stem}{src.suffix.lower()}"


def merge_split(
    dataset_roots: Sequence[Path],
    output_root: Path,
    split: str,
    class_names: Sequence[str],
    copy_files: bool,
) -> int:
    """合并一个 split 的分类数据。"""
    total = 0
    for class_name in class_names:
        (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    for dataset_index, dataset_root in enumerate(dataset_roots, start=1):
        for class_name in class_names:
            images = iter_class_images(dataset_root, split, class_name)
            for src in images:
                dst = output_root / split / class_name / unique_output_name(dataset_index, src)
                link_or_copy(src, dst, copy_files)
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge folder-style recognition datasets into one dataset")
    parser.add_argument(
        "--inputs",
        default="data/animals10/recognition,data/coco_animals/recognition",
        help="逗号分隔的识别数据根目录，每个目录需包含 train/val/<class_name>",
    )
    parser.add_argument("--output", default="data/animals10_coco/recognition", help="合并后的识别数据输出目录")
    parser.add_argument("--classes", default="dog,horse,elephant,cat,cow,sheep", help="逗号分隔类别列表，顺序必须与训练配置一致")
    parser.add_argument("--splits", default="train,val", help="逗号分隔 split 列表")
    parser.add_argument("--copy-files", action="store_true", help="复制文件而不是软链接")
    parser.add_argument("--overwrite", action="store_true", help="清空输出目录后重新合并")
    args = parser.parse_args()

    dataset_roots = [Path(path) for path in parse_csv(args.inputs)]
    output_root = Path(args.output)
    class_names = parse_csv(args.classes)
    splits = parse_csv(args.splits)

    missing = [str(path) for path in dataset_roots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"输入数据集不存在: {missing}")

    reset_dir(output_root, args.overwrite)
    for split in splits:
        count = merge_split(dataset_roots, output_root, split, class_names, args.copy_files)
        print(f"[INFO] {split}: merged {count} recognition images")
    print(f"[INFO] merged recognition dataset: {output_root}")


if __name__ == "__main__":
    main()
