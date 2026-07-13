import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def parse_class_map(value: Optional[str], dataset_count: int) -> List[Optional[Dict[int, int]]]:
    """解析每个输入数据集的类别 ID 重映射规则。

    格式示例：
    - 空字符串：不重映射
    - "0:0,1:1,2:2;0:0,1:1"：第一个数据集和第二个数据集分别使用不同映射
    """
    if not value:
        return [None] * dataset_count
    groups = value.split(";")
    if len(groups) != dataset_count:
        raise ValueError("--class-maps 的分组数量必须与 --inputs 数量一致")
    maps: List[Optional[Dict[int, int]]] = []
    for group in groups:
        group = group.strip()
        if not group:
            maps.append(None)
            continue
        mapping: Dict[int, int] = {}
        for item in group.split(","):
            old_id, new_id = item.split(":", 1)
            mapping[int(old_id)] = int(new_id)
        maps.append(mapping)
    return maps


def iter_images(image_dir: Path) -> Iterable[Path]:
    """遍历图片目录。"""
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def remap_label_line(line: str, class_map: Optional[Dict[int, int]]) -> Optional[str]:
    """按类别映射重写 YOLO 标签行。"""
    parts = line.split()
    if len(parts) < 5:
        return None
    old_cls = int(float(parts[0]))
    if class_map is None:
        new_cls = old_cls
    else:
        if old_cls not in class_map:
            return None
        new_cls = class_map[old_cls]
    return " ".join([str(new_cls), *parts[1:]])


def merge_label_file(src_label: Path, dst_label: Path, class_map: Optional[Dict[int, int]]) -> int:
    """合并并重映射单个 YOLO 标签文件。"""
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return 0
    lines = []
    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        mapped = remap_label_line(raw_line, class_map)
        if mapped is not None:
            lines.append(mapped)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def unique_output_stem(dataset_index: int, src: Path) -> str:
    """生成稳定且不易冲突的输出文件 stem。"""
    return f"ds{dataset_index:02d}_{src.stem.replace(' ', '_')}"


def split_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """返回某个 YOLO 数据集 split 的 images 和 labels 目录。"""
    return dataset_root / split / "images", dataset_root / split / "labels"


def merge_split(
    dataset_roots: Sequence[Path],
    output_root: Path,
    split: str,
    class_maps: Sequence[Optional[Dict[int, int]]],
    copy_files: bool,
) -> Tuple[int, int]:
    """合并一个 split 的 YOLO 检测数据。"""
    out_image_dir = output_root / split / "images"
    out_label_dir = output_root / split / "labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    box_count = 0

    for dataset_index, dataset_root in enumerate(dataset_roots, start=1):
        image_dir, label_dir = split_dirs(dataset_root, split)
        for src_image in iter_images(image_dir):
            out_stem = unique_output_stem(dataset_index, src_image)
            dst_image = out_image_dir / f"{out_stem}{src_image.suffix.lower()}"
            dst_label = out_label_dir / f"{out_stem}.txt"
            src_label = label_dir / f"{src_image.stem}.txt"
            link_or_copy(src_image, dst_image, copy_files)
            box_count += merge_label_file(src_label, dst_label, class_maps[dataset_index - 1])
            image_count += 1
    return image_count, box_count


def write_dataset_yaml(output_root: Path, class_names: Sequence[str]) -> None:
    """写出 YOLO 数据集说明文件。"""
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    payload = f"path: {output_root}\ntrain: train/images\nval: val/images\nnc: {len(class_names)}\nnames:\n{names}\n"
    (output_root / "data.yaml").write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YOLO-format detection datasets into one dataset")
    parser.add_argument(
        "--inputs",
        default="data/coco_animals/detection",
        help="逗号分隔的检测数据根目录，每个目录需包含 train/images,train/labels,val/images,val/labels",
    )
    parser.add_argument("--output", default="data/animals10_coco/detection", help="合并后的检测数据输出目录")
    parser.add_argument("--classes", default="dog,horse,elephant,cat,cow,sheep", help="逗号分隔类别列表，顺序必须与训练配置一致")
    parser.add_argument("--splits", default="train,val", help="逗号分隔 split 列表")
    parser.add_argument("--class-maps", default="", help="可选类别 ID 重映射，按输入数据集用分号分组，如 '0:0,1:1;0:0,1:1'")
    parser.add_argument("--copy-files", action="store_true", help="复制文件而不是软链接")
    parser.add_argument("--overwrite", action="store_true", help="清空输出目录后重新合并")
    args = parser.parse_args()

    dataset_roots = [Path(path) for path in parse_csv(args.inputs)]
    output_root = Path(args.output)
    class_names = parse_csv(args.classes)
    splits = parse_csv(args.splits)
    class_maps = parse_class_map(args.class_maps, len(dataset_roots))

    missing = [str(path) for path in dataset_roots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"输入数据集不存在: {missing}")

    reset_dir(output_root, args.overwrite)
    for split in splits:
        image_count, box_count = merge_split(dataset_roots, output_root, split, class_maps, args.copy_files)
        print(f"[INFO] {split}: merged {image_count} detection images, {box_count} boxes")
    write_dataset_yaml(output_root, class_names)
    print(f"[INFO] merged detection dataset: {output_root}")


if __name__ == "__main__":
    main()
