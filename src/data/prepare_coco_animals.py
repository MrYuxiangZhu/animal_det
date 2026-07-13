import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

COCO_ANIMAL_CLASSES = ["dog", "horse", "elephant", "cat", "cow", "sheep"]
COCO_80_ANIMAL_CLASSES = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_classes(value: str) -> List[str]:
    """解析逗号分隔类别列表。"""
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise ValueError("类别列表不能为空")
    return classes


def load_json(path: Path) -> Dict:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    """写入格式化 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def reset_dir(path: Path, overwrite: bool) -> None:
    """创建输出目录，必要时清空已有内容。"""
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    """把图片复制或软链接到输出目录。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def valid_bbox_xywh(bbox: Sequence[float], image_w: int, image_h: int) -> Optional[Tuple[float, float, float, float]]:
    """裁剪并校验 COCO xywh bbox。"""
    x, y, w, h = map(float, bbox)
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_w), x + max(0.0, w))
    y2 = min(float(image_h), y + max(0.0, h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 1.0 or clipped_h <= 1.0:
        return None
    return x1, y1, clipped_w, clipped_h


def coco_bbox_to_yolo(bbox: Sequence[float], image_w: int, image_h: int) -> Optional[Tuple[float, float, float, float]]:
    """把 COCO xywh bbox 转成归一化 YOLO cxcywh。"""
    clipped = valid_bbox_xywh(bbox, image_w, image_h)
    if clipped is None:
        return None
    x, y, w, h = clipped
    cx = (x + w / 2.0) / image_w
    cy = (y + h / 2.0) / image_h
    return cx, cy, w / image_w, h / image_h


def group_annotations(annotations: Iterable[Dict]) -> Dict[int, List[Dict]]:
    """按 image_id 组织标注。"""
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def category_maps(coco: Dict, class_names: Sequence[str]) -> Tuple[Dict[int, str], Dict[int, int], List[Dict]]:
    """构建 COCO category_id 到类别名和工程 class_id 的映射。"""
    wanted = set(class_names)
    category_id_to_name = {int(cat["id"]): cat["name"] for cat in coco.get("categories", [])}
    coco_id_to_class_id = {
        int(cat["id"]): class_names.index(cat["name"])
        for cat in coco.get("categories", [])
        if cat["name"] in wanted
    }
    categories = [{"id": idx + 1, "name": name, "supercategory": "animal"} for idx, name in enumerate(class_names)]
    return category_id_to_name, coco_id_to_class_id, categories


def filter_annotations_for_image(
    anns: Sequence[Dict],
    coco_id_to_class_id: Dict[int, int],
    image_w: int,
    image_h: int,
) -> List[Dict]:
    """过滤某张图里属于目标动物类别且 bbox 有效的标注。"""
    filtered = []
    for ann in anns:
        coco_category_id = int(ann.get("category_id", -1))
        if coco_category_id not in coco_id_to_class_id:
            continue
        if ann.get("iscrowd", 0):
            continue
        if valid_bbox_xywh(ann.get("bbox", []), image_w, image_h) is None:
            continue
        filtered.append(ann)
    return filtered


def build_detection_split(
    coco: Dict,
    image_root: Path,
    output_root: Path,
    split: str,
    class_names: Sequence[str],
    copy_images: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """生成当前 split 的 YOLO 检测数据和过滤后的 COCO 标注。"""
    category_id_to_name, coco_id_to_class_id, coco_categories = category_maps(coco, class_names)
    anns_by_image = group_annotations(coco.get("annotations", []))
    image_out_dir = output_root / "detection" / split / "images"
    label_out_dir = output_root / "detection" / split / "labels"
    coco_image_out_dir = output_root / "coco" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    coco_image_out_dir.mkdir(parents=True, exist_ok=True)

    filtered_images = []
    filtered_annotations = []
    next_ann_id = 1
    used_file_names = set()

    for image in coco.get("images", []):
        image_id = int(image["id"])
        image_w = int(image["width"])
        image_h = int(image["height"])
        anns = filter_annotations_for_image(anns_by_image.get(image_id, []), coco_id_to_class_id, image_w, image_h)
        if not anns:
            continue

        src_image = image_root / image["file_name"]
        if not src_image.exists():
            continue
        dst_name = image["file_name"]
        if dst_name in used_file_names:
            dst_name = f"{image_id:012d}{src_image.suffix.lower()}"
        used_file_names.add(dst_name)
        detection_image_dst = image_out_dir / dst_name
        coco_image_dst = coco_image_out_dir / dst_name
        link_or_copy(src_image, detection_image_dst, copy_images)
        link_or_copy(src_image, coco_image_dst, copy_images)

        label_lines = []
        new_image = dict(image)
        new_image["file_name"] = dst_name
        filtered_images.append(new_image)

        for ann in anns:
            class_id = coco_id_to_class_id[int(ann["category_id"])]
            yolo_bbox = coco_bbox_to_yolo(ann["bbox"], image_w, image_h)
            if yolo_bbox is None:
                continue
            cx, cy, w, h = yolo_bbox
            label_lines.append(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")

            x, y, bw, bh = valid_bbox_xywh(ann["bbox"], image_w, image_h) or ann["bbox"]
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id
            new_ann["category_id"] = class_id + 1
            new_ann["bbox"] = [float(x), float(y), float(bw), float(bh)]
            new_ann["area"] = float(bw * bh)
            filtered_annotations.append(new_ann)
            next_ann_id += 1

        label_path = label_out_dir / f"{Path(dst_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    filtered_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_categories,
    }
    write_json(output_root / "coco" / "annotations" / f"instances_{split}.json", filtered_coco)
    return filtered_images, filtered_annotations, coco_categories


def build_recognition_split(
    coco: Dict,
    image_root: Path,
    output_root: Path,
    split: str,
    class_names: Sequence[str],
    min_crop_size: int,
    crop_padding: float,
) -> int:
    """生成当前 split 的分类 crop 数据。"""
    category_id_to_name, coco_id_to_class_id, _ = category_maps(coco, class_names)
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    out_count = 0

    for ann in coco.get("annotations", []):
        coco_category_id = int(ann.get("category_id", -1))
        if coco_category_id not in coco_id_to_class_id:
            continue
        if ann.get("iscrowd", 0):
            continue
        image = images_by_id.get(int(ann["image_id"]))
        if image is None:
            continue
        src_image = image_root / image["file_name"]
        if not src_image.exists():
            continue
        image_w = int(image["width"])
        image_h = int(image["height"])
        clipped = valid_bbox_xywh(ann.get("bbox", []), image_w, image_h)
        if clipped is None:
            continue
        x, y, w, h = clipped
        if min(w, h) < min_crop_size:
            continue

        pad_x = w * crop_padding
        pad_y = h * crop_padding
        x1 = max(0, int(round(x - pad_x)))
        y1 = max(0, int(round(y - pad_y)))
        x2 = min(image_w, int(round(x + w + pad_x)))
        y2 = min(image_h, int(round(y + h + pad_y)))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = category_id_to_name[coco_category_id]
        out_dir = output_root / "recognition" / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(ann['image_id']):012d}_{int(ann['id']):012d}.jpg"
        if out_path.exists():
            out_count += 1
            continue
        try:
            with Image.open(src_image) as im:
                crop = im.convert("RGB").crop((x1, y1, x2, y2))
                crop.save(out_path, quality=95)
            out_count += 1
        except OSError:
            continue
    return out_count


def write_class_names(output_root: Path, class_names: Sequence[str]) -> None:
    """记录预处理使用的类别顺序。"""
    payload = {"class_names": list(class_names), "num_classes": len(class_names)}
    write_json(output_root / "class_names.json", payload)


def prepare_split(
    coco_root: Path,
    output_root: Path,
    split: str,
    class_names: Sequence[str],
    copy_images: bool,
    min_crop_size: int,
    crop_padding: float,
) -> None:
    """预处理单个 COCO split。"""
    year_split = f"{split}2017"
    annotation_path = coco_root / "annotations" / f"instances_{year_split}.json"
    image_root = coco_root / year_split
    if not annotation_path.exists():
        raise FileNotFoundError(f"找不到 COCO 标注文件: {annotation_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"找不到 COCO 图片目录: {image_root}")

    coco = load_json(annotation_path)
    images, annotations, _ = build_detection_split(coco, image_root, output_root, split, class_names, copy_images)
    crop_count = build_recognition_split(coco, image_root, output_root, split, class_names, min_crop_size, crop_padding)
    print(f"[INFO] {split}: detection images={len(images)}, boxes={len(annotations)}, recognition crops={crop_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COCO animal subset for detection, recognition, and COCO-format training")
    parser.add_argument("--coco-root", default="data/coco", help="原始 COCO 根目录，包含 train2017/val2017/annotations")
    parser.add_argument("--output-root", default="data/coco_animals", help="预处理输出目录")
    parser.add_argument("--classes", default=",".join(COCO_ANIMAL_CLASSES), help="逗号分隔类别列表，顺序会作为工程 class_id 顺序")
    parser.add_argument("--splits", default="train,val", help="逗号分隔 split，目前支持 train,val")
    parser.add_argument("--copy-images", action="store_true", help="复制图片而不是创建软链接")
    parser.add_argument("--overwrite", action="store_true", help="清空输出目录后重新生成")
    parser.add_argument("--min-crop-size", type=int, default=32, help="分类 crop 最小边长，小于该值的目标跳过")
    parser.add_argument("--crop-padding", type=float, default=0.08, help="分类 crop bbox 四周扩展比例")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    output_root = Path(args.output_root)
    class_names = parse_classes(args.classes)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    unsupported = sorted(set(splits) - {"train", "val"})
    if unsupported:
        raise ValueError(f"不支持的 split: {unsupported}")

    reset_dir(output_root, args.overwrite)
    write_class_names(output_root, class_names)
    for split in splits:
        prepare_split(
            coco_root=coco_root,
            output_root=output_root,
            split=split,
            class_names=class_names,
            copy_images=args.copy_images,
            min_crop_size=args.min_crop_size,
            crop_padding=args.crop_padding,
        )
    print(f"[INFO] COCO animal subset prepared at: {output_root}")


if __name__ == "__main__":
    main()
