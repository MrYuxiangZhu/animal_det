from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset

from src.data.transforms import image_to_tensor, letterbox


class AnimalDetectionDataset(Dataset):
    """读取 YOLO 格式标注的数据集。

    每张图片对应一个同名 txt 文件，每行格式：
    class_id center_x center_y width height
    坐标均相对原图宽高归一化到 0-1。
    """

    def __init__(self, root: str, image_dir: str, label_dir: str, image_size: int) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``AnimalDetectionDataset``。
        
        Args:
            root: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            image_dir: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            label_dir: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            image_size: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.label_dir = self.root / label_dir
        self.image_size = image_size
        suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted([p for p in self.image_dir.rglob("*") if p.suffix.lower() in suffixes])
        if not self.images:
            raise RuntimeError(f"没有在 {self.image_dir} 找到图片。")

    def __len__(self) -> int:
        """返回数据集或容器中的样本数量，供 DataLoader 或外部迭代逻辑使用。
        
        所属类: ``AnimalDetectionDataset``。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        """按索引读取一个样本，并完成必要的数据解码、预处理和标签转换。
        
        所属类: ``AnimalDetectionDataset``。
        
        Args:
            idx: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"图片读取失败: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        image, scale, (pad_x, pad_y) = letterbox(image, self.image_size)

        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes: List[List[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = map(float, line.split())
                abs_cx = cx * original_w * scale + pad_x
                abs_cy = cy * original_h * scale + pad_y
                abs_w = w * original_w * scale
                abs_h = h * original_h * scale
                boxes.append([cls, abs_cx / self.image_size, abs_cy / self.image_size, abs_w / self.image_size, abs_h / self.image_size])

        target = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)
        meta = {"image_path": str(image_path)}
        return image_to_tensor(image), target, meta


def detection_collate(batch):
    """检测任务每张图目标数量不同，因此 targets 保持列表。"""
    images, targets, metas = zip(*batch)
    return torch.stack(images, dim=0), list(targets), list(metas)
