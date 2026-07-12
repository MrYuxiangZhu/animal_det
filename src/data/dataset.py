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
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            root: 数据集根目录路径；检测任务指向 detection 目录，识别任务指向 recognition 目录。
            image_dir: 相对 root 的图片目录，例如 train/images 或 val/images。
            label_dir: 相对 root 的 YOLO 标签目录，例如 train/labels 或 val/labels。
            image_size: 模型输入图像尺寸，图片会缩放或 letterbox 到该大小。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        self.root = Path(root)
        self.image_dir = self._resolve_dir(image_dir)
        self.label_dir = self._resolve_dir(label_dir)
        self.image_size = image_size
        suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted([p for p in self.image_dir.rglob("*") if p.suffix.lower() in suffixes])
        if not self.images:
            raise RuntimeError(
                f"没有在 {self.image_dir} 找到图片。请按 data/<dataset_name>/detection/train/images 这类结构整理数据，"
                f"或检查 configs/default.yaml 中 data.root/train_images/val_images。"
            )

    def _resolve_dir(self, relative_or_abs: str) -> Path:
        """将配置中的相对路径解析到当前数据集根目录下。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            relative_or_abs: relative_or_abs 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            解析后的目录路径。
        """
        path = Path(relative_or_abs)
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        """返回数据集或容器中的样本数量，供 DataLoader 或外部迭代逻辑使用。
        
        所属类: ``AnimalDetectionDataset``。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        """按索引读取一个样本，并完成必要的数据解码、预处理和标签转换。
        
        所属类: ``AnimalDetectionDataset``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            idx: 样本索引，由 PyTorch DataLoader 传入，用于读取指定图片和标签。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
