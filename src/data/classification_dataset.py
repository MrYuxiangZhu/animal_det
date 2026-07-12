from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset

from src.data.transforms import image_to_tensor, resize_square


class AnimalClassificationDataset(Dataset):
    """读取按类别文件夹组织的动物分类数据集。

    目录格式示例：
    data/animal_cls/train/cat/xxx.jpg
    data/animal_cls/train/dog/xxx.jpg
    data/animal_cls/val/cat/yyy.jpg
    """

    def __init__(self, root: str, split: str, class_names: List[str], image_size: int, transform: Optional[Callable] = None) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``AnimalClassificationDataset``。
        
        Args:
            root: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            split: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            image_size: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            transform: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        self.root = Path(root) / split
        self.class_names = class_names
        self.class_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
        self.image_size = image_size
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        for cls_name in class_names:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for path in cls_dir.rglob("*"):
                if path.suffix.lower() in suffixes:
                    self.samples.append((path, self.class_to_id[cls_name]))
        if not self.samples:
            raise RuntimeError(f"没有在 {self.root} 找到分类图片，请检查数据目录。")

    def __len__(self) -> int:
        """返回数据集或容器中的样本数量，供 DataLoader 或外部迭代逻辑使用。
        
        所属类: ``AnimalClassificationDataset``。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """按索引读取一个样本，并完成必要的数据解码、预处理和标签转换。
        
        所属类: ``AnimalClassificationDataset``。
        
        Args:
            idx: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"图片读取失败: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_square(image, self.image_size)
        tensor = image_to_tensor(image)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)
