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
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
