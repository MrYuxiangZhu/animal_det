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
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            root: 数据集根目录路径；检测任务指向 detection 目录，识别任务指向 recognition 目录。
            split: 数据划分名称，通常为 train、val 或 test，用于选择对应子目录。
            class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
            image_size: 模型输入图像尺寸，图片会缩放或 letterbox 到该大小。
            transform: transform 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """按索引读取一个样本，并完成必要的数据解码、预处理和标签转换。
        
        所属类: ``AnimalClassificationDataset``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            idx: 样本索引，由 PyTorch DataLoader 传入，用于读取指定图片和标签。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
