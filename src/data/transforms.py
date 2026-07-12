from typing import Tuple

import cv2
import numpy as np
import torch


def letterbox(image: np.ndarray, image_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """等比例缩放并填充到正方形，返回图像、缩放比例和左上填充量。"""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("输入图像尺寸非法。")
    scale = min(image_size / h, image_size / w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def resize_square(image: np.ndarray, image_size: int) -> np.ndarray:
    """直接缩放到正方形，主要用于分类模型。"""
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """BGR/RGB uint8 图像转为归一化 CHW Tensor。"""
    ### `.permute(2, 0, 1)` 通道维度转换（关键）

    #  OpenCV/PIL/numpy 默认格式：`[H, W, C]` 高、宽、通道
    # PyTorch CNN /timm/ OpenCLIP 强制要求输入格式：`[C, H, W]` 通道、高、宽
    # permute(2,0,1) 代表重新排列维度顺序：
    # 原第 2 维（通道 C）放到最前面
    # 原第 0 维（高度 H）放中间
    # 原第 1 维（宽度 W）放最后

    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()
