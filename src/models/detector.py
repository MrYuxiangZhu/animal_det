from typing import Tuple

import torch
from torch import nn

from src.models.backbone import ConvBNAct, TinyBackbone


class AnimalDetector(nn.Module):
    """单尺度 anchor 检测器。

    输出张量形状为 [B, A, H, W, 5 + C]：
    5 分别表示 tx、ty、tw、th、objectness，C 表示类别 logits。
    """

    def __init__(self, num_classes: int, num_anchors: int = 3, width_mult: float = 0.75) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``AnimalDetector``。
        
        Args:
            num_classes: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            num_anchors: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            width_mult: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = TinyBackbone(width_mult=width_mult)
        self.neck = nn.Sequential(
            ConvBNAct(self.backbone.out_channels, self.backbone.out_channels, 3, 1),
            ConvBNAct(self.backbone.out_channels, self.backbone.out_channels // 2, 1, 1),
            ConvBNAct(self.backbone.out_channels // 2, self.backbone.out_channels, 3, 1),
        )
        self.head = nn.Conv2d(self.backbone.out_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``AnimalDetector``。
        
        Args:
            x: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        feats = self.neck(self.backbone(x))
        pred = self.head(feats)
        b, _, h, w = pred.shape
        pred = pred.view(b, self.num_anchors, 5 + self.num_classes, h, w)
        return pred.permute(0, 1, 3, 4, 2).contiguous()


def decode_predictions(raw: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将网络原始输出解码为归一化 cxcywh、objectness 和类别概率。"""
    b, a, h, w, _ = raw.shape
    device = raw.device
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, h, w, 2).float()
    xy = (raw[..., 0:2].sigmoid() + grid) / torch.tensor([w, h], device=device)
    wh = raw[..., 2:4].exp() * anchors.view(1, a, 1, 1, 2)
    boxes = torch.cat((xy, wh), dim=-1)
    objectness = raw[..., 4].sigmoid()
    class_probs = raw[..., 5:].sigmoid()
    return boxes, objectness, class_probs
