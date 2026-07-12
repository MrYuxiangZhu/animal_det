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
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            num_classes: 类别数量，必须与 class_names 长度一致。
            num_anchors: num_anchors 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            width_mult: width_mult 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            x: x 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
