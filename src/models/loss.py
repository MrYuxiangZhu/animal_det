from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.box_ops import box_iou, xywh_to_xyxy


class DetectionLoss(nn.Module):
    """YOLO 风格检测损失，包含框回归、置信度和多标签类别 BCE。"""

    def __init__(self, anchors: List[List[float]], num_classes: int) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``DetectionLoss``。
        
        Args:
            anchors: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            num_classes: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def build_targets(self, preds: torch.Tensor, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据配置构建可复用组件，降低入口函数中的业务耦合。
        
        所属类: ``DetectionLoss``。
        
        Args:
            preds: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            targets: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        b, a, h, w, _ = preds.shape
        device = preds.device
        obj_target = torch.zeros((b, a, h, w), device=device)
        box_target = torch.zeros((b, a, h, w, 4), device=device)
        cls_target = torch.zeros((b, a, h, w, self.num_classes), device=device)
        positive = torch.zeros((b, a, h, w), dtype=torch.bool, device=device)

        for bi, boxes in enumerate(targets):
            boxes = boxes.to(device)
            for obj in boxes:
                cls_id = int(obj[0].item())
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue
                cx, cy, bw, bh = obj[1:].clamp(0.0, 1.0).tolist()
                if bw <= 0 or bh <= 0:
                    continue
                gx = min(max(int(cx * w), 0), w - 1)
                gy = min(max(int(cy * h), 0), h - 1)
                anchor_ious = box_iou(
                    xywh_to_xyxy(torch.tensor([[0.5, 0.5, bw, bh]], device=device)),
                    xywh_to_xyxy(torch.cat((torch.full((a, 2), 0.5, device=device), self.anchors), dim=1)),
                ).squeeze(0)
                best_a = int(anchor_ious.argmax().item())
                obj_target[bi, best_a, gy, gx] = 1.0
                box_target[bi, best_a, gy, gx] = torch.tensor([cx * w - gx, cy * h - gy, bw, bh], device=device)
                cls_target[bi, best_a, gy, gx, cls_id] = 1.0
                positive[bi, best_a, gy, gx] = True
        return obj_target, box_target, cls_target, positive

    def forward(self, preds: torch.Tensor, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``DetectionLoss``。
        
        Args:
            preds: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            targets: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        obj_target, box_target, cls_target, positive = self.build_targets(preds, targets)
        b, a, h, w, _ = preds.shape
        device = preds.device

        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, h, w, 2).float()
        pred_xy = preds[..., 0:2].sigmoid()
        pred_wh = preds[..., 2:4].exp() * self.anchors.view(1, a, 1, 1, 2)
        pred_box_for_loss = torch.cat((pred_xy, pred_wh), dim=-1)

        if positive.any():
            box_loss = F.smooth_l1_loss(pred_box_for_loss[positive], box_target[positive], reduction="mean")
            cls_loss = self.bce(preds[..., 5:][positive], cls_target[positive]).mean()
        else:
            box_loss = preds[..., 0:4].sum() * 0.0
            cls_loss = preds[..., 5:].sum() * 0.0

        obj_loss_raw = self.bce(preds[..., 4], obj_target)
        obj_weight = torch.where(obj_target > 0, torch.ones_like(obj_target) * 5.0, torch.ones_like(obj_target) * 0.5)
        obj_loss = (obj_loss_raw * obj_weight).mean()
        total = 5.0 * box_loss + obj_loss + cls_loss
        parts = {"total": float(total.detach().cpu()), "box": float(box_loss.detach().cpu()), "obj": float(obj_loss.detach().cpu()), "cls": float(cls_loss.detach().cpu())}
        return total, parts
