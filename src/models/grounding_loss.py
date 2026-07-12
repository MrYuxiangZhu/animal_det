from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.grounding_dino_like import decode_grounding_boxes


class GroundingDetectionLoss(nn.Module):
    """文本条件检测损失，用类别文本查询监督网格级开放词汇检测。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def build_targets(self, box_raw: torch.Tensor, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, w, _ = box_raw.shape
        device = box_raw.device
        obj_target = torch.zeros((b, h, w), device=device)
        box_target = torch.zeros((b, h, w, 4), device=device)
        cls_target = torch.zeros((b, self.num_classes, h, w), device=device)
        positive = torch.zeros((b, h, w), dtype=torch.bool, device=device)
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
                obj_target[bi, gy, gx] = 1.0
                box_target[bi, gy, gx] = torch.tensor([cx, cy, bw, bh], device=device)
                cls_target[bi, cls_id, gy, gx] = 1.0
                positive[bi, gy, gx] = True
        return obj_target, box_target, cls_target, positive

    def forward(self, box_raw: torch.Tensor, objectness: torch.Tensor, class_logits: torch.Tensor, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        obj_target, box_target, cls_target, positive = self.build_targets(box_raw, targets)
        pred_boxes = decode_grounding_boxes(box_raw)
        if positive.any():
            box_loss = F.smooth_l1_loss(pred_boxes[positive], box_target[positive], reduction="mean")
            cls_loss = self.bce(class_logits, cls_target).permute(0, 2, 3, 1)[positive].mean()
        else:
            box_loss = box_raw.sum() * 0.0
            cls_loss = class_logits.sum() * 0.0
        obj_weight = torch.where(obj_target > 0, torch.ones_like(obj_target) * 5.0, torch.ones_like(obj_target) * 0.5)
        obj_loss = (self.bce(objectness, obj_target) * obj_weight).mean()
        total = 5.0 * box_loss + obj_loss + cls_loss
        parts = {"total": float(total.detach().cpu()), "box": float(box_loss.detach().cpu()), "obj": float(obj_loss.detach().cpu()), "cls": float(cls_loss.detach().cpu())}
        return total, parts
