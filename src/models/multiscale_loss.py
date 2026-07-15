from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.box_ops import box_iou, xywh_to_xyxy


def bbox_ciou_xyxy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute CIoU between aligned xyxy boxes."""
    pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(-1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target.unbind(-1)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    pred_w = (pred_x2 - pred_x1).clamp(min=eps)
    pred_h = (pred_y2 - pred_y1).clamp(min=eps)
    tgt_w = (tgt_x2 - tgt_x1).clamp(min=eps)
    tgt_h = (tgt_y2 - tgt_y1).clamp(min=eps)
    pred_area = pred_w * pred_h
    tgt_area = tgt_w * tgt_h
    union = pred_area + tgt_area - inter
    iou = inter / union.clamp(min=eps)

    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    tgt_cx = (tgt_x1 + tgt_x2) / 2
    tgt_cy = (tgt_y1 + tgt_y2) / 2
    center_dist = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

    enc_x1 = torch.minimum(pred_x1, tgt_x1)
    enc_y1 = torch.minimum(pred_y1, tgt_y1)
    enc_x2 = torch.maximum(pred_x2, tgt_x2)
    enc_y2 = torch.maximum(pred_y2, tgt_y2)
    enc_diag = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2)

    v = (4 / (torch.pi**2)) * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=eps)
    return iou - center_dist / enc_diag.clamp(min=eps) - alpha * v


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss with an explicit valid mask."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(targets > 0, prob, 1 - prob)
    alpha_t = torch.where(targets > 0, torch.full_like(targets, alpha), torch.full_like(targets, 1 - alpha))
    loss = alpha_t * (1 - pt).pow(gamma) * bce
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum().clamp(min=1.0)


class MultiScaleDetectionLoss(nn.Module):
    """YOLO-style multi-scale loss with CIoU, focal objectness and ignore mask."""

    def __init__(
        self,
        anchors: List[List[List[float]]],
        num_classes: int,
        box_gain: float = 5.0,
        obj_gain: float = 1.0,
        cls_gain: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ignore_iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.num_classes = num_classes
        self.box_gain = box_gain
        self.obj_gain = obj_gain
        self.cls_gain = cls_gain
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ignore_iou_threshold = ignore_iou_threshold

    def decode_scale(self, preds: torch.Tensor, scale_idx: int) -> torch.Tensor:
        b, a, h, w, _ = preds.shape
        device = preds.device
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, h, w, 2).float()
        xy = (preds[..., 0:2].sigmoid() + grid) / torch.tensor([w, h], device=device)
        wh = preds[..., 2:4].exp().clamp(max=1e4) * self.anchors[scale_idx].view(1, a, 1, 1, 2)
        return torch.cat((xy, wh), dim=-1)

    def build_targets(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]):
        device = outputs[0].device
        num_scales = len(outputs)
        obj_targets = []
        cls_targets = []
        box_targets = []
        positive_masks = []
        ignore_masks = []

        for preds in outputs:
            b, a, h, w, _ = preds.shape
            obj_targets.append(torch.zeros((b, a, h, w), device=device))
            cls_targets.append(torch.zeros((b, a, h, w, self.num_classes), device=device))
            box_targets.append(torch.zeros((b, a, h, w, 4), device=device))
            positive_masks.append(torch.zeros((b, a, h, w), dtype=torch.bool, device=device))
            ignore_masks.append(torch.zeros((b, a, h, w), dtype=torch.bool, device=device))

        flat_anchors = self.anchors.view(-1, 2)
        anchors_per_scale = self.anchors.shape[1]

        for bi, boxes in enumerate(targets):
            boxes = boxes.to(device)
            for obj in boxes:
                cls_id = int(obj[0].item())
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue
                cx, cy, bw, bh = obj[1:].clamp(0.0, 1.0)
                if bw <= 0 or bh <= 0:
                    continue

                gt_wh_box = xywh_to_xyxy(torch.tensor([[0.5, 0.5, bw, bh]], device=device))
                anchor_boxes = xywh_to_xyxy(torch.cat((torch.full((flat_anchors.shape[0], 2), 0.5, device=device), flat_anchors), dim=1))
                anchor_ious = box_iou(gt_wh_box, anchor_boxes).squeeze(0)
                best_flat_idx = int(anchor_ious.argmax().item())
                scale_idx = best_flat_idx // anchors_per_scale
                anchor_idx = best_flat_idx % anchors_per_scale

                _, _, h, w, _ = outputs[scale_idx].shape
                gx = min(max(int(cx.item() * w), 0), w - 1)
                gy = min(max(int(cy.item() * h), 0), h - 1)

                obj_targets[scale_idx][bi, anchor_idx, gy, gx] = 1.0
                cls_targets[scale_idx][bi, anchor_idx, gy, gx, cls_id] = 1.0
                box_targets[scale_idx][bi, anchor_idx, gy, gx] = torch.stack((cx, cy, bw, bh))
                positive_masks[scale_idx][bi, anchor_idx, gy, gx] = True

        with torch.no_grad():
            for scale_idx in range(num_scales):
                pred_boxes = self.decode_scale(outputs[scale_idx], scale_idx)
                b, a, h, w, _ = pred_boxes.shape
                for bi, boxes in enumerate(targets):
                    boxes = boxes.to(device)
                    if boxes.numel() == 0:
                        continue
                    valid = (boxes[:, 0] >= 0) & (boxes[:, 0] < self.num_classes)
                    valid_boxes = boxes[valid]
                    if valid_boxes.numel() == 0:
                        continue
                    gt_xyxy = xywh_to_xyxy(valid_boxes[:, 1:].clamp(0.0, 1.0))
                    pred_xyxy = xywh_to_xyxy(pred_boxes[bi].reshape(-1, 4))
                    max_iou = box_iou(pred_xyxy, gt_xyxy).max(dim=1).values.view(a, h, w)
                    ignore_masks[scale_idx][bi] = max_iou > self.ignore_iou_threshold
                ignore_masks[scale_idx] = ignore_masks[scale_idx] & ~positive_masks[scale_idx]

        return obj_targets, cls_targets, box_targets, positive_masks, ignore_masks

    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        obj_targets, cls_targets, box_targets, positive_masks, ignore_masks = self.build_targets(outputs, targets)
        box_losses = []
        obj_losses = []
        cls_losses = []

        for scale_idx, preds in enumerate(outputs):
            pred_boxes = self.decode_scale(preds, scale_idx)
            positive = positive_masks[scale_idx]
            if positive.any():
                pred_xyxy = xywh_to_xyxy(pred_boxes[positive])
                target_xyxy = xywh_to_xyxy(box_targets[scale_idx][positive])
                ciou = bbox_ciou_xyxy(pred_xyxy, target_xyxy)
                box_loss = (1 - ciou).mean()
                cls_loss = F.binary_cross_entropy_with_logits(preds[..., 5:][positive], cls_targets[scale_idx][positive], reduction="mean")
            else:
                box_loss = preds[..., 0:4].sum() * 0.0
                cls_loss = preds[..., 5:].sum() * 0.0

            valid_obj_mask = ~ignore_masks[scale_idx]
            obj_loss = focal_bce_with_logits(
                preds[..., 4],
                obj_targets[scale_idx],
                valid_obj_mask,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            box_losses.append(box_loss)
            obj_losses.append(obj_loss)
            cls_losses.append(cls_loss)

        box_total = torch.stack(box_losses).mean()
        obj_total = torch.stack(obj_losses).mean()
        cls_total = torch.stack(cls_losses).mean()
        total = self.box_gain * box_total + self.obj_gain * obj_total + self.cls_gain * cls_total
        parts = {
            "total": float(total.detach().cpu()),
            "box": float(box_total.detach().cpu()),
            "obj": float(obj_total.detach().cpu()),
            "cls": float(cls_total.detach().cpu()),
        }
        return total, parts
