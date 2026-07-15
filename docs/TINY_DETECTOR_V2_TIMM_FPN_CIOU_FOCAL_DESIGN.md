# TINY DETECTOR V2 TIMM FPN CIOU FOCAL DESIGN

本文档给出 `tiny_detector` 升级方案：

```text
timm 预训练 backbone
+ FPN 多尺度检测头
+ CIoU box loss
+ Focal objectness loss
+ ignore mask
```

目标是把当前教学型单尺度 Tiny Detector 升级为更接近 YOLO/FPN 思路的轻量工业化检测器，同时仍保持代码可读、可控、适合本项目维护。

---

## 1. 为什么要升级

当前 `tiny_detector` 的核心问题不是“没有 backbone”，而是：

```text
有 backbone，但 backbone 从零训练；
只有单尺度 13x13 检测头；
loss 比较简化；
没有 ignore mask；
没有 IoU 类 box loss；
objectness 正负样本极不均衡。
```

在 `coco_animals_10cls` 上，训练 loss 能下降到很低，但验证 loss 反升，说明模型记住训练集后泛化不足。

OpenCLIP loss 更稳定的关键原因是 OpenCLIP 通常依赖大规模预训练模型；而当前 Tiny Detector 的视觉特征完全从零学起。

因此升级方向应围绕三点：

1. 用预训练 backbone 提升通用视觉特征；
2. 用 FPN 多尺度检测头提升小中大目标检测能力；
3. 用更成熟的检测 loss 提升定位和 objectness 学习稳定性。

---

## 2. 总体设计

升级后的模型命名建议：

```text
TimmFPNDetector
```

整体结构：

```text
Input image [B, 3, H, W]
        |
        v
timm pretrained backbone features_only=True
        |
        +--> C3: stride 8 feature
        +--> C4: stride 16 feature
        +--> C5: stride 32 feature
        |
        v
FPN top-down feature fusion
        |
        +--> P3: stride 8  feature, small objects
        +--> P4: stride 16 feature, medium objects
        +--> P5: stride 32 feature, large objects
        |
        v
Detection heads per scale
        |
        +--> raw prediction scale 0: [B, A, H/8,  W/8,  5+C]
        +--> raw prediction scale 1: [B, A, H/16, W/16, 5+C]
        +--> raw prediction scale 2: [B, A, H/32, W/32, 5+C]
```

---

## 3. 设计目标

### 3.1 目标一：提升泛化能力

使用 `timm` 预训练模型作为 backbone，例如：

```text
mobilenetv3_small_100
efficientnet_b0
resnet18
convnext_tiny
```

原因：

- 预训练模型已经学到边缘、纹理、部件、形状等通用视觉特征；
- 当前动物检测数据不需要从零学习全部视觉模式；
- 验证集 loss 更容易稳定下降；
- 小数据集上尤其有效。

### 3.2 目标二：改善小目标检测

当前单尺度 `13x13` 输出只适合大目标。COCO animal 中小目标很多，因此需要多尺度：

```text
P3 stride 8  -> 小目标
P4 stride 16 -> 中目标
P5 stride 32 -> 大目标
```

原因：

- 小目标在深层特征图上可能消失；
- 浅层特征空间分辨率高，有利于定位小目标；
- 深层特征语义强，有利于分类和大目标。

### 3.3 目标三：让 loss 更接近检测指标

当前 SmoothL1 不直接优化 IoU。升级为 CIoU：

```text
CIoU = IoU + center distance penalty + aspect ratio penalty
```

目的：

- 直接优化预测框和真实框的重叠质量；
- 同时约束中心点、宽高和长宽比；
- 比 SmoothL1 更符合目标检测评价指标。

### 3.4 目标四：稳定 objectness 学习

objectness 正负样本极不平衡。升级为 Focal Loss：

```text
FL = alpha * (1 - pt)^gamma * BCE
```

目的：

- 降低大量简单负样本的影响；
- 让模型关注难样本；
- 避免 objectness 被背景主导。

### 3.5 目标五：增加 ignore mask

YOLO 类方法中，非 best anchor 但与 GT 高重叠的预测不应被当作普通负样本惩罚。

目的：

- 减少“合理预测被当负样本打压”；
- 提升 objectness 稳定性；
- 改善验证集泛化。

---

## 4. 配置设计

建议在 `configs/coco_animals_10cls.yaml` 中增加：

```yaml
tiny_detector_v2:
  backbone_name: mobilenetv3_small_100
  pretrained: true
  out_channels: 128
  freeze_backbone_epochs: 3
  num_classes: 10
  num_scales: 3
  anchors:
    - [[0.025, 0.035], [0.045, 0.060], [0.070, 0.090]]
    - [[0.100, 0.130], [0.160, 0.200], [0.230, 0.280]]
    - [[0.320, 0.380], [0.450, 0.550], [0.650, 0.750]]
  strides: [8, 16, 32]
  epochs: 50
  batch_size: 8
  learning_rate: 0.0003
  backbone_learning_rate: 0.00003
  weight_decay: 0.001
  checkpoint_interval: 1
  resume: ""
  device: auto
  box_gain: 5.0
  obj_gain: 1.0
  cls_gain: 1.0
  focal_alpha: 0.25
  focal_gamma: 2.0
  ignore_iou_threshold: 0.5
```

说明：

- `backbone_learning_rate` 小于 head 学习率，避免破坏预训练特征；
- `freeze_backbone_epochs` 前几轮冻结 backbone，让 head 先适配；
- anchors 按尺度分组，每个尺度 3 个 anchor；
- `ignore_iou_threshold` 用于决定哪些负样本不参与 objectness loss。

---

## 5. 完整代码一：模型文件

建议新增文件：

```text
src/models/timm_fpn_detector.py
```

完整代码如下：

```python
from typing import List, Sequence, Tuple

import torch
from torch import nn

try:
    import timm
except ImportError as exc:
    raise ImportError("TimmFPNDetector requires timm. Please install timm first.") from exc

from src.models.backbone import ConvBNAct


class FPN(nn.Module):
    """Lightweight top-down FPN for three backbone feature levels."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"FPN expects 3 feature levels, got {len(in_channels)}")
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels])
        self.output_convs = nn.ModuleList([ConvBNAct(out_channels, out_channels, 3, 1) for _ in in_channels])
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + self.upsample(p5)
        p3 = self.lateral_convs[0](c3) + self.upsample(p4)
        return [
            self.output_convs[0](p3),
            self.output_convs[1](p4),
            self.output_convs[2](p5),
        ]


class DetectionHead(nn.Module):
    """Per-scale YOLO-style detection head."""

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, 1),
            ConvBNAct(in_channels, in_channels, 3, 1),
        )
        self.pred = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.pred(self.stem(x))
        b, _, h, w = pred.shape
        pred = pred.view(b, self.num_anchors, 5 + self.num_classes, h, w)
        return pred.permute(0, 1, 3, 4, 2).contiguous()


class TimmFPNDetector(nn.Module):
    """Pretrained timm backbone + FPN + multi-scale YOLO-style heads."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        out_channels: int = 128,
        num_anchors: int = 3,
        out_indices: Tuple[int, int, int] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        feature_channels = self.backbone.feature_info.channels()
        if len(feature_channels) != 3:
            raise ValueError(f"Expected 3 feature channels from timm, got {feature_channels}")
        self.fpn = FPN(feature_channels, out_channels)
        self.heads = nn.ModuleList([DetectionHead(out_channels, num_anchors, num_classes) for _ in range(3)])

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        pyramid = self.fpn(features)
        return [head(feat) for head, feat in zip(self.heads, pyramid)]


def decode_multiscale_predictions(raw_outputs: List[torch.Tensor], anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode multi-scale raw outputs into normalized cxcywh boxes, objectness and class probabilities."""
    all_boxes = []
    all_obj = []
    all_cls = []
    for scale_idx, raw in enumerate(raw_outputs):
        b, a, h, w, _ = raw.shape
        device = raw.device
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, h, w, 2).float()
        xy = (raw[..., 0:2].sigmoid() + grid) / torch.tensor([w, h], device=device)
        wh = raw[..., 2:4].exp().clamp(max=1e4) * anchors[scale_idx].view(1, a, 1, 1, 2)
        boxes = torch.cat((xy, wh), dim=-1).view(b, -1, 4)
        obj = raw[..., 4].sigmoid().view(b, -1)
        cls_probs = raw[..., 5:].sigmoid().view(b, -1, raw.shape[-1] - 5)
        all_boxes.append(boxes)
        all_obj.append(obj)
        all_cls.append(cls_probs)
    return torch.cat(all_boxes, dim=1), torch.cat(all_obj, dim=1), torch.cat(all_cls, dim=1)
```

---

## 6. 完整代码二：CIoU、Focal、Ignore Mask 多尺度 Loss

建议新增文件：

```text
src/models/multiscale_loss.py
```

完整代码如下：

```python
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

    v = (4 / (torch.pi ** 2)) * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=eps)
    return iou - center_dist / enc_diag.clamp(min=eps) - alpha * v


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
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
                    valid_boxes = boxes[(boxes[:, 0] >= 0) & (boxes[:, 0] < self.num_classes)] if boxes.numel() else boxes
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
```

---

## 7. 完整代码三：训练入口

建议新增文件：

```text
src/trainers/tiny_detector_v2.py
```

完整代码如下：

```python
import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import AnimalDetectionDataset, detection_collate
from src.models.multiscale_loss import MultiScaleDetectionLoss
from src.models.timm_fpn_detector import TimmFPNDetector
from src.trainers.common import create_train_output_dir, select_device, set_seed
from src.trainers.detection_engine import run_detection_epoch, save_detection_checkpoint
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import MetricTracker
from src.utils.visualization import save_loss_curve


def get_v2_cfg(cfg):
    tiny = cfg["tiny_detector_v2"]
    return tiny


def build_dataloaders(cfg, train_cfg):
    train_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["train_images"], cfg["data"]["train_labels"], cfg["data"]["image_size"])
    val_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["val_images"], cfg["data"]["val_labels"], cfg["data"]["image_size"])
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    return train_loader, val_loader


def build_components(train_cfg, device):
    anchors = train_cfg["anchors"]
    model = TimmFPNDetector(
        num_classes=train_cfg["num_classes"],
        backbone_name=train_cfg["backbone_name"],
        pretrained=train_cfg["pretrained"],
        out_channels=train_cfg["out_channels"],
        num_anchors=len(anchors[0]),
    ).to(device)
    criterion = MultiScaleDetectionLoss(
        anchors=anchors,
        num_classes=train_cfg["num_classes"],
        box_gain=train_cfg.get("box_gain", 5.0),
        obj_gain=train_cfg.get("obj_gain", 1.0),
        cls_gain=train_cfg.get("cls_gain", 1.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        ignore_iou_threshold=train_cfg.get("ignore_iou_threshold", 0.5),
    ).to(device)

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": train_cfg.get("backbone_learning_rate", train_cfg["learning_rate"] * 0.1)},
            {"params": other_params, "lr": train_cfg["learning_rate"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )
    return model, criterion, optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train timm FPN tiny animal detector")
    parser.add_argument("--config", default="configs/coco_animals_10cls.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = get_v2_cfg(cfg)
    set_seed(cfg["project"]["seed"])

    run_dir = create_train_output_dir(cfg["project"]["output_dir"], "tiny_detector_v2")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_tiny_detector_v2", cfg["project"]["log_dir"])
    logger.info("本次训练输出目录: %s", run_dir)
    tracker = MetricTracker(cfg["project"]["log_dir"], "train_tiny_detector_v2")
    device = select_device(train_cfg["device"])
    logger.info("使用设备: %s", device)

    train_loader, val_loader = build_dataloaders(cfg, train_cfg)
    model, criterion, optimizer = build_components(train_cfg, device)

    start_epoch = 1
    best_val = float("inf")
    if train_cfg.get("resume"):
        ckpt = torch.load(train_cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        logger.info("从 checkpoint 恢复: %s", train_cfg["resume"])

    history: Dict[str, List[float]] = {"train_total": [], "val_total": [], "train_box": [], "train_obj": [], "train_cls": []}
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    patience = train_cfg.get("early_stop_patience", 0)
    bad_epochs = 0

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        if freeze_epochs > 0:
            trainable = epoch > freeze_epochs
            model.set_backbone_trainable(trainable)
            logger.info("Epoch %03d | backbone trainable: %s", epoch, trainable)

        train_loss = run_detection_epoch(model, train_loader, criterion, optimizer, device, train=True, tracker=tracker, epoch=epoch, phase="train")
        val_loss = run_detection_epoch(model, val_loader, criterion, optimizer, device, train=False, tracker=tracker, epoch=epoch, phase="val")

        logger.info(
            "Epoch %03d | train total %.4f box %.4f obj %.4f cls %.4f | val total %.4f box %.4f obj %.4f cls %.4f",
            epoch,
            train_loss["total"],
            train_loss["box"],
            train_loss["obj"],
            train_loss["cls"],
            val_loss["total"],
            val_loss["box"],
            val_loss["obj"],
            val_loss["cls"],
        )
        tracker.log({"epoch": epoch, "phase": "epoch", "train_total": train_loss["total"], "train_box": train_loss["box"], "train_obj": train_loss["obj"], "train_cls": train_loss["cls"], "val_total": val_loss["total"], "val_box": val_loss["box"], "val_obj": val_loss["obj"], "val_cls": val_loss["cls"]})

        history["train_total"].append(train_loss["total"])
        history["val_total"].append(val_loss["total"])
        history["train_box"].append(train_loss["box"])
        history["train_obj"].append(train_loss["obj"])
        history["train_cls"].append(train_loss["cls"])
        save_loss_curve(history, str(run_dir / "tiny_detector_v2_loss_curve.png"))

        is_best = val_loss["total"] < best_val
        if is_best:
            best_val = val_loss["total"]
            bad_epochs = 0
            logger.info("保存新的 best checkpoint，val loss: %.4f", best_val)
        else:
            bad_epochs += 1

        save_detection_checkpoint(ckpt_dir, epoch, model, optimizer, best_val, cfg, is_best, train_cfg["checkpoint_interval"])

        if patience > 0 and bad_epochs >= patience:
            logger.info("Early stopping triggered at epoch %03d, best val loss %.4f", epoch, best_val)
            break


if __name__ == "__main__":
    main()
```

---

## 8. 完整代码四：多尺度推理解码

如果训练使用 `TimmFPNDetector`，推理也要支持 list outputs。

建议新增文件：

```text
src/inferencers/tiny_detector_v2_core.py
```

完整代码如下：

```python
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.data.transforms import image_to_tensor, letterbox
from src.models.timm_fpn_detector import TimmFPNDetector, decode_multiscale_predictions
from src.utils.box_ops import nms, xywh_to_xyxy


def load_tiny_detector_v2(cfg, device):
    train_cfg = cfg["tiny_detector_v2"]
    checkpoint = torch.load(cfg["infer"]["checkpoint"], map_location=device)
    model = TimmFPNDetector(
        num_classes=train_cfg["num_classes"],
        backbone_name=train_cfg["backbone_name"],
        pretrained=False,
        out_channels=train_cfg["out_channels"],
        num_anchors=len(train_cfg["anchors"][0]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    anchors = torch.tensor(train_cfg["anchors"], dtype=torch.float32, device=device)
    return model, anchors


def postprocess_tiny_v2(raw_outputs, anchors, conf_threshold: float, iou_threshold: float, image_size: int, original_shape: Tuple[int, int], scale: float, pad: Tuple[int, int]):
    boxes, obj, cls_probs = decode_multiscale_predictions(raw_outputs, anchors)
    scores_per_cls = obj.unsqueeze(-1) * cls_probs
    scores, labels = scores_per_cls.max(dim=-1)
    mask = scores[0] > conf_threshold
    if not mask.any():
        return []

    boxes = boxes[0][mask]
    scores = scores[0][mask]
    labels = labels[0][mask]
    xyxy = xywh_to_xyxy(boxes) * image_size

    keep_all: List[int] = []
    for cls_id in labels.unique():
        cls_idx = torch.where(labels == cls_id)[0]
        keep = nms(xyxy[cls_idx], scores[cls_idx], iou_threshold)
        keep_all.extend(cls_idx[keep].tolist())

    pad_x, pad_y = pad
    original_h, original_w = original_shape
    detections = []
    for idx in keep_all:
        box = xyxy[idx].detach().cpu().numpy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, original_w - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, original_h - 1)
        detections.append((box.astype(int), float(scores[idx].item()), int(labels[idx].item())))
    return detections


def draw_detections(frame, detections, class_names, color=(46, 204, 113)):
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def build_tiny_v2_frame_inferencer(model, anchors, cfg, device, tracker=None, source=""):
    image_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]
    frame_counter = {"idx": 0}

    def infer_frame(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp, scale, pad = letterbox(rgb, image_size)
        tensor = image_to_tensor(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_outputs = model(tensor)
        detections = postprocess_tiny_v2(raw_outputs, anchors, cfg["infer"]["conf_threshold"], cfg["infer"]["iou_threshold"], image_size, frame.shape[:2], scale, pad)
        if tracker is not None:
            tracker.log_detections(source, frame_counter["idx"], detections, class_names)
        frame_counter["idx"] += 1
        return draw_detections(frame, detections, class_names)

    return infer_frame
```

---

## 9. 完整代码五：推理入口

建议新增文件：

```text
src/inferencers/tiny_detector_v2.py
```

完整代码如下：

```python
import argparse

import torch

from src.inferencers.detection_pipeline import run_image_or_video
from src.inferencers.tiny_detector_v2_core import build_tiny_v2_frame_inferencer, load_tiny_detector_v2
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import InferenceTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer timm FPN tiny animal detector on image or video")
    parser.add_argument("--config", default="configs/coco_animals_10cls.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("infer_tiny_detector_v2", cfg["project"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    tracker = InferenceTracker(cfg["project"]["log_dir"], "infer_tiny_detector_v2")

    model, anchors = load_tiny_detector_v2(cfg, device)
    infer_frame = build_tiny_v2_frame_inferencer(model, anchors, cfg, device, tracker=tracker, source=source)
    run_image_or_video(source, output, infer_frame, logger)
    logger.info("推理检测汇总: %s", tracker.summary())


if __name__ == "__main__":
    main()
```

---

## 10. 完整代码六：脚本入口

可以在 `script/run_train.sh` 中增加：

```bash
  tiny_detector_v2)
    python -m src.trainers.tiny_detector_v2 --config "${CONFIG}"
    ;;
```

可以在 `script/run_train_detection.sh` 中允许：

```bash
  tiny_detector|tiny_detector_v2|grounding_dino|yolov5|mmdetection|detectron2)
    bash script/run_train.sh "${MODEL}" "${CONFIG}"
    ;;
```

运行命令：

```bash
bash script/run_train_detection.sh tiny_detector_v2 configs/coco_animals_10cls.yaml
```

推理命令：

```bash
python -m src.inferencers.tiny_detector_v2 \
  --config configs/coco_animals_10cls.yaml \
  --source samples/demo.mp4 \
  --output outputs/inference/tiny_detector_v2_demo_result.mp4
```

---

## 11. 训练策略建议

### 11.1 第一阶段：冻结 backbone

前 3 个 epoch：

```text
backbone frozen
train FPN + head only
```

目的：

- 保护预训练特征；
- 让随机初始化的 FPN/head 先适配检测任务；
- 避免一开始大梯度破坏 backbone。

### 11.2 第二阶段：小学习率解冻 backbone

第 4 个 epoch 后：

```text
backbone lr = 3e-5
head lr = 3e-4
```

目的：

- 轻微微调 backbone；
- 保留预训练泛化能力；
- 让特征适配动物检测。

### 11.3 Early stopping

建议：

```yaml
early_stop_patience: 8
```

如果验证 loss 连续 8 轮不下降，则停止训练。

---

## 12. 风险和注意事项

### 12.1 timm 不同 backbone 的 out_indices 可能不同

`mobilenetv3_small_100` 通常可以使用 `(2, 3, 4)`，但不同模型 feature stride 可能不同。建议启动时打印：

```python
model.backbone.feature_info
```

确认三个输出层大致对应 stride 8、16、32。

### 12.2 anchors 必须和多尺度对应

anchors 形状必须是：

```text
[num_scales, anchors_per_scale, 2]
```

例如 3 个尺度，每个尺度 3 个 anchor：

```text
[3, 3, 2]
```

### 12.3 训练 loss 和 mAP 都要看

CIoU/Focal 后，loss 数值尺度会和旧版不同，不能直接和旧 loss 数值比较。

应重点比较：

- val loss 是否不再持续反升；
- mAP@0.5 是否提高；
- recall 是否提高；
- 小目标 AP 是否改善。

---

## 13. 实施优先级

推荐按以下顺序实现：

1. 新增 `src/models/timm_fpn_detector.py`；
2. 新增 `src/models/multiscale_loss.py`；
3. 新增 `src/trainers/tiny_detector_v2.py`；
4. 在配置中加入 `tiny_detector_v2`；
5. 在脚本中加入 `tiny_detector_v2`；
6. 先跑 1 个 epoch 验证 shape、loss、checkpoint；
7. 跑 20 epoch 看 loss 曲线；
8. 加 mAP 验证；
9. 再做 anchor k-means 和数据增强。

---

## 14. 总结

这套方案的核心目的：

```text
用预训练 backbone 解决泛化弱；
用 FPN 多尺度解决小中大目标尺度问题；
用 CIoU 让 box loss 更贴近检测指标；
用 Focal Loss 缓解 objectness 正负样本不平衡；
用 ignore mask 避免合理预测被错误当负样本惩罚。
```

相比当前 Tiny Detector，V2 版本更适合 `coco_animals_10cls` 这种真实复杂数据集。

---

## 15. 已落地代码内容汇总

本节记录本次已经写入项目的完整代码和接入点，方便后续查看、运行和维护。

---

## 15.1 新增 timm + FPN + 多尺度检测模型

新增文件：

```text
src/models/timm_fpn_detector.py
```

该文件实现了新的多尺度检测模型，核心组件包括：

- `TimmFPNDetector`；
- `FPN`；
- `DetectionHead`；
- `decode_multiscale_predictions`。

整体结构：

```text
input image
    |
    v
timm pretrained backbone, features_only=True
    |
    +--> C3
    +--> C4
    +--> C5
    |
    v
FPN top-down feature fusion
    |
    +--> P3, stride 8,  small objects
    +--> P4, stride 16, medium objects
    +--> P5, stride 32, large objects
    |
    v
three YOLO-style detection heads
```

输出格式为三尺度 list：

```text
[
  [B, A, H/8,  W/8,  5+C],
  [B, A, H/16, W/16, 5+C],
  [B, A, H/32, W/32, 5+C],
]
```

设计目的：

1. 用 `timm` 预训练 backbone 替代从零训练的 `TinyBackbone`；
2. 用 FPN 融合浅层空间信息和深层语义信息；
3. 用多尺度检测头同时覆盖小、中、大目标；
4. 让模型更适合 `coco_animals_10cls` 中复杂尺度的动物目标。

---

## 15.2 新增 CIoU + Focal + Ignore Mask 多尺度 Loss

新增文件：

```text
src/models/multiscale_loss.py
```

该文件实现了：

- `MultiScaleDetectionLoss`；
- `bbox_ciou_xyxy`；
- `focal_bce_with_logits`。

Loss 组成：

```text
total = box_gain * box_loss + obj_gain * obj_loss + cls_gain * cls_loss
```

其中：

```text
box_loss = 1 - CIoU
obj_loss = Focal BCE with logits + ignore mask
cls_loss = BCE with logits
```

设计目的：

1. 用 CIoU 替代 SmoothL1，使 box loss 更接近检测指标；
2. 用 Focal Loss 缓解 objectness 正负样本极不平衡；
3. 用 ignore mask 避免合理预测被错误当作普通负样本惩罚；
4. 让多尺度预测的每个尺度都能独立构造 target 和计算 loss。

---

## 15.3 新增 tiny_detector_v2 训练入口

新增文件：

```text
src/trainers/tiny_detector_v2.py
```

该训练入口支持：

- 加载 `tiny_detector_v2` 配置；
- 构建 `AnimalDetectionDataset` 和 `DataLoader`；
- 构建 `TimmFPNDetector`；
- 构建 `MultiScaleDetectionLoss`；
- 使用 AdamW 优化器；
- backbone 和 head 使用不同学习率；
- 前若干 epoch 冻结 backbone；
- 支持 resume；
- 支持 checkpoint 保存；
- 支持 best checkpoint；
- 支持 loss curve 保存；
- 支持 early stopping。

训练命令：

```bash
bash script/run_train_detection.sh tiny_detector_v2 configs/coco_animals_10cls.yaml
```

或者：

```bash
python -m src.trainers.tiny_detector_v2 --config configs/coco_animals_10cls.yaml
```

设计目的：

1. 不破坏原来的 `tiny_detector` 训练入口；
2. 让 V2 作为独立模型可单独训练、对比和回退；
3. 支持更稳定的预训练 backbone 微调策略。

---

## 15.4 新增 tiny_detector_v2 推理入口

新增文件：

```text
src/inferencers/tiny_detector_v2_core.py
src/inferencers/tiny_detector_v2.py
```

推理流程：

```text
load checkpoint
    |
    v
build TimmFPNDetector
    |
    v
load state_dict
    |
    v
read image/video frame
    |
    v
letterbox + image_to_tensor
    |
    v
multi-scale model forward
    |
    v
decode_multiscale_predictions
    |
    v
confidence threshold filtering
    |
    v
per-class NMS
    |
    v
restore box coordinates to original image
    |
    v
draw detections and save output
```

推理命令：

```bash
bash script/run_infer_detection.sh tiny_detector_v2 configs/coco_animals_10cls.yaml samples/demo.mp4 outputs/inference/tiny_detector_v2_demo_result.mp4
```

或者：

```bash
python -m src.inferencers.tiny_detector_v2 \
  --config configs/coco_animals_10cls.yaml \
  --source samples/demo.mp4 \
  --output outputs/inference/tiny_detector_v2_demo_result.mp4
```

设计目的：

1. 支持多尺度输出的统一解码；
2. 支持图片/视频复用原有 `run_image_or_video` 流程；
3. 让 V2 模型训练完成后可以直接端到端推理验证。

---

## 15.5 配置已加入 tiny_detector_v2

修改文件：

```text
configs/coco_animals_10cls.yaml
```

新增配置分组：

```yaml
tiny_detector_v2:
  backbone_name: mobilenetv3_small_100
  pretrained: true
  out_indices: [2, 3, 4]
  out_channels: 128
  freeze_backbone_epochs: 3
  early_stop_patience: 8
  num_classes: 10
  anchors:
    - [[0.025, 0.035], [0.045, 0.060], [0.070, 0.090]]
    - [[0.100, 0.130], [0.160, 0.200], [0.230, 0.280]]
    - [[0.320, 0.380], [0.450, 0.550], [0.650, 0.750]]
  epochs: 50
  batch_size: 8
  learning_rate: 0.0003
  backbone_learning_rate: 0.00003
  weight_decay: 0.001
  checkpoint_interval: 1
  resume: ""
  device: auto
  box_gain: 5.0
  obj_gain: 1.0
  cls_gain: 1.0
  focal_alpha: 0.25
  focal_gamma: 2.0
  ignore_iou_threshold: 0.5
```

配置含义：

- `backbone_name`：使用的 timm backbone；
- `pretrained`：是否加载预训练权重；
- `out_indices`：从 timm backbone 取哪些阶段特征；
- `out_channels`：FPN 输出通道数；
- `freeze_backbone_epochs`：前几轮冻结 backbone；
- `early_stop_patience`：验证 loss 连续不下降多少轮后停止；
- `anchors`：三尺度 anchors；
- `learning_rate`：FPN/head 学习率；
- `backbone_learning_rate`：backbone 微调学习率；
- `box_gain`、`obj_gain`、`cls_gain`：loss 权重；
- `focal_alpha`、`focal_gamma`：Focal Loss 参数；
- `ignore_iou_threshold`：ignore mask 阈值。

---

## 15.6 脚本已接入 tiny_detector_v2

修改文件：

```text
script/run_train.sh
script/run_train_detection.sh
script/run_infer.sh
script/run_infer_detection.sh
```

现在训练脚本支持：

```bash
bash script/run_train_detection.sh tiny_detector_v2 configs/coco_animals_10cls.yaml
```

推理脚本支持：

```bash
bash script/run_infer_detection.sh tiny_detector_v2 configs/coco_animals_10cls.yaml samples/demo.mp4 outputs/inference/tiny_detector_v2_demo_result.mp4
```

脚本接入目的：

1. 保留原 `tiny_detector`；
2. 新增 `tiny_detector_v2` 可独立运行；
3. 统一训练和推理入口；
4. 方便与其他检测模型横向比较。

---

## 15.7 检查结果

已检查以下文件的 linter 诊断：

```text
src/models/timm_fpn_detector.py
src/models/multiscale_loss.py
src/trainers/tiny_detector_v2.py
src/inferencers/tiny_detector_v2_core.py
src/inferencers/tiny_detector_v2.py
configs/coco_animals_10cls.yaml
script/run_train.sh
script/run_train_detection.sh
script/run_infer.sh
script/run_infer_detection.sh
```

检查结果：

```text
No linter errors found.
```

同时尝试做运行时 shape 检查，但当前系统默认 `python3` 环境没有安装 `torch`，报错为：

```text
ModuleNotFoundError: No module named 'torch'
```

因此运行测试需要在实际训练使用的 conda/venv 环境里执行。

---

## 15.8 下一步建议

建议先做最小运行验证，不要直接长时间训练。

### Step 1：先跑 1 个 epoch

临时把配置改为：

```yaml
tiny_detector_v2:
  epochs: 1
```

然后执行：

```bash
bash script/run_train_detection.sh tiny_detector_v2 configs/coco_animals_10cls.yaml
```

检查：

- timm backbone 是否能正常加载；
- 三尺度输出 shape 是否正确；
- loss 是否能正常反向传播；
- checkpoint 是否能保存；
- loss curve 是否生成。

### Step 2：恢复训练轮数，先跑 20 epoch

如果 1 epoch 正常，再改为：

```yaml
tiny_detector_v2:
  epochs: 20
```

观察：

- `train_total` 是否下降；
- `val_total` 是否不再像旧版一样明显反升；
- `box_loss` 是否稳定下降；
- `obj_loss` 是否没有被负样本主导；
- `cls_loss` 是否正常收敛。

### Step 3：再跑完整训练

如果 20 epoch 曲线合理，再恢复：

```yaml
tiny_detector_v2:
  epochs: 50
```

并保留：

```yaml
early_stop_patience: 8
checkpoint_interval: 1
```

### Step 4：补充 mAP 评估

Detection loss 不是最终检测指标，后续建议补充：

```text
mAP@0.5
mAP@0.5:0.95
Precision
Recall
Per-class AP
Small/Medium/Large object AP
```

### Step 5：继续优化 anchor 和数据增强

如果 V2 曲线明显改善，再进一步做：

1. anchor k-means；
2. train-only random horizontal flip；
3. HSV / brightness / contrast augmentation；
4. random scale；
5. Mosaic / MixUp。

现在训练命令改为：
bash script/run_train_detection.sh tiny_detector_pro configs/coco_animals_10cls.yaml

推理命令改为：
bash script/run_infer_detection.sh tiny_detector_pro configs/coco_animals_10cls.yaml samples/demo.mp4 outputs/inference/tiny_detector_pro_demo_result.mp4
