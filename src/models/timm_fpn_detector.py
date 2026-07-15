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

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_convs[0](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
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
