from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.data.transforms import image_to_tensor, letterbox
from src.models.detector import AnimalDetector, decode_predictions
from src.utils.box_ops import nms, xywh_to_xyxy


def load_tiny_detector(cfg, device):
    """加载模型、权重或外部依赖，并返回后续流程需要使用的对象。
    
    Args:
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    checkpoint = torch.load(cfg["infer"]["checkpoint"], map_location=device)
    model = AnimalDetector(cfg["model"]["num_classes"], cfg["model"]["num_anchors"], cfg["model"]["width_mult"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    anchors = torch.tensor(cfg["model"]["anchors"], dtype=torch.float32, device=device)
    return model, anchors


def postprocess_tiny(raw, anchors, conf_threshold: float, iou_threshold: float, image_size: int, original_shape: Tuple[int, int], scale: float, pad: Tuple[int, int]):
    """对模型原始输出做后处理，生成可解释的检测框、类别和置信度。
    
    Args:
        raw: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        anchors: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        conf_threshold: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        iou_threshold: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        image_size: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        original_shape: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        scale: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        pad: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    boxes, obj, cls_probs = decode_predictions(raw, anchors)
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
    """把检测或识别结果绘制到图像帧上，用于可视化推理结果。
    
    Args:
        frame: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        detections: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        color: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def build_tiny_frame_inferencer(model, anchors, cfg, device, tracker=None, source=""):
    """根据配置构建可复用组件，降低入口函数中的业务耦合。
    
    Args:
        model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        anchors: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        tracker: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    image_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]
    frame_counter = {"idx": 0}

    def infer_frame(frame):
        """执行单次推理或分类逻辑，输出结构化预测结果。
        
        Args:
            frame: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp, scale, pad = letterbox(rgb, image_size)
        tensor = image_to_tensor(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(tensor)
        detections = postprocess_tiny(raw, anchors, cfg["infer"]["conf_threshold"], cfg["infer"]["iou_threshold"], image_size, frame.shape[:2], scale, pad)
        if tracker is not None:
            tracker.log_detections(source, frame_counter["idx"], detections, class_names)
        frame_counter["idx"] += 1
        return draw_detections(frame, detections, class_names)

    return infer_frame
