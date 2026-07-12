from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.data.transforms import image_to_tensor, letterbox
from src.inferencers.tiny_detector_core import draw_detections
from src.models.clip_like import SimpleTokenizer
from src.models.grounding_dino_like import GroundingDINOAnimal, decode_grounding_boxes
from src.utils.box_ops import nms, xywh_to_xyxy


def load_grounding_model(cfg, device, class_names):
    """加载模型、权重或外部依赖，并返回后续流程需要使用的对象。
    
    Args:
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    grounding_cfg = cfg["grounding_dino"]
    tokenizer = SimpleTokenizer(context_length=grounding_cfg["context_length"])
    prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = tokenizer.encode(prompts).to(device)
    model = GroundingDINOAnimal(tokenizer.vocab_size, grounding_cfg["context_length"], len(class_names), grounding_cfg["hidden_dim"], grounding_cfg["width_mult"]).to(device)
    checkpoint = torch.load(cfg["infer"].get("grounding_checkpoint", "outputs/checkpoints/grounding_dino/best.pt"), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, text_tokens


def postprocess_grounding(box_raw, objectness, class_logits, conf_threshold: float, iou_threshold: float, image_size: int, original_shape: Tuple[int, int], scale: float, pad: Tuple[int, int]):
    """对模型原始输出做后处理，生成可解释的检测框、类别和置信度。
    
    Args:
        box_raw: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        objectness: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_logits: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        conf_threshold: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        iou_threshold: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        image_size: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        original_shape: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        scale: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        pad: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    boxes = decode_grounding_boxes(box_raw)[0]
    obj = objectness.sigmoid()[0]
    cls_probs = class_logits.sigmoid()[0]
    scores_per_cls = obj.unsqueeze(0) * cls_probs
    scores, labels = scores_per_cls.flatten(1).max(dim=0)
    mask = scores > conf_threshold
    if not mask.any():
        return []
    flat_boxes = boxes.reshape(-1, 4)[mask]
    scores = scores[mask]
    labels = labels[mask]
    xyxy = xywh_to_xyxy(flat_boxes) * image_size
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


def build_grounding_frame_inferencer(model, text_tokens, cfg, class_names, device, tracker=None, source=""):
    """根据配置构建可复用组件，降低入口函数中的业务耦合。
    
    Args:
        model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        text_tokens: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        tracker: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    image_size = cfg["data"]["image_size"]
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
            box_raw, objectness, class_logits = model(tensor, text_tokens)
        detections = postprocess_grounding(box_raw, objectness, class_logits, cfg["infer"]["conf_threshold"], cfg["infer"]["iou_threshold"], image_size, frame.shape[:2], scale, pad)
        if tracker is not None:
            tracker.log_detections(source, frame_counter["idx"], detections, class_names)
        frame_counter["idx"] += 1
        return draw_detections(frame, detections, class_names, color=(52, 152, 219))

    return infer_frame
