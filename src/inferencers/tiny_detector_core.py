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
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        raw: 检测头未后处理的原始输出张量。
        anchors: YOLO 风格 anchor 尺寸列表或张量，用于解码宽高和匹配目标。
        conf_threshold: 置信度阈值，低于该分数的预测框会被过滤。
        iou_threshold: NMS 的 IoU 阈值，用于去除高度重叠的重复框。
        image_size: 模型输入图像尺寸，图片会缩放或 letterbox 到该大小。
        original_shape: 原始图像高宽，用于把 letterbox 后坐标映射回原图。
        scale: letterbox 缩放比例，用于还原检测框到原始图像坐标。
        pad: letterbox 左上角填充值 (pad_x, pad_y)，用于坐标反变换。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        frame: OpenCV 读取的一帧 BGR 图像，用于单帧检测、绘框或视频写出。
        detections: 检测结果列表，每项通常为 box、score、class_id。
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
        color: color 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        model: 待训练或待推理的 PyTorch 模型实例。
        anchors: YOLO 风格 anchor 尺寸列表或张量，用于解码宽高和匹配目标。
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
        tracker: 指标或推理跟踪器，用于写入 jsonl/csv 结构化日志。
        source: 输入图片、视频、目录或数据源路径，由推理入口传入。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    image_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]
    frame_counter = {"idx": 0}

    def infer_frame(frame):
        """执行单次推理或分类逻辑，输出结构化预测结果。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            frame: OpenCV 读取的一帧 BGR 图像，用于单帧检测、绘框或视频写出。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
