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
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        box_raw: GroundingDINO-like 模型输出的原始框回归张量。
        objectness: 检测模型输出的目标存在性 logits。
        class_logits: 文本条件检测模型输出的类别 logits，形状通常为 [B, C, H, W]。
        conf_threshold: 置信度阈值，低于该分数的预测框会被过滤。
        iou_threshold: NMS 的 IoU 阈值，用于去除高度重叠的重复框。
        image_size: 模型输入图像尺寸，图片会缩放或 letterbox 到该大小。
        original_shape: 原始图像高宽，用于把 letterbox 后坐标映射回原图。
        scale: letterbox 缩放比例，用于还原检测框到原始图像坐标。
        pad: letterbox 左上角填充值 (pad_x, pad_y)，用于坐标反变换。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        model: 待训练或待推理的 PyTorch 模型实例。
        text_tokens: 文本 prompt 经 tokenizer 编码后的 token 张量。
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
        tracker: 指标或推理跟踪器，用于写入 jsonl/csv 结构化日志。
        source: 输入图片、视频、目录或数据源路径，由推理入口传入。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    image_size = cfg["data"]["image_size"]
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
            box_raw, objectness, class_logits = model(tensor, text_tokens)
        detections = postprocess_grounding(box_raw, objectness, class_logits, cfg["infer"]["conf_threshold"], cfg["infer"]["iou_threshold"], image_size, frame.shape[:2], scale, pad)
        if tracker is not None:
            tracker.log_detections(source, frame_counter["idx"], detections, class_names)
        frame_counter["idx"] += 1
        return draw_detections(frame, detections, class_names, color=(52, 152, 219))

    return infer_frame
