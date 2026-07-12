import argparse

import torch

from src.inferencers.detection_pipeline import run_image_or_video
from src.inferencers.tiny_detector_core import build_tiny_frame_inferencer, load_tiny_detector
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import InferenceTracker


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    parser = argparse.ArgumentParser(description="Infer tiny animal detector on image or video")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_tiny_detector", cfg["project"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    tracker = InferenceTracker(cfg["project"]["log_dir"], "infer_tiny_detector")
    model, anchors = load_tiny_detector(cfg, device)
    infer_frame = build_tiny_frame_inferencer(model, anchors, cfg, device, tracker=tracker, source=source)
    run_image_or_video(source, output, infer_frame, logger)
    logger.info("推理检测汇总: %s", tracker.summary())


if __name__ == "__main__":
    main()
