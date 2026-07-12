import argparse

import torch

from src.inferencers.detection_pipeline import run_image_or_video
from src.inferencers.grounding_dino_core import build_grounding_frame_inferencer, load_grounding_model
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import InferenceTracker
from src.utils.tracker import InferenceTracker


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="GroundingDINO-like open-vocabulary animal detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--text", default=None, help="逗号分隔动物名称，例如 cat,dog,fox")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_grounding_dino", cfg["project"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.text:
        class_names = [x.strip() for x in args.text.split(",")]
    else:
        checkpoint = torch.load(cfg["infer"].get("grounding_checkpoint", "outputs/checkpoints/grounding_dino/best.pt"), map_location=device)
        class_names = checkpoint.get("class_names", cfg["data"]["class_names"])
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    tracker = InferenceTracker(cfg["project"]["log_dir"], "infer_grounding_dino")
    model, text_tokens = load_grounding_model(cfg, device, class_names)
    infer_frame = build_grounding_frame_inferencer(model, text_tokens, cfg, class_names, device, tracker=tracker, source=source)
    run_image_or_video(source, output, infer_frame, logger)
    logger.info("推理检测汇总: %s", tracker.summary())


if __name__ == "__main__":
    main()
