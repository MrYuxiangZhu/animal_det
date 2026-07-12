import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.industrial import python_executable, run_command
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with Ultralytics YOLOv5 repo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_yolov5", cfg["project"]["log_dir"])
    yolo_cfg = cfg["yolov5"]
    repo = Path(yolo_cfg["repo"])
    if not (repo / "detect.py").exists():
        raise RuntimeError(f"未找到 YOLOv5 仓库: {repo}")
    source = args.source or cfg["infer"]["source"]
    output_dir = Path(args.output or cfg["infer"]["output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = cfg["infer"].get("yolov5_checkpoint", yolo_cfg["infer_weights"])
    cmd = [
        python_executable(),
        "detect.py",
        "--weights",
        weights,
        "--source",
        source,
        "--img",
        str(cfg["data"]["image_size"]),
        "--conf-thres",
        str(cfg["infer"]["conf_threshold"]),
        "--iou-thres",
        str(cfg["infer"]["iou_threshold"]),
        "--project",
        str(output_dir),
        "--name",
        "yolov5_result",
        "--exist-ok",
    ]
    logger.info("启动 YOLOv5 推理: %s", " ".join(cmd))
    run_command(cmd, cwd=str(repo))


if __name__ == "__main__":
    main()
