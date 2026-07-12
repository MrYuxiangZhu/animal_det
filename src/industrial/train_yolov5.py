import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.industrial import python_executable, run_command, write_yolo_data_yaml
from src.utils.logger import setup_logger


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    parser = argparse.ArgumentParser(description="Train animal detector with Ultralytics YOLOv5 repo")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("train_yolov5", cfg["project"]["log_dir"])
    yolo_cfg = cfg["yolov5"]
    repo = Path(yolo_cfg["repo"])
    if not (repo / "train.py").exists():
        raise RuntimeError(f"未找到 YOLOv5 仓库: {repo}。请先执行: git clone https://github.com/ultralytics/yolov5 {repo}")

    data_yaml = yolo_cfg["data_yaml"]
    write_yolo_data_yaml(data_yaml, cfg["data"]["root"], cfg["data"]["train_images"], cfg["data"]["val_images"], cfg["data"]["class_names"])
    cmd = [
        python_executable(),
        "train.py",
        "--img",
        str(cfg["data"]["image_size"]),
        "--batch",
        str(yolo_cfg["batch_size"]),
        "--epochs",
        str(yolo_cfg["epochs"]),
        "--data",
        str(Path.cwd() / data_yaml),
        "--weights",
        yolo_cfg["weights"],
        "--project",
        str(Path.cwd() / cfg["project"]["output_dir"] / "yolov5"),
        "--name",
        yolo_cfg["run_name"],
    ]
    logger.info("启动 YOLOv5 训练: %s", " ".join(cmd))
    run_command(cmd, cwd=str(repo))


if __name__ == "__main__":
    main()
