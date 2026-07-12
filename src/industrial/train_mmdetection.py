import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.industrial import python_executable, require_package, run_command
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train animal detector with MMDetection")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("train_mmdetection", cfg["project"]["log_dir"])
    mm_cfg = cfg["mmdetection"]
    require_package("mmdet", "pip install -U openmim && mim install mmengine mmcv mmdet")
    work_dir = Path(cfg["project"]["output_dir"]) / "mmdetection" / mm_cfg["run_name"]
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python_executable(), "-m", "mmdet.tools.train", mm_cfg["config"], "--work-dir", str(work_dir)]
    logger.info("启动 MMDetection 训练: %s", " ".join(cmd))
    run_command(cmd)


if __name__ == "__main__":
    main()
