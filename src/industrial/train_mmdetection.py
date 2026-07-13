import argparse
from pathlib import Path

from src.trainers.common import create_train_output_dir
from src.utils.config import load_config
from src.utils.industrial import python_executable, require_package, run_command
from src.utils.logger import setup_logger


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="Train animal detector with MMDetection")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("train_mmdetection", cfg["project"]["log_dir"])
    mm_cfg = cfg["mmdetection"]
    require_package("mmdet", "pip install -U openmim && mim install mmengine mmcv mmdet")
    work_dir = create_train_output_dir(cfg["project"]["output_dir"], "mmdetection")
    logger.info("本次训练输出目录: %s", work_dir)
    cmd = [python_executable(), "-m", "mmdet.tools.train", mm_cfg["config"], "--work-dir", str(work_dir)]
    logger.info("启动 MMDetection 训练: %s", " ".join(cmd))
    run_command(cmd)


if __name__ == "__main__":
    main()
