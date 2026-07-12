import argparse
from pathlib import Path

from src.specialized.superanimal_adapter import run_superanimal_inference
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="DeepLabCut SuperAnimal pose inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_superanimal", cfg["project"]["log_dir"])
    source = args.source or cfg["infer"]["source"]
    output_dir = str(Path(args.output or cfg["infer"]["output"]).parent)
    run_superanimal_inference(source, output_dir, cfg)
    logger.info("SuperAnimal 姿态推理完成，输出目录: %s", output_dir)


if __name__ == "__main__":
    main()
