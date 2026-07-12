import argparse
from pathlib import Path

from src.specialized.superanimal_adapter import run_superanimal_inference
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
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
