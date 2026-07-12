import argparse

from src.specialized.wildlife_adapter import run_camera_traps_pipeline
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Microsoft CameraTraps/Pytorch-Wildlife pipeline inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_pytorch_wildlife", cfg["project"]["log_dir"])
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    run_camera_traps_pipeline(source, output, cfg)
    logger.info("Pytorch-Wildlife 推理完成: %s", output)


if __name__ == "__main__":
    main()
