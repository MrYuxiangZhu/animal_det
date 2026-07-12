import argparse

import torch

from src.specialized.openclip_adapter import classify_image, write_topk
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCLIP zero-shot animal recognition")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--text", default=None, help="逗号分隔候选动物类别，例如 red panda,snow leopard")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_openclip", cfg["project"]["log_dir"])
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    class_names = [x.strip() for x in args.text.split(",")] if args.text else cfg["data"]["class_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = classify_image(source, class_names, cfg, device)
    write_topk(result, output, cfg["openclip"]["topk"])
    for name, score in result[: cfg["openclip"]["topk"]]:
        logger.info("%s: %.4f", name, score)


if __name__ == "__main__":
    main()
