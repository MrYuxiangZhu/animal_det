import argparse

import cv2
import torch

from src.data.transforms import image_to_tensor, resize_square
from src.models.clip_like import MiniCLIP, SimpleTokenizer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    parser = argparse.ArgumentParser(description="MiniCLIP animal zero-shot image classification")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--text", default=None, help="逗号分隔候选动物类别，例如 cat,dog,fox")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_clip", cfg["project"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_cfg = cfg["clip"]
    checkpoint_path = cfg["infer"].get("clip_checkpoint", "outputs/checkpoints/clip/best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = [x.strip() for x in args.text.split(",")] if args.text else checkpoint.get("class_names", cfg["data"]["class_names"])

    tokenizer = SimpleTokenizer(context_length=clip_cfg["context_length"])
    model = MiniCLIP(tokenizer.vocab_size, clip_cfg["context_length"], clip_cfg["embed_dim"], clip_cfg["width_mult"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source = args.source or cfg["infer"]["source"]
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"无法读取图片: {source}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_square(image, clip_cfg["image_size"])
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    prompts = [f"a photo of a {name}" for name in class_names]
    tokens = tokenizer.encode(prompts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(tensor)
        text_features = model.encode_text(tokens)
        probs = (image_features @ text_features.t()).softmax(dim=-1)[0]
    order = probs.argsort(descending=True)
    lines = []
    for idx in order[: min(5, len(class_names))]:
        line = f"{class_names[int(idx)]}: {float(probs[idx]):.4f}"
        lines.append(line)
        logger.info(line)
        print(line)
    if args.output:
        from pathlib import Path

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
