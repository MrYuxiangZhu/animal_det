import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from src.data.transforms import image_to_tensor, resize_square
from src.industrial.train_timm import TimmClassifier
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer animal classifier with timm")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_timm", cfg["project"]["log_dir"])
    timm_cfg = cfg["timm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg["infer"].get("timm_checkpoint", "outputs/checkpoints/timm/best.pt"), map_location=device)
    class_names = ckpt.get("class_names", cfg["data"]["class_names"])
    model = TimmClassifier(timm_cfg["model_name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    source = args.source or cfg["infer"]["source"]
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"timm 分类推理只支持图片输入，无法读取: {source}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image_to_tensor(resize_square(rgb, timm_cfg["image_size"])).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = F.softmax(model(tensor), dim=-1)[0]
    topk = torch.topk(prob, k=min(5, len(class_names)))
    output = args.output or cfg["infer"]["output"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{class_names[int(idx)]}: {float(score):.4f}" for score, idx in zip(topk.values, topk.indices)]
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    logger.info("timm 分类推理完成: %s", output)
    for line in lines:
        logger.info(line)


if __name__ == "__main__":
    main()
