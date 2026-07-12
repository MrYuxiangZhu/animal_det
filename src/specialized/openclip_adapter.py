from pathlib import Path
from typing import List

import cv2
import torch

from src.utils.industrial import require_package


def load_openclip_model(model_name: str, pretrained: str, device):
    require_package("open_clip", "pip install open_clip_torch")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def classify_image(source: str, class_names: List[str], cfg, device):
    from PIL import Image

    openclip_cfg = cfg["openclip"]
    model, preprocess, tokenizer = load_openclip_model(openclip_cfg["model_name"], openclip_cfg["pretrained"], device)
    image = Image.open(source).convert("RGB")
    prompts = [openclip_cfg["prompt_template"].format(name=name) for name in class_names]
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    order = probs.argsort(descending=True)
    return [(class_names[int(idx)], float(probs[idx])) for idx in order]


def write_topk(result, output: str, topk: int) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}: {score:.4f}" for name, score in result[:topk]]
    Path(output).write_text("\n".join(lines), encoding="utf-8")
