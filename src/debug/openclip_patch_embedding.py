import argparse

import torch

from src.specialized.openclip_adapter import load_openclip_model
from src.trainers.common import select_device
from src.utils.config import load_config


def main() -> None:
    """打印 OpenCLIP ViT patch embedding 的关键源码层和张量形状。"""
    parser = argparse.ArgumentParser(description="Debug OpenCLIP patch embedding shapes")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = select_device(cfg["train"]["device"])
    openclip_cfg = cfg["openclip"]
    model, _, _ = load_openclip_model(openclip_cfg["model_name"], openclip_cfg.get("pretrained", ""), device, openclip_cfg.get("checkpoint_path", ""))
    visual = model.visual
    print("visual class:", visual.__class__)
    print("patch embedding conv1:", visual.conv1)
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        y = visual.conv1(x)
        print("input:", tuple(x.shape))
        print("after conv1:", tuple(y.shape))
        y_flat = y.reshape(y.shape[0], y.shape[1], -1)
        print("after flatten:", tuple(y_flat.shape))
        y_tokens = y_flat.permute(0, 2, 1)
        print("after permute patch tokens:", tuple(y_tokens.shape))
        if hasattr(visual, "class_embedding"):
            cls = visual.class_embedding.to(y_tokens.dtype).expand(y_tokens.shape[0], 1, -1)
            print("class token:", tuple(cls.shape))
            print("after prepend class token:", tuple(torch.cat([cls, y_tokens], dim=1).shape))
        if hasattr(visual, "positional_embedding"):
            print("positional embedding:", tuple(visual.positional_embedding.shape))


if __name__ == "__main__":
    main()
