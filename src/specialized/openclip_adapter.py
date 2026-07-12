from pathlib import Path
from typing import List

import cv2
import torch

from src.utils.industrial import require_package


def load_openclip_model(model_name: str, pretrained: str, device, checkpoint_path: str = ""):
    """加载 OpenCLIP 模型，支持在线预训练标签和离线本地权重。
    
    Args:
        model_name: OpenCLIP 模型结构名称，例如 ``ViT-B-32``。
        pretrained: 在线权重标签；当 ``checkpoint_path`` 不为空时可设为 ``""`` 或 ``null``。
        device: 模型加载设备，例如 ``cuda`` 或 ``cpu``。
        checkpoint_path: 离线下载好的权重路径，支持 ``.safetensors`` / ``.pt`` / ``.bin``。
    
    Returns:
        返回 ``model``、``preprocess`` 和 ``tokenizer``，用于训练或零样本推理。
    """
    require_package("open_clip", "pip install open_clip_torch")
    import open_clip

    local_ckpt = str(checkpoint_path or "").strip()
    if local_ckpt:
        ckpt_path = Path(local_ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"OpenCLIP 本地权重不存在: {ckpt_path}")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None, device=device)
        state = _load_openclip_state_dict(str(ckpt_path), device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] OpenCLIP 本地权重缺少 {len(missing)} 个参数，示例: {missing[:5]}")
        if unexpected:
            print(f"[WARN] OpenCLIP 本地权重多出 {len(unexpected)} 个参数，示例: {unexpected[:5]}")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _load_openclip_state_dict(checkpoint_path: str, device):
    """读取离线 OpenCLIP 权重文件并规范化为 ``state_dict``。
    
    Args:
        checkpoint_path: 本地权重文件路径。
        device: 权重映射设备。
    
    Returns:
        可直接传给 ``model.load_state_dict`` 的参数字典。
    """
    suffix = Path(checkpoint_path).suffix.lower()
    if suffix == ".safetensors":
        require_package("safetensors", "pip install safetensors")
        from safetensors.torch import load_file

        state = load_file(checkpoint_path, device=str(device))
    else:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def classify_image(source: str, class_names: List[str], cfg, device):
    """执行单次推理或分类逻辑，输出结构化预测结果。
    
    Args:
        source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    from PIL import Image

    openclip_cfg = cfg["openclip"]
    model, preprocess, tokenizer = load_openclip_model(openclip_cfg["model_name"], openclip_cfg.get("pretrained", ""), device, openclip_cfg.get("checkpoint_path", ""))
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
    """将运行结果、配置或中间产物写入磁盘，便于复用和排查问题。
    
    Args:
        result: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        output: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        topk: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}: {score:.4f}" for name, score in result[:topk]]
    Path(output).write_text("\n".join(lines), encoding="utf-8")
