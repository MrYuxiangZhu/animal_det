from pathlib import Path
from typing import List

import torch

from src.utils.industrial import require_package
from src.utils.third_party import import_open_clip, OPENCLIP_LOCAL_SRC


def load_openclip_model(model_name: str, pretrained: str, device, checkpoint_path: str = ""):
    """加载 OpenCLIP 模型，支持在线预训练标签和离线本地权重。
    
    Args:
        model_name: 模型结构名称，例如 ViT-B-32、resnet50 或第三方框架配置中的网络名。
        pretrained: 预训练权重标签；为空时不联网下载，通常配合本地 checkpoint_path 使用。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
        checkpoint_path: 本地权重文件路径，支持 safetensors、pt 或 bin 等格式。
    
    Returns:
        返回 ``model``、``preprocess`` 和 ``tokenizer``，用于训练或零样本推理。
    """
    if not OPENCLIP_LOCAL_SRC.exists():
        require_package("open_clip", "pip install open_clip_torch")
    open_clip = import_open_clip()
    print(f"[INFO] OpenCLIP module: {getattr(open_clip, '__file__', 'unknown')}")

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
        checkpoint_path: 本地权重文件路径，支持 safetensors、pt 或 bin 等格式。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
    
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
        source: 输入图片、视频、目录或数据源路径，由推理入口传入。
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        result: 模型推理结果列表，通常包含类别名和置信度。
        output: 推理结果输出路径，可以是图片、视频、文本或 JSON 文件。
        topk: 输出排名最高的类别数量。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}: {score:.4f}" for name, score in result[:topk]]
    Path(output).write_text("\n".join(lines), encoding="utf-8")
