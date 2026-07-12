import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.classification_dataset import AnimalClassificationDataset
from src.models.clip_like import MiniCLIP, SimpleTokenizer, clip_contrastive_loss
from src.trainers.common import select_device, set_seed
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.visualization import save_loss_curve


def make_prompts(class_names: List[str]) -> List[str]:
    """根据输入信息生成后续模型或训练流程需要的辅助数据。
    
    Args:
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    return [f"a photo of a {name}" for name in class_names]


def run_epoch(model, tokenizer, loader, class_names, optimizer, device, train: bool) -> Dict[str, float]:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        tokenizer: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        loader: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        optimizer: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        train: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="clip_train" if train else "clip_val"):
            images = images.to(device)
            labels = labels.to(device)
            prompts = [f"a photo of a {class_names[int(label.item())]}" for label in labels]
            tokens = tokenizer.encode(prompts).to(device)
            logits_i, logits_t = model(images, tokens)
            loss = clip_contrastive_loss(logits_i, logits_t)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            with torch.no_grad():
                all_tokens = tokenizer.encode(make_prompts(class_names)).to(device)
                image_features = model.encode_image(images)
                text_features = model.encode_text(all_tokens)
                logits = image_features @ text_features.t()
                acc = (logits.argmax(dim=1) == labels).float().mean()
            total_loss += float(loss.detach().cpu())
            total_acc += float(acc.detach().cpu())
            steps += 1
    return {"total": total_loss / max(steps, 1), "acc": total_acc / max(steps, 1)}


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    parser = argparse.ArgumentParser(description="Train MiniCLIP for animal zero-shot classification")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = setup_logger("train_clip", cfg["project"]["log_dir"])
    device = select_device(cfg["train"]["device"])
    class_names = cfg["data"]["class_names"]
    clip_cfg = cfg["clip"]

    train_set = AnimalClassificationDataset(clip_cfg["data_root"], "train", class_names, clip_cfg["image_size"])
    val_set = AnimalClassificationDataset(clip_cfg["data_root"], "val", class_names, clip_cfg["image_size"])
    train_loader = DataLoader(train_set, batch_size=clip_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=clip_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    tokenizer = SimpleTokenizer(context_length=clip_cfg["context_length"])
    model = MiniCLIP(tokenizer.vocab_size, clip_cfg["context_length"], clip_cfg["embed_dim"], clip_cfg["width_mult"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=clip_cfg["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    ckpt_dir = Path(cfg["project"]["output_dir"]) / "checkpoints" / "clip"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: Dict[str, List[float]] = {"train_total": [], "val_total": []}

    for epoch in range(1, clip_cfg["epochs"] + 1):
        train_loss = run_epoch(model, tokenizer, train_loader, class_names, optimizer, device, True)
        val_loss = run_epoch(model, tokenizer, val_loader, class_names, optimizer, device, False)
        history["train_total"].append(train_loss["total"])
        history["val_total"].append(val_loss["total"])
        save_loss_curve(history, clip_cfg["loss_curve"])
        logger.info("Epoch %03d | train loss %.4f acc %.4f | val loss %.4f acc %.4f", epoch, train_loss["total"], train_loss["acc"], val_loss["total"], val_loss["acc"])
        ckpt = {"epoch": epoch, "model": model.state_dict(), "tokenizer": {"context_length": tokenizer.context_length}, "class_names": class_names, "config": cfg}
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info("保存 CLIP best checkpoint: %.4f", best_val)


if __name__ == "__main__":
    main()
