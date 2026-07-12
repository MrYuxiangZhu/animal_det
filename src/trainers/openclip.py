import argparse
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.classification_dataset import AnimalClassificationDataset
from src.specialized.openclip_adapter import load_openclip_model
from src.trainers.common import select_device, set_seed
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import MetricTracker
from src.utils.visualization import save_loss_curve


class OpenCLIPImageDataset(Dataset):
    """复用分类数据集样本路径，但使用 OpenCLIP 官方 preprocess。"""

    def __init__(self, root: str, split: str, class_names: List[str], preprocess) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``OpenCLIPImageDataset``。
        
        Args:
            root: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            split: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            preprocess: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        base = AnimalClassificationDataset(root, split, class_names, image_size=224)
        self.samples = base.samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        """返回数据集或容器中的样本数量，供 DataLoader 或外部迭代逻辑使用。
        
        所属类: ``OpenCLIPImageDataset``。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """按索引读取一个样本，并完成必要的数据解码、预处理和标签转换。
        
        所属类: ``OpenCLIPImageDataset``。
        
        Args:
            idx: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), torch.tensor(label, dtype=torch.long)


class OpenCLIPLinearClassifier(nn.Module):
    """冻结 OpenCLIP 图像编码器，只训练轻量分类头。"""

    def __init__(self, clip_model, embed_dim: int, num_classes: int, freeze_encoder: bool = True) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``OpenCLIPLinearClassifier``。
        
        Args:
            clip_model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            embed_dim: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            num_classes: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            freeze_encoder: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        super().__init__()
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder
        self.classifier = nn.Linear(embed_dim, num_classes)
        if freeze_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``OpenCLIPLinearClassifier``。
        
        Args:
            images: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.classifier(features.float())


def infer_embed_dim(clip_model, preprocess, device) -> int:
    """执行单次推理或分类逻辑，输出结构化预测结果。
    
    Args:
        clip_model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        preprocess: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        features = clip_model.encode_image(dummy)
    return int(features.shape[-1])


def run_epoch(model, loader, criterion, optimizer, device, train: bool, tracker: MetricTracker, epoch: int) -> Dict[str, float]:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        loader: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        criterion: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        optimizer: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        device: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        train: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        tracker: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        epoch: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    phase = "train" if train else "val"
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, (images, labels) in enumerate(tqdm(loader, desc=f"openclip_{phase}"), start=1):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            pred = logits.argmax(dim=1)
            correct = int((pred == labels).sum().item())
            total_correct += correct
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_samples += labels.numel()
            tracker.log({"epoch": epoch, "phase": phase, "step": step, "loss": float(loss.detach().cpu()), "acc": correct / max(labels.numel(), 1)})
    return {"total": total_loss / max(total_samples, 1), "acc": total_correct / max(total_samples, 1)}


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    parser = argparse.ArgumentParser(description="Train OpenCLIP linear classifier for animal recognition")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = setup_logger("train_openclip", cfg["project"]["log_dir"])
    tracker = MetricTracker(cfg["project"]["log_dir"], "train_openclip")
    device = select_device(cfg["train"]["device"])
    class_names = cfg["data"]["class_names"]
    openclip_cfg = cfg["openclip"]

    clip_model, preprocess, _ = load_openclip_model(openclip_cfg["model_name"], openclip_cfg.get("pretrained", ""), device, openclip_cfg.get("checkpoint_path", ""))
    embed_dim = infer_embed_dim(clip_model, preprocess, device)
    model = OpenCLIPLinearClassifier(clip_model, embed_dim, len(class_names), freeze_encoder=openclip_cfg.get("freeze_encoder", True)).to(device)

    train_set = OpenCLIPImageDataset(openclip_cfg["data_root"], "train", class_names, preprocess)
    val_set = OpenCLIPImageDataset(openclip_cfg["data_root"], "val", class_names, preprocess)
    train_loader = DataLoader(train_set, batch_size=openclip_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=openclip_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=openclip_cfg["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    ckpt_dir = Path(cfg["project"]["output_dir"]) / "checkpoints" / "openclip"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: Dict[str, List[float]] = {"train_total": [], "val_total": []}
    best_val = float("inf")

    for epoch in range(1, openclip_cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True, tracker, epoch)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False, tracker, epoch)
        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(val_metrics["total"])
        save_loss_curve(history, openclip_cfg["loss_curve"])
        tracker.log({"epoch": epoch, "phase": "epoch", "train_loss": train_metrics["total"], "train_acc": train_metrics["acc"], "val_loss": val_metrics["total"], "val_acc": val_metrics["acc"]})
        logger.info("Epoch %03d | train loss %.4f acc %.4f | val loss %.4f acc %.4f", epoch, train_metrics["total"], train_metrics["acc"], val_metrics["total"], val_metrics["acc"])
        ckpt = {"epoch": epoch, "model": model.classifier.state_dict(), "class_names": class_names, "embed_dim": embed_dim, "config": cfg}
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(ckpt, ckpt_dir / "best_linear.pt")
            logger.info("保存 OpenCLIP linear best checkpoint: %.4f", best_val)


if __name__ == "__main__":
    main()
