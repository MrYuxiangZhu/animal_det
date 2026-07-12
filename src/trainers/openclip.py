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
        base = AnimalClassificationDataset(root, split, class_names, image_size=224)
        self.samples = base.samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), torch.tensor(label, dtype=torch.long)


class OpenCLIPLinearClassifier(nn.Module):
    """冻结 OpenCLIP 图像编码器，只训练轻量分类头。"""

    def __init__(self, clip_model, embed_dim: int, num_classes: int, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder
        self.classifier = nn.Linear(embed_dim, num_classes)
        if freeze_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.classifier(features.float())


def infer_embed_dim(clip_model, preprocess, device) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        features = clip_model.encode_image(dummy)
    return int(features.shape[-1])


def run_epoch(model, loader, criterion, optimizer, device, train: bool, tracker: MetricTracker, epoch: int) -> Dict[str, float]:
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

    clip_model, preprocess, _ = load_openclip_model(openclip_cfg["model_name"], openclip_cfg["pretrained"], device)
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
