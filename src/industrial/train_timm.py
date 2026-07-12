import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.classification_dataset import AnimalClassificationDataset
from src.train import select_device, set_seed
from src.utils.config import load_config
from src.utils.industrial import require_package
from src.utils.logger import setup_logger
from src.utils.visualization import save_loss_curve


class TimmClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        require_package("timm", "pip install timm")
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def run_epoch(model, loader, optimizer, device, train: bool) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="timm_train" if train else "timm_val"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_acc += float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())
            steps += 1
    return {"total": total_loss / max(steps, 1), "acc": total_acc / max(steps, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train animal classifier with timm backbone")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = setup_logger("train_timm", cfg["project"]["log_dir"])
    timm_cfg = cfg["timm"]
    class_names: List[str] = cfg["data"]["class_names"]
    device = select_device(cfg["train"]["device"])

    train_set = AnimalClassificationDataset(timm_cfg["data_root"], "train", class_names, timm_cfg["image_size"])
    val_set = AnimalClassificationDataset(timm_cfg["data_root"], "val", class_names, timm_cfg["image_size"])
    train_loader = DataLoader(train_set, batch_size=timm_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=timm_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    model = TimmClassifier(timm_cfg["model_name"], len(class_names), timm_cfg["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=timm_cfg["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    ckpt_dir = Path(cfg["project"]["output_dir"]) / "checkpoints" / "timm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: Dict[str, List[float]] = {"train_total": [], "val_total": []}

    for epoch in range(1, timm_cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True)
        val_loss = run_epoch(model, val_loader, optimizer, device, False)
        history["train_total"].append(train_loss["total"])
        history["val_total"].append(val_loss["total"])
        save_loss_curve(history, timm_cfg["loss_curve"])
        logger.info("Epoch %03d | train loss %.4f acc %.4f | val loss %.4f acc %.4f", epoch, train_loss["total"], train_loss["acc"], val_loss["total"], val_loss["acc"])
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save({"model": model.state_dict(), "class_names": class_names, "config": cfg}, ckpt_dir / "best.pt")
            logger.info("保存 timm best checkpoint: %.4f", best_val)


if __name__ == "__main__":
    main()
