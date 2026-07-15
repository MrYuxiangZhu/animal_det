import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import AnimalDetectionDataset, detection_collate
from src.models.multiscale_loss import MultiScaleDetectionLoss
from src.models.timm_fpn_detector import TimmFPNDetector
from src.trainers.common import create_train_output_dir, select_device, set_seed
from src.trainers.detection_engine import run_detection_epoch, save_detection_checkpoint
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import MetricTracker
from src.utils.visualization import save_loss_curve


def get_pro_cfg(cfg):
    return cfg["tiny_detector_pro"]


def build_dataloaders(cfg, train_cfg):
    train_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["train_images"], cfg["data"]["train_labels"], cfg["data"]["image_size"])
    val_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["val_images"], cfg["data"]["val_labels"], cfg["data"]["image_size"])
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    return train_loader, val_loader


def build_components(train_cfg, device):
    anchors = train_cfg["anchors"]
    model = TimmFPNDetector(
        num_classes=train_cfg["num_classes"],
        backbone_name=train_cfg["backbone_name"],
        pretrained=train_cfg["pretrained"],
        out_channels=train_cfg["out_channels"],
        num_anchors=len(anchors[0]),
        out_indices=tuple(train_cfg.get("out_indices", [2, 3, 4])),
    ).to(device)
    criterion = MultiScaleDetectionLoss(
        anchors=anchors,
        num_classes=train_cfg["num_classes"],
        box_gain=train_cfg.get("box_gain", 5.0),
        obj_gain=train_cfg.get("obj_gain", 1.0),
        cls_gain=train_cfg.get("cls_gain", 1.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        ignore_iou_threshold=train_cfg.get("ignore_iou_threshold", 0.5),
    ).to(device)

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": train_cfg.get("backbone_learning_rate", train_cfg["learning_rate"] * 0.1)},
            {"params": other_params, "lr": train_cfg["learning_rate"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )
    return model, criterion, optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train timm FPN pro animal detector")
    parser.add_argument("--config", default="configs/coco_animals_10cls.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = get_pro_cfg(cfg)
    set_seed(cfg["project"]["seed"])

    run_dir = create_train_output_dir(cfg["project"]["output_dir"], "tiny_detector_pro")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train_tiny_detector_pro", cfg["project"]["log_dir"])
    logger.info("本次训练输出目录: %s", run_dir)
    tracker = MetricTracker(cfg["project"]["log_dir"], "train_tiny_detector_pro")
    device = select_device(train_cfg["device"])
    logger.info("使用设备: %s", device)

    train_loader, val_loader = build_dataloaders(cfg, train_cfg)
    model, criterion, optimizer = build_components(train_cfg, device)

    start_epoch = 1
    best_val = float("inf")
    if train_cfg.get("resume"):
        ckpt = torch.load(train_cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        logger.info("从 checkpoint 恢复: %s", train_cfg["resume"])

    history: Dict[str, List[float]] = {"train_total": [], "val_total": [], "train_box": [], "train_obj": [], "train_cls": []}
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    patience = train_cfg.get("early_stop_patience", 0)
    bad_epochs = 0

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        if freeze_epochs > 0:
            trainable = epoch > freeze_epochs
            model.set_backbone_trainable(trainable)
            logger.info("Epoch %03d | backbone trainable: %s", epoch, trainable)

        train_loss = run_detection_epoch(model, train_loader, criterion, optimizer, device, train=True, tracker=tracker, epoch=epoch, phase="train")
        val_loss = run_detection_epoch(model, val_loader, criterion, optimizer, device, train=False, tracker=tracker, epoch=epoch, phase="val")

        logger.info(
            "Epoch %03d | train total %.4f box %.4f obj %.4f cls %.4f | val total %.4f box %.4f obj %.4f cls %.4f",
            epoch,
            train_loss["total"],
            train_loss["box"],
            train_loss["obj"],
            train_loss["cls"],
            val_loss["total"],
            val_loss["box"],
            val_loss["obj"],
            val_loss["cls"],
        )
        tracker.log({"epoch": epoch, "phase": "epoch", "train_total": train_loss["total"], "train_box": train_loss["box"], "train_obj": train_loss["obj"], "train_cls": train_loss["cls"], "val_total": val_loss["total"], "val_box": val_loss["box"], "val_obj": val_loss["obj"], "val_cls": val_loss["cls"]})

        history["train_total"].append(train_loss["total"])
        history["val_total"].append(val_loss["total"])
        history["train_box"].append(train_loss["box"])
        history["train_obj"].append(train_loss["obj"])
        history["train_cls"].append(train_loss["cls"])
        save_loss_curve(history, str(run_dir / "tiny_detector_pro_loss_curve.png"))

        is_best = val_loss["total"] < best_val
        if is_best:
            best_val = val_loss["total"]
            bad_epochs = 0
            logger.info("保存新的 best checkpoint，val loss: %.4f", best_val)
        else:
            bad_epochs += 1

        save_detection_checkpoint(ckpt_dir, epoch, model, optimizer, best_val, cfg, is_best, train_cfg["checkpoint_interval"])

        if patience > 0 and bad_epochs >= patience:
            logger.info("Early stopping triggered at epoch %03d, best val loss %.4f", epoch, best_val)
            break


if __name__ == "__main__":
    main()
