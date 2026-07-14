import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import AnimalDetectionDataset, detection_collate
from src.models.detector import AnimalDetector
from src.models.loss import DetectionLoss
from src.trainers.common import create_train_output_dir, select_device, set_seed
from src.trainers.detection_engine import run_detection_epoch, save_detection_checkpoint
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.tracker import MetricTracker
from src.utils.visualization import save_loss_curve


def get_tiny_cfg(cfg):
    """返回 tiny_detector 专属配置；旧配置文件没有该分组时回退到 train/model。"""
    tiny = cfg.get("tiny_detector", {})
    train_cfg = {**cfg["train"], **tiny}
    model_cfg = {**cfg["model"], **tiny}
    return train_cfg, model_cfg


def build_dataloaders(cfg, train_cfg):
    """根据配置构建可复用组件，降低入口函数中的业务耦合。
    
    Args:
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        train_cfg: tiny_detector 训练配置，优先来自 tiny_detector 分组。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    train_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["train_images"], cfg["data"]["train_labels"], cfg["data"]["image_size"])
    val_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["val_images"], cfg["data"]["val_labels"], cfg["data"]["image_size"])
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    return train_loader, val_loader


def build_components(train_cfg, model_cfg, device):
    """根据配置构建可复用组件，降低入口函数中的业务耦合。
    
    Args:
        train_cfg: tiny_detector 训练配置，优先来自 tiny_detector 分组。
        model_cfg: tiny_detector 模型配置，优先来自 tiny_detector 分组。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    model = AnimalDetector(model_cfg["num_classes"], model_cfg["num_anchors"], model_cfg["width_mult"]).to(device)
    criterion = DetectionLoss(model_cfg["anchors"], model_cfg["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    return model, criterion, optimizer


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="Train tiny animal detector")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg, model_cfg = get_tiny_cfg(cfg)
    set_seed(cfg["project"]["seed"])
    run_dir = create_train_output_dir(cfg["project"]["output_dir"], "tiny_detector")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_tiny_detector", cfg["project"]["log_dir"])
    logger.info("本次训练输出目录: %s", run_dir)
    tracker = MetricTracker(cfg["project"]["log_dir"], "train_tiny_detector")
    device = select_device(train_cfg["device"])
    logger.info("使用设备: %s", device)

    train_loader, val_loader = build_dataloaders(cfg, train_cfg)
    model, criterion, optimizer = build_components(train_cfg, model_cfg, device)
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
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        train_loss = run_detection_epoch(model, train_loader, criterion, optimizer, device, train=True, tracker=tracker, epoch=epoch, phase="train")
        val_loss = run_detection_epoch(model, val_loader, criterion, optimizer, device, train=False, tracker=tracker, epoch=epoch, phase="val")
        logger.info("Epoch %03d | train total %.4f box %.4f obj %.4f cls %.4f | val total %.4f", epoch, train_loss["total"], train_loss["box"], train_loss["obj"], train_loss["cls"], val_loss["total"])
        tracker.log({"epoch": epoch, "phase": "epoch", "train_total": train_loss["total"], "train_box": train_loss["box"], "train_obj": train_loss["obj"], "train_cls": train_loss["cls"], "val_total": val_loss["total"], "val_box": val_loss["box"], "val_obj": val_loss["obj"], "val_cls": val_loss["cls"]})
        history["train_total"].append(train_loss["total"])
        history["val_total"].append(val_loss["total"])
        history["train_box"].append(train_loss["box"])
        history["train_obj"].append(train_loss["obj"])
        history["train_cls"].append(train_loss["cls"])
        save_loss_curve(history, str(run_dir / "tiny_detector_loss_curve.png"))
        is_best = val_loss["total"] < best_val
        if is_best:
            best_val = val_loss["total"]
            logger.info("保存新的 best checkpoint，val loss: %.4f", best_val)
        save_detection_checkpoint(ckpt_dir, epoch, model, optimizer, best_val, cfg, is_best, train_cfg["checkpoint_interval"])


if __name__ == "__main__":
    main()
