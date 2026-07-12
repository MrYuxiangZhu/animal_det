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
from src.utils.metrics import classification_metric_history, classification_stats_from_counts, log_classification_epoch, log_per_class_metrics, update_classification_history
from src.utils.tracker import MetricTracker
from src.utils.visualization import save_bar_chart, save_loss_curve, save_metric_curves


def make_prompts(class_names: List[str]) -> List[str]:
    """根据输入信息生成后续模型或训练流程需要的辅助数据。
    
    Args:
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    return [f"a photo of a {name}" for name in class_names]


def run_epoch(model, tokenizer, loader, class_names, optimizer, device, train: bool, tracker: MetricTracker, epoch: int) -> Dict[str, float]:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        model: 待训练或待推理的 PyTorch 模型实例。
        tokenizer: 文本 tokenizer，用于把类别 prompt 转换为模型可读 token。
        loader: PyTorch DataLoader，负责按 batch 提供训练或验证数据。
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
        optimizer: 优化器对象，训练阶段用于清梯度、反向传播后更新参数。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
        train: 布尔值；为 True 时启用训练模式和反向传播，为 False 时执行验证/评估。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_samples = 0
    steps = 0
    num_classes = len(class_names)
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)
    phase = "train" if train else "val"
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
                pred = logits.argmax(dim=1)
                correct = int((pred == labels).sum().item())
                topk = min(5, logits.shape[1])
                top5_correct = int((logits.topk(topk, dim=1).indices == labels.view(-1, 1)).any(dim=1).sum().item())
                pred_cpu = pred.detach().cpu()
                labels_cpu = labels.detach().cpu()
                for cls_id in range(num_classes):
                    cls_pred = pred_cpu == cls_id
                    cls_true = labels_cpu == cls_id
                    tp[cls_id] += int((cls_pred & cls_true).sum().item())
                    fp[cls_id] += int((cls_pred & ~cls_true).sum().item())
                    fn[cls_id] += int((~cls_pred & cls_true).sum().item())
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_correct += correct
            total_top5_correct += top5_correct
            total_samples += labels.numel()
            steps += 1
            tracker.log({"epoch": epoch, "phase": phase, "step": steps, "loss": float(loss.detach().cpu()), "acc": correct / max(labels.numel(), 1), "top5_acc": top5_correct / max(labels.numel(), 1), "lr": optimizer.param_groups[0]["lr"]})
    metrics = classification_stats_from_counts(tp, fp, fn, total_correct, total_top5_correct, total_samples)
    metrics["total"] = total_loss / max(total_samples, 1)
    return metrics


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="Train MiniCLIP for animal zero-shot classification")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = setup_logger("train_clip", cfg["project"]["log_dir"])
    tracker = MetricTracker(cfg["project"]["log_dir"], "train_clip")
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
    loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    metric_history: Dict[str, List[float]] = classification_metric_history()

    for epoch in range(1, clip_cfg["epochs"] + 1):
        train_loss = run_epoch(model, tokenizer, train_loader, class_names, optimizer, device, True, tracker, epoch)
        val_loss = run_epoch(model, tokenizer, val_loader, class_names, optimizer, device, False, tracker, epoch)
        update_classification_history(loss_history, metric_history, train_loss, val_loss)
        save_loss_curve(loss_history, clip_cfg["loss_curve"])
        save_metric_curves(metric_history, clip_cfg.get("metric_curve", "outputs/clip_metric_curve.png"), "MiniCLIP Classification Metrics")
        tracker.log({"epoch": epoch, "phase": "epoch", "train_loss": train_loss["total"], "train_acc": train_loss["acc"], "train_top5_acc": train_loss["top5_acc"], "train_macro_precision": train_loss["macro_precision"], "train_macro_recall": train_loss["macro_recall"], "train_macro_f1": train_loss["macro_f1"], "val_loss": val_loss["total"], "val_acc": val_loss["acc"], "val_top5_acc": val_loss["top5_acc"], "val_macro_precision": val_loss["macro_precision"], "val_macro_recall": val_loss["macro_recall"], "val_macro_f1": val_loss["macro_f1"], "lr": optimizer.param_groups[0]["lr"]})
        log_classification_epoch(logger, epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"])
        log_per_class_metrics(logger, epoch, class_names, val_loss, "val")
        save_bar_chart({name: float(val_loss["f1_per_class"][idx]) for idx, name in enumerate(class_names)}, clip_cfg.get("class_f1_curve", "outputs/clip_class_f1.png"), "MiniCLIP Val F1 Per Class", "F1")
        save_bar_chart({name: float(val_loss["precision_per_class"][idx]) for idx, name in enumerate(class_names)}, clip_cfg.get("class_precision_curve", "outputs/clip_class_precision.png"), "MiniCLIP Val Precision Per Class", "Precision")
        save_bar_chart({name: float(val_loss["recall_per_class"][idx]) for idx, name in enumerate(class_names)}, clip_cfg.get("class_recall_curve", "outputs/clip_class_recall.png"), "MiniCLIP Val Recall Per Class", "Recall")
        ckpt = {"epoch": epoch, "model": model.state_dict(), "tokenizer": {"context_length": tokenizer.context_length}, "class_names": class_names, "config": cfg}
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info("保存 CLIP best checkpoint: %.4f", best_val)


if __name__ == "__main__":
    main()
