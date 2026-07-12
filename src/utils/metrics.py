from typing import Dict, List

import torch


def classification_stats(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[str, object]:
    """计算分类任务 batch 或 epoch 级别的核心性能指标。

    Args:
        logits: 模型输出的分类得分张量，形状为 ``[batch_size, num_classes]``。
        labels: 真实类别 ID 张量，形状为 ``[batch_size]``。
        num_classes: 分类类别数量，必须与 logits 的类别维度一致。

    Returns:
        包含 accuracy、top5 accuracy、macro precision、macro recall、macro F1 和逐类指标的字典。
    """
    pred = logits.argmax(dim=1)
    topk = min(5, logits.shape[1])
    top5_correct = (logits.topk(topk, dim=1).indices == labels.view(-1, 1)).any(dim=1).sum().item()
    total = labels.numel()
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)
    pred_cpu = pred.detach().cpu()
    labels_cpu = labels.detach().cpu()
    for cls_id in range(num_classes):
        cls_pred = pred_cpu == cls_id
        cls_true = labels_cpu == cls_id
        tp[cls_id] = int((cls_pred & cls_true).sum().item())
        fp[cls_id] = int((cls_pred & ~cls_true).sum().item())
        fn[cls_id] = int((~cls_pred & cls_true).sum().item())
    return classification_stats_from_counts(tp, fp, fn, int((pred == labels).sum().item()), int(top5_correct), total)


def classification_stats_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, correct: int, top5_correct: int, total: int) -> Dict[str, object]:
    """根据累计混淆统计计算分类性能指标。

    Args:
        tp: 每个类别的 true positive 数量。
        fp: 每个类别的 false positive 数量。
        fn: 每个类别的 false negative 数量。
        correct: 所有类别累计预测正确样本数。
        top5_correct: 真实类别落入 top-5 预测的样本数。
        total: 参与评估的样本总数。

    Returns:
        分类指标字典，适合写入日志、CSV 和绘图函数。
    """
    precision_per_class = tp.float() / (tp + fp).clamp(min=1).float()
    recall_per_class = tp.float() / (tp + fn).clamp(min=1).float()
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp(min=1e-8)
    return {
        "acc": correct / max(total, 1),
        "top5_acc": top5_correct / max(total, 1),
        "macro_precision": float(precision_per_class.mean().item()),
        "macro_recall": float(recall_per_class.mean().item()),
        "macro_f1": float(f1_per_class.mean().item()),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def log_classification_epoch(logger, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float, prefix: str = "") -> None:
    """把分类任务 epoch 级核心指标打印到日志。

    Args:
        logger: Python logger 实例，通常由 ``setup_logger`` 创建。
        epoch: 当前训练轮次。
        train_metrics: 训练集 epoch 指标字典。
        val_metrics: 验证集 epoch 指标字典。
        lr: 当前学习率。
        prefix: 日志前缀，用于区分不同模型或实验。
    """
    tag = f"{prefix} " if prefix else ""
    logger.info(
        "%sEpoch %03d | train loss %.4f acc %.4f top5 %.4f precision %.4f recall %.4f f1 %.4f | val loss %.4f acc %.4f top5 %.4f precision %.4f recall %.4f f1 %.4f | gap_loss %.4f gap_acc %.4f lr %.6g",
        tag,
        epoch,
        train_metrics["total"],
        train_metrics["acc"],
        train_metrics["top5_acc"],
        train_metrics["macro_precision"],
        train_metrics["macro_recall"],
        train_metrics["macro_f1"],
        val_metrics["total"],
        val_metrics["acc"],
        val_metrics["top5_acc"],
        val_metrics["macro_precision"],
        val_metrics["macro_recall"],
        val_metrics["macro_f1"],
        val_metrics["total"] - train_metrics["total"],
        train_metrics["acc"] - val_metrics["acc"],
        lr,
    )


def log_per_class_metrics(logger, epoch: int, class_names: List[str], metrics: Dict[str, object], phase: str = "val") -> None:
    """把每个类别的 precision、recall 和 F1 打印到日志。

    Args:
        logger: Python logger 实例。
        epoch: 当前训练轮次。
        class_names: 类别名列表，索引顺序必须与模型输出类别 ID 一致。
        metrics: 包含 ``precision_per_class``、``recall_per_class``、``f1_per_class`` 的指标字典。
        phase: 指标所属阶段，通常为 ``train`` 或 ``val``。
    """
    for key, label in (("precision_per_class", "precision"), ("recall_per_class", "recall"), ("f1_per_class", "f1")):
        values = {name: round(float(metrics[key][idx]), 4) for idx, name in enumerate(class_names)}
        logger.info("Epoch %03d | per-class %s %s: %s", epoch, phase, label, values)


def update_classification_history(loss_history: Dict[str, List[float]], metric_history: Dict[str, List[float]], train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    """更新分类训练的 loss 和 metric 曲线历史。

    Args:
        loss_history: loss 曲线历史字典，会追加 train_loss 和 val_loss。
        metric_history: 指标曲线历史字典，会追加 acc/top5/precision/recall/F1。
        train_metrics: 当前 epoch 训练集指标。
        val_metrics: 当前 epoch 验证集指标。
    """
    loss_history["train_loss"].append(train_metrics["total"])
    loss_history["val_loss"].append(val_metrics["total"])
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key in ("acc", "top5_acc", "macro_f1", "macro_precision", "macro_recall"):
            metric_history[f"{prefix}_{key}"].append(metrics[key])


def classification_metric_history() -> Dict[str, List[float]]:
    """创建分类任务指标曲线历史容器。

    Returns:
        包含 train/val 的 acc、top5、macro precision、macro recall、macro F1 的空列表字典。
    """
    return {"train_acc": [], "val_acc": [], "train_top5_acc": [], "val_top5_acc": [], "train_macro_f1": [], "val_macro_f1": [], "train_macro_precision": [], "val_macro_precision": [], "train_macro_recall": [], "val_macro_recall": []}
