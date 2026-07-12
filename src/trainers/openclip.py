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
from src.utils.visualization import save_loss_curve, save_metric_curves


class OpenCLIPImageDataset(Dataset):
    """复用分类数据集样本路径，但使用 OpenCLIP 官方 preprocess。"""

    def __init__(self, root: str, split: str, class_names: List[str], preprocess) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``OpenCLIPImageDataset``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            root: 数据集根目录路径；检测任务指向 detection 目录，识别任务指向 recognition 目录。
            split: 数据划分名称，通常为 train、val 或 test，用于选择对应子目录。
            class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
            preprocess: OpenCLIP 官方图像预处理函数，与预训练权重的归一化方式保持一致。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        base = AnimalClassificationDataset(root, split, class_names, image_size=224)
        self.samples = base.samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        """返回数据集或容器中的样本数量，供 DataLoader 或外部迭代逻辑使用。
        
        所属类: ``OpenCLIPImageDataset``。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """按索引读取一个样本，并完成必要的数据解码、预处理和标签转换。
        
        所属类: ``OpenCLIPImageDataset``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            idx: 样本索引，由 PyTorch DataLoader 传入，用于读取指定图片和标签。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), torch.tensor(label, dtype=torch.long)


class OpenCLIPLinearClassifier(nn.Module):
    """冻结 OpenCLIP 图像编码器，只训练轻量分类头。"""

    def __init__(self, clip_model, embed_dim: int, num_classes: int, freeze_encoder: bool = True) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``OpenCLIPLinearClassifier``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            clip_model: OpenCLIP 模型实例，提供 encode_image 和 encode_text 特征编码接口。
            embed_dim: OpenCLIP 图像特征维度，用于构建线性分类头输入层。
            num_classes: 类别数量，必须与 class_names 长度一致。
            freeze_encoder: 是否冻结预训练图像编码器；True 时只训练线性分类头。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            images: 输入图像张量批次，形状通常为 [B, C, H, W]。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.classifier(features.float())


def infer_embed_dim(clip_model, preprocess, device) -> int:
    """执行单次推理或分类逻辑，输出结构化预测结果。
    
    Args:
        clip_model: OpenCLIP 模型实例，提供 encode_image 和 encode_text 特征编码接口。
        preprocess: OpenCLIP 官方图像预处理函数，与预训练权重的归一化方式保持一致。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        features = clip_model.encode_image(dummy)
    return int(features.shape[-1])


def run_epoch(model, loader, criterion, optimizer, device, train: bool, tracker: MetricTracker, epoch: int) -> Dict[str, float]:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        model: 待训练或待推理的 PyTorch 模型实例。
        loader: PyTorch DataLoader，负责按 batch 提供训练或验证数据。
        criterion: 损失函数对象，用于根据模型输出和标签计算训练损失。
        optimizer: 优化器对象，训练阶段用于清梯度、反向传播后更新参数。
        device: torch 运行设备，例如 cuda、cuda:0 或 cpu。
        train: 布尔值；为 True 时启用训练模式和反向传播，为 False 时执行验证/评估。
        tracker: 指标或推理跟踪器，用于写入 jsonl/csv 结构化日志。
        epoch: 当前训练轮次，从 1 开始记录，用于日志、曲线和 checkpoint 命名。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    # 设置模型为训练模式：启用Dropout、BatchNorm训练行为；传入train=False则切换eval验证模式
    model.train(train)
    # 累计整个epoch的总损失
    total_loss = 0.0
    num_classes = len(getattr(loader.dataset, "class_names", [])) or int(getattr(model.classifier, "out_features", 0))
    total_correct = 0
    total_top5_correct = 0
    total_samples = 0
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)
    # 标记当前是训练阶段还是验证阶段
    phase = "train" if train else "val"
    # 训练时启用梯度计算，验证时关闭梯度计算节省显存
    context = torch.enable_grad() if train else torch.no_grad()
    # 进入梯度上下文环境
    with context:
        # 遍历dataloader每一个batch，tqdm生成进度条，step从1开始计数
        for step, (images, labels) in enumerate(tqdm(loader, desc=f"openclip_{phase}"), start=1):
            # 图像张量搬运到GPU/指定计算设备
            images = images.to(device)
            # 标签张量搬运到GPU/指定计算设备
            labels = labels.to(device)
            # 模型前向传播，输出原始分类得分logits [batch, num_classes]
            logits = model(images)
            # 计算当前batch的损失（交叉熵损失）
            loss = criterion(logits, labels)
            # 仅训练阶段执行反向传播与参数更新
            if train:
                # 清空优化器历史梯度，set_to_none=True直接释放梯度显存更省内存
                optimizer.zero_grad(set_to_none=True)
                # 损失反向传播，计算各参数梯度
                loss.backward()
                # 根据梯度更新模型权重参数
                optimizer.step()
            pred = logits.argmax(dim=1)
            correct = int((pred == labels).sum().item())
            topk = min(5, logits.shape[1])
            top5_correct = int((logits.topk(topk, dim=1).indices == labels.view(-1, 1)).any(dim=1).sum().item())
            total_correct += correct
            total_top5_correct += top5_correct
            pred_cpu = pred.detach().cpu()
            labels_cpu = labels.detach().cpu()
            for cls_id in range(num_classes):
                cls_pred = pred_cpu == cls_id
                cls_true = labels_cpu == cls_id
                tp[cls_id] += int((cls_pred & cls_true).sum().item())
                fp[cls_id] += int((cls_pred & ~cls_true).sum().item())
                fn[cls_id] += int((~cls_pred & cls_true).sum().item())
            # loss.detach()切断梯度计算图，cpu()移回CPU；乘以batch样本数，按样本加权累计总损失
            total_loss += float(loss.detach().cpu()) * labels.numel()
            # 累加当前batch样本数量到总样本数
            total_samples += labels.numel()
            tracker.log({"epoch": epoch, "phase": phase, "step": step, "loss": float(loss.detach().cpu()), "acc": correct / max(labels.numel(), 1), "top5_acc": top5_correct / max(labels.numel(), 1), "lr": optimizer.param_groups[0]["lr"]})
    precision_per_class = tp.float() / (tp + fp).clamp(min=1).float()
    recall_per_class = tp.float() / (tp + fn).clamp(min=1).float()
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp(min=1e-8)
    macro_precision = float(precision_per_class.mean().item())
    macro_recall = float(recall_per_class.mean().item())
    macro_f1 = float(f1_per_class.mean().item())
    return {"total": total_loss / max(total_samples, 1), "acc": total_correct / max(total_samples, 1), "top5_acc": total_top5_correct / max(total_samples, 1), "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1}


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
    num_workers = int(openclip_cfg.get("num_workers", cfg["data"]["num_workers"]))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=openclip_cfg["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=openclip_cfg["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=openclip_cfg["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    ckpt_dir = Path(cfg["project"]["output_dir"]) / "checkpoints" / "openclip"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    metric_history: Dict[str, List[float]] = {"train_acc": [], "val_acc": [], "train_top5_acc": [], "val_top5_acc": [], "train_macro_f1": [], "val_macro_f1": [], "train_macro_precision": [], "val_macro_precision": [], "train_macro_recall": [], "val_macro_recall": []}
    best_val = float("inf")

    for epoch in range(1, openclip_cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True, tracker, epoch)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False, tracker, epoch)
        loss_history["train_loss"].append(train_metrics["total"])
        loss_history["val_loss"].append(val_metrics["total"])
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            metric_history[f"{prefix}_acc"].append(metrics["acc"])
            metric_history[f"{prefix}_top5_acc"].append(metrics["top5_acc"])
            metric_history[f"{prefix}_macro_f1"].append(metrics["macro_f1"])
            metric_history[f"{prefix}_macro_precision"].append(metrics["macro_precision"])
            metric_history[f"{prefix}_macro_recall"].append(metrics["macro_recall"])
        save_loss_curve(loss_history, openclip_cfg["loss_curve"])
        save_metric_curves(metric_history, openclip_cfg.get("metric_curve", "outputs/openclip_metric_curve.png"), "OpenCLIP Classification Metrics")
        tracker.log({"epoch": epoch, "phase": "epoch", "train_loss": train_metrics["total"], "train_acc": train_metrics["acc"], "train_top5_acc": train_metrics["top5_acc"], "train_macro_precision": train_metrics["macro_precision"], "train_macro_recall": train_metrics["macro_recall"], "train_macro_f1": train_metrics["macro_f1"], "val_loss": val_metrics["total"], "val_acc": val_metrics["acc"], "val_top5_acc": val_metrics["top5_acc"], "val_macro_precision": val_metrics["macro_precision"], "val_macro_recall": val_metrics["macro_recall"], "val_macro_f1": val_metrics["macro_f1"], "lr": optimizer.param_groups[0]["lr"]})
        logger.info("Epoch %03d | train loss %.4f acc %.4f top5 %.4f f1 %.4f | val loss %.4f acc %.4f top5 %.4f f1 %.4f", epoch, train_metrics["total"], train_metrics["acc"], train_metrics["top5_acc"], train_metrics["macro_f1"], val_metrics["total"], val_metrics["acc"], val_metrics["top5_acc"], val_metrics["macro_f1"])
        ckpt = {"epoch": epoch, "model": model.classifier.state_dict(), "class_names": class_names, "embed_dim": embed_dim, "config": cfg}
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(ckpt, ckpt_dir / "best_linear.pt")
            logger.info("保存 OpenCLIP linear best checkpoint: %.4f", best_val)


if __name__ == "__main__":
    main()
