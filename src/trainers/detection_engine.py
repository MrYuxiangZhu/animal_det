from typing import Dict, Optional

import torch
from tqdm import tqdm


def run_detection_epoch(model, loader, criterion, optimizer, device, train: bool, tracker: Optional[object] = None, epoch: int = 0, phase: str = "train") -> Dict[str, float]:
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
        phase: 阶段名称，例如 train、val 或 epoch，用于日志和指标区分。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    model.train(train)
    sums = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    steps = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets, _ in tqdm(loader, desc="train" if train else "val"):
            images = images.to(device)
            preds = model(images)
            loss, parts = criterion(preds, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            for key in sums:
                sums[key] += parts[key]
            if tracker is not None:
                tracker.log({"epoch": epoch, "phase": phase, "step": steps + 1, **{f"step_{k}": v for k, v in parts.items()}})
            steps += 1
    return {key: value / max(steps, 1) for key, value in sums.items()}


def save_detection_checkpoint(ckpt_dir, epoch, model, optimizer, best_val, cfg, is_best: bool, checkpoint_interval: int) -> None:
    """将运行结果、配置或中间产物写入磁盘，便于复用和排查问题。
    
    Args:
        ckpt_dir: 目录路径参数，函数会在该目录下查找输入或保存输出。
        epoch: 当前训练轮次，从 1 开始记录，用于日志、曲线和 checkpoint 命名。
        model: 待训练或待推理的 PyTorch 模型实例。
        optimizer: 优化器对象，训练阶段用于清梯度、反向传播后更新参数。
        best_val: best_val 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        cfg: 已解析的 YAML 配置字典，包含 project/data/model/train/infer 等运行参数。
        is_best: is_best 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        checkpoint_interval: checkpoint_interval 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val": best_val, "config": cfg}
    if epoch % checkpoint_interval == 0:
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
    if is_best:
        torch.save(ckpt, ckpt_dir / "best.pt")
