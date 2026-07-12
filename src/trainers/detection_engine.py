from typing import Dict, Optional

import torch
from tqdm import tqdm


def run_detection_epoch(model, loader, criterion, optimizer, device, train: bool, tracker: Optional[object] = None, epoch: int = 0, phase: str = "train") -> Dict[str, float]:
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
        phase: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
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
        ckpt_dir: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        epoch: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        model: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        optimizer: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        best_val: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cfg: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        is_best: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        checkpoint_interval: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val": best_val, "config": cfg}
    if epoch % checkpoint_interval == 0:
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
    if is_best:
        torch.save(ckpt, ckpt_dir / "best.pt")
