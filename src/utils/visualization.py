from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_curve(history: Dict[str, List[float]], output_path: str) -> None:
    """保存训练总损失及子损失曲线。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for key, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_metric_curves(history: Dict[str, List[float]], output_path: str, title: str = "Training Metrics") -> None:
    """保存除 loss 以外的训练指标曲线。

    Args:
        history: 指标历史字典，key 为指标名，value 为按 epoch 记录的数值列表。
        output_path: 曲线图片输出路径，通常保存到 outputs 目录。
        title: 图表标题，用于区分不同训练任务或指标集合。
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for key, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.6, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.ylim(0.0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
