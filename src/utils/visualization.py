from pathlib import Path
from typing import Dict, List

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
