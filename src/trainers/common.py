import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置全局运行状态，保证实验可复现或流程可控。
    
    Args:
        seed: seed 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    """根据配置选择运行策略或设备。
    
    Args:
        name: 任务或 logger 名称，用于生成日志文件名和运行子目录名。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
