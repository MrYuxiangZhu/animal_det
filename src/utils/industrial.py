import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def require_package(import_name: str, install_hint: str) -> None:
    """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
    
    Args:
        import_name: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        install_hint: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"缺少依赖包 {import_name}。请先安装：{install_hint}")


def run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        cmd: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        cwd: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    process = subprocess.run(cmd, cwd=cwd, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def write_yolo_data_yaml(path: str, data_root: str, train_images: str, val_images: str, class_names: List[str]) -> None:
    """将运行结果、配置或中间产物写入磁盘，便于复用和排查问题。
    
    Args:
        path: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        data_root: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        train_images: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        val_images: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": data_root,
        "train": train_images,
        "val": val_images,
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def python_executable() -> str:
    """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    return sys.executable
