import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def require_package(import_name: str, install_hint: str) -> None:
    """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
    
    Args:
        import_name: Python import 包名，用于检查依赖是否安装。
        install_hint: 依赖缺失时展示给用户的安装命令或官方文档提示。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"缺少依赖包 {import_name}。请先安装：{install_hint}")


def run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        cmd: 将要执行的外部命令参数列表，会传给 subprocess.run。
        cwd: 外部命令执行目录。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    process = subprocess.run(cmd, cwd=cwd, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def write_yolo_data_yaml(path: str, data_root: str, train_images: str, val_images: str, class_names: List[str]) -> None:
    """将运行结果、配置或中间产物写入磁盘，便于复用和排查问题。
    
    Args:
        path: 文件或目录路径，含义由调用处决定。
        data_root: 分类数据集根目录，目录下应包含 train/val 以及按类别命名的子文件夹。
        train_images: YOLOv5 data.yaml 中的训练图片相对路径。
        val_images: YOLOv5 data.yaml 中的验证图片相对路径。
        class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    return sys.executable
