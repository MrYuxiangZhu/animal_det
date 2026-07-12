import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def require_package(import_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"缺少依赖包 {import_name}。请先安装：{install_hint}")


def run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    process = subprocess.run(cmd, cwd=cwd, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def write_yolo_data_yaml(path: str, data_root: str, train_images: str, val_images: str, class_names: List[str]) -> None:
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
    return sys.executable
