import logging
import os
import sys
from datetime import datetime
from pathlib import Path


_RUN_LOG_DIRS = {}


def get_run_log_dir(name: str, log_dir: str = "logs") -> Path:
    """为一次训练或推理运行创建并返回独立日志目录。
    
    Args:
        name: 任务或 logger 名称，用于生成日志文件名和运行子目录名。
        log_dir: 日志根目录；运行时会在其中创建日期时间命名的子目录。
    
    Returns:
        形如 ``logs/YYYYMMDD_HHMMSS_<name>`` 的目录路径。
    """
    key = f"{Path(log_dir).resolve()}::{name}"
    if key in _RUN_LOG_DIRS:
        return _RUN_LOG_DIRS[key]
    env_key = f"ANIMAL_DET_LOG_DIR_{name.upper()}"
    if env_key in os.environ:
        run_dir = Path(os.environ[env_key])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
        run_dir = Path(log_dir) / f"{timestamp}_{safe_name}"
        suffix = 1
        while run_dir.exists():
            run_dir = Path(log_dir) / f"{timestamp}_{safe_name}_{suffix:02d}"
            suffix += 1
        os.environ[env_key] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _RUN_LOG_DIRS[key] = run_dir
    return run_dir


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """创建同时输出到控制台和本次运行日志目录文件的 logger。
    
    Args:
        name: 任务或 logger 名称，用于生成日志文件名和运行子目录名。
        log_dir: 日志根目录；运行时会在其中创建日期时间命名的子目录。
        level: Python logging 日志等级，例如 logging.INFO。
    
    Returns:
        已配置 stream handler 和 file handler 的 ``logging.Logger``。
    """
    run_log_dir = get_run_log_dir(name, log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    target_file = run_log_dir / f"{name}.log"
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == target_file:
            return logger
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(target_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("本次运行日志目录: %s", run_log_dir)
    return logger
