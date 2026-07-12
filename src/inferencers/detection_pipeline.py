from pathlib import Path
from typing import Callable, Iterable

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def run_image_or_video(source: str, output: str, infer_frame: Callable, logger, progress_interval: int = 50) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        output: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        infer_frame: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        logger: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        progress_interval: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(source).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"无法读取图片: {source}")
        cv2.imwrite(output, infer_frame(frame))
        logger.info("图片推理完成: %s", output)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(infer_frame(frame))
        count += 1
        if count % progress_interval == 0:
            logger.info("已处理 %d 帧", count)
    cap.release()
    writer.release()
    logger.info("视频推理完成: %s，总帧数: %d", output, count)
