from pathlib import Path
from typing import Callable, Iterable

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def run_image_or_video(source: str, output: str, infer_frame: Callable, logger, progress_interval: int = 50) -> None:
    """执行一个完整流程步骤，通常包含训练、验证、推理或外部框架调用。
    
    Args:
        source: 输入图片、视频、目录或数据源路径，由推理入口传入。
        output: 推理结果输出路径，可以是图片、视频、文本或 JSON 文件。
        infer_frame: infer_frame 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        logger: logger 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        progress_interval: progress_interval 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
