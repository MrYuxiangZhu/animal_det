import argparse
from pathlib import Path

import cv2

from src.utils.config import load_config
from src.utils.industrial import require_package
from src.utils.logger import setup_logger


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    parser = argparse.ArgumentParser(description="Infer with Detectron2")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_detectron2", cfg["project"]["log_dir"])
    require_package("detectron2", "按官方说明安装 detectron2: https://detectron2.readthedocs.io/")
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.model_zoo import get_config_file
    from detectron2.utils.visualizer import Visualizer

    d2_cfg = cfg["detectron2"]
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    checkpoint = cfg["infer"].get("detectron2_checkpoint", d2_cfg["checkpoint"])
    pred_cfg = get_cfg()
    pred_cfg.merge_from_file(get_config_file(d2_cfg["model_zoo_config"]))
    pred_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg["data"]["class_names"])
    pred_cfg.MODEL.WEIGHTS = checkpoint
    pred_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg["infer"]["conf_threshold"]
    predictor = DefaultPredictor(pred_cfg)
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"Detectron2 当前适配器支持图片输入，无法读取: {source}")
    outputs = predictor(image)
    visualizer = Visualizer(image[:, :, ::-1])
    result = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, result)
    logger.info("Detectron2 推理完成: %s", output)


if __name__ == "__main__":
    main()
