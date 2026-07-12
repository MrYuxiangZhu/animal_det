import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.industrial import require_package
from src.utils.logger import setup_logger


def main() -> None:
    """命令行入口函数，解析参数、加载配置并调度对应的训练或推理流程。
    
    Returns:
        该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
    """
    parser = argparse.ArgumentParser(description="Train animal detector with Detectron2")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("train_detectron2", cfg["project"]["log_dir"])
    require_package("detectron2", "按官方说明安装 detectron2: https://detectron2.readthedocs.io/")
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.model_zoo import get_checkpoint_url, get_config_file

    d2_cfg = cfg["detectron2"]
    register_coco_instances("animal_train", {}, d2_cfg["train_json"], d2_cfg["train_images"])
    register_coco_instances("animal_val", {}, d2_cfg["val_json"], d2_cfg["val_images"])
    train_cfg = get_cfg()
    train_cfg.merge_from_file(get_config_file(d2_cfg["model_zoo_config"]))
    train_cfg.DATASETS.TRAIN = ("animal_train",)
    train_cfg.DATASETS.TEST = ("animal_val",)
    train_cfg.DATALOADER.NUM_WORKERS = cfg["data"]["num_workers"]
    train_cfg.MODEL.WEIGHTS = get_checkpoint_url(d2_cfg["model_zoo_config"]) if d2_cfg["pretrained"] else ""
    train_cfg.SOLVER.IMS_PER_BATCH = d2_cfg["batch_size"]
    train_cfg.SOLVER.BASE_LR = d2_cfg["learning_rate"]
    train_cfg.SOLVER.MAX_ITER = d2_cfg["max_iter"]
    train_cfg.SOLVER.STEPS = d2_cfg["steps"]
    train_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg["data"]["class_names"])
    train_cfg.OUTPUT_DIR = str(Path(cfg["project"]["output_dir"]) / "detectron2" / d2_cfg["run_name"])
    Path(train_cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer = DefaultTrainer(train_cfg)
    trainer.resume_or_load(resume=False)
    logger.info("启动 Detectron2 训练，输出目录: %s", train_cfg.OUTPUT_DIR)
    trainer.train()


if __name__ == "__main__":
    main()
