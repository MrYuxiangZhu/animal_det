import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.industrial import require_package
from src.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with MMDetection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("infer_mmdetection", cfg["project"]["log_dir"])
    require_package("mmdet", "pip install -U openmim && mim install mmengine mmcv mmdet")
    from mmdet.apis import DetInferencer

    mm_cfg = cfg["mmdetection"]
    source = args.source or cfg["infer"]["source"]
    output = args.output or cfg["infer"]["output"]
    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg["infer"].get("mmdet_checkpoint", mm_cfg["checkpoint"])
    inferencer = DetInferencer(model=mm_cfg["config"], weights=checkpoint, device=mm_cfg["device"])
    inferencer(source, out_dir=str(out_dir), pred_score_thr=cfg["infer"]["conf_threshold"])
    logger.info("MMDetection 推理完成，输出目录: %s", out_dir)


if __name__ == "__main__":
    main()
