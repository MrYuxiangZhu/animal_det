#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-tiny_detector}"
CONFIG="${2:-configs/coco_animals_10cls.yaml}"

case "${MODEL}" in
  tiny_detector|grounding_dino|yolov5|mmdetection|detectron2)
    bash script/run_train.sh "${MODEL}" "${CONFIG}"
    ;;
  *)
    echo "[ERROR] ${MODEL} 不是检测模型"
    echo "检测模型可选: tiny_detector | grounding_dino | yolov5 | mmdetection | detectron2"
    exit 1
    ;;
esac

# bash script/run_train_detection.sh tiny_detector configs/coco_animals_10cls.yaml
# bash script/run_train_detection.sh grounding_dino configs/coco_animals_10cls.yaml