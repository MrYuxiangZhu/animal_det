#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-tiny_detector}"
CONFIG="${2:-configs/default.yaml}"

case "${MODEL}" in
  tiny_detector)
    python -m src.trainers.tiny_detector --config "${CONFIG}"
    ;;
  clip)
    python -m src.trainers.clip --config "${CONFIG}"
    ;;
  grounding_dino)
    python -m src.trainers.grounding_dino --config "${CONFIG}"
    ;;
  yolov5)
    python -m src.trainers.yolov5 --config "${CONFIG}"
    ;;
  timm)
    python -m src.trainers.timm --config "${CONFIG}"
    ;;
  openclip)
    python -m src.trainers.openclip --config "${CONFIG}"
    ;;
  mmdetection)
    python -m src.trainers.mmdetection --config "${CONFIG}"
    ;;
  detectron2)
    python -m src.trainers.detectron2 --config "${CONFIG}"
    ;;
  *)
    echo "[ERROR] 未知模型: ${MODEL}"
    echo "可选模型: tiny_detector | clip | grounding_dino | yolov5 | timm | mmdetection | detectron2"
    exit 1
    ;;
esac
