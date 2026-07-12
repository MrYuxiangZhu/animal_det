#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-tiny_detector}"

case "${MODEL}" in
  tiny_detector)
    bash envs/setup_tiny_detector.sh
    ;;
  clip|grounding_dino)
    bash envs/setup_clip.sh
    ;;
  yolov5)
    bash envs/setup_yolov5.sh
    ;;
  timm)
    bash envs/setup_timm.sh
    ;;
  mmdetection)
    bash envs/setup_mmdetection.sh
    ;;
  detectron2)
    bash envs/setup_detectron2.sh
    ;;
  openclip)
    bash envs/setup_openclip.sh
    ;;
  superanimal)
    bash envs/setup_superanimal.sh
    ;;
  pytorch_wildlife)
    bash envs/setup_pytorch_wildlife.sh
    ;;
  birder)
    bash envs/setup_birder.sh
    ;;
  all)
    bash envs/setup_tiny_detector.sh
    bash envs/setup_clip.sh
    bash envs/setup_yolov5.sh
    bash envs/setup_timm.sh
    bash envs/setup_mmdetection.sh
    bash envs/setup_detectron2.sh
    bash envs/setup_openclip.sh
    bash envs/setup_superanimal.sh
    bash envs/setup_pytorch_wildlife.sh
    bash envs/setup_birder.sh
    ;;
  *)
    echo "[ERROR] 未知模型环境: ${MODEL}"
    echo "可选: tiny_detector | clip | grounding_dino | yolov5 | timm | mmdetection | detectron2 | openclip | superanimal | pytorch_wildlife | birder | all"
    exit 1
    ;;
esac
