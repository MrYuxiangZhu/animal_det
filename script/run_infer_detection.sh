#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-tiny_detector}"
CONFIG="${2:-configs/default.yaml}"
SOURCE="${3:-}"
OUTPUT="${4:-}"
TEXT="${5:-}"

case "${MODEL}" in
  tiny_detector|tiny_detector_pro|grounding_dino|yolov5|mmdetection|detectron2)
    bash script/run_infer.sh "${MODEL}" "${CONFIG}" "${SOURCE}" "${OUTPUT}" "${TEXT}"
    ;;
  *)
    echo "[ERROR] ${MODEL} 不是检测模型"
    echo "检测模型可选: tiny_detector | tiny_detector_pro | grounding_dino | yolov5 | mmdetection | detectron2"
    exit 1
    ;;
esac
