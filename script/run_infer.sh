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

ARGS=(--config "${CONFIG}")
if [[ -n "${SOURCE}" ]]; then
  ARGS+=(--source "${SOURCE}")
fi
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

case "${MODEL}" in
  tiny_detector)
    python -m src.inferencers.tiny_detector "${ARGS[@]}"
    ;;
  tiny_detector_pro)
    python -m src.inferencers.tiny_detector_pro "${ARGS[@]}"
    ;;
  clip)
    if [[ -n "${TEXT}" ]]; then
      ARGS+=(--text "${TEXT}")
    fi
    python -m src.inferencers.clip "${ARGS[@]}"
    ;;
  grounding_dino)
    if [[ -n "${TEXT}" ]]; then
      ARGS+=(--text "${TEXT}")
    fi
    python -m src.inferencers.grounding_dino "${ARGS[@]}"
    ;;
  yolov5)
    python -m src.inferencers.yolov5 "${ARGS[@]}"
    ;;
  timm)
    python -m src.inferencers.timm "${ARGS[@]}"
    ;;
  mmdetection)
    python -m src.inferencers.mmdetection "${ARGS[@]}"
    ;;
  detectron2)
    python -m src.inferencers.detectron2 "${ARGS[@]}"
    ;;
  openclip)
    if [[ -n "${TEXT}" ]]; then
      ARGS+=(--text "${TEXT}")
    fi
    python -m src.inferencers.openclip "${ARGS[@]}"
    ;;
  superanimal)
    python -m src.inferencers.superanimal "${ARGS[@]}"
    ;;
  pytorch_wildlife)
    python -m src.inferencers.pytorch_wildlife "${ARGS[@]}"
    ;;
  birder)
    python -m src.inferencers.birder "${ARGS[@]}"
    ;;
  *)
    echo "[ERROR] 未知模型: ${MODEL}"
    echo "可选模型: tiny_detector | tiny_detector_pro | clip | grounding_dino | yolov5 | timm | mmdetection | detectron2 | openclip | superanimal | pytorch_wildlife | birder"
    exit 1
    ;;
esac
