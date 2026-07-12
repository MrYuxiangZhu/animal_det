#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-timm}"
CONFIG="${2:-configs/default.yaml}"
SOURCE="${3:-}"
OUTPUT="${4:-}"
TEXT="${5:-}"

case "${MODEL}" in
  timm|clip|openclip|birder)
    bash script/run_infer.sh "${MODEL}" "${CONFIG}" "${SOURCE}" "${OUTPUT}" "${TEXT}"
    ;;
  *)
    echo "[ERROR] ${MODEL} 不是识别模型"
    echo "识别模型可选: timm | clip | openclip | birder"
    exit 1
    ;;
esac
