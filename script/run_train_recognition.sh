#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-timm}"
CONFIG="${2:-configs/default.yaml}"

case "${MODEL}" in
  timm|clip|openclip)
    bash script/run_train.sh "${MODEL}" "${CONFIG}"
    ;;
  *)
    echo "[ERROR] ${MODEL} 不是识别模型"
    echo "识别模型可选: timm | clip | openclip"
    echo "说明: timm 默认使用 ViT Transformer；clip 使用 Transformer 文本编码器"
    exit 1
    ;;
esac
