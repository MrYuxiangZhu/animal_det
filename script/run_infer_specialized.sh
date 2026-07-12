#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-openclip}"
CONFIG="${2:-configs/default.yaml}"
SOURCE="${3:-}"
OUTPUT="${4:-}"
TEXT="${5:-}"

case "${MODEL}" in
  openclip|superanimal|pytorch_wildlife|birder)
    bash script/run_infer.sh "${MODEL}" "${CONFIG}" "${SOURCE}" "${OUTPUT}" "${TEXT}"
    ;;
  *)
    echo "[ERROR] ${MODEL} 不是专用动物 Transformer/行为分析模型"
    echo "可选: openclip | superanimal | pytorch_wildlife | birder"
    exit 1
    ;;
esac
