#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:-link}"
COCO_ROOT="${2:-data/coco}"
OUTPUT_ROOT="${3:-data/coco_animals}"

DEFAULT_CLASSES="dog,horse,elephant,cat,cow,sheep"
COCO_10_ANIMAL_CLASSES="bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe"

case "${MODE}" in
  link)
    python3 -m src.data.prepare_coco_animals \
      --coco-root "${COCO_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --classes "${DEFAULT_CLASSES}" \
      --overwrite
    ;;
  copy)
    python3 -m src.data.prepare_coco_animals \
      --coco-root "${COCO_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --classes "${DEFAULT_CLASSES}" \
      --copy-images \
      --overwrite
    ;;
  coco10|10cls|all)
    if [[ "${3:-}" == "" ]]; then
      OUTPUT_ROOT="data/coco_animals_10cls"
    fi
    python3 -m src.data.prepare_coco_animals \
      --coco-root "${COCO_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --classes "${COCO_10_ANIMAL_CLASSES}" \
      --overwrite
    ;;
  coco10-copy|10cls-copy|all-copy)
    if [[ "${3:-}" == "" ]]; then
      OUTPUT_ROOT="data/coco_animals_10cls"
    fi
    python3 -m src.data.prepare_coco_animals \
      --coco-root "${COCO_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --classes "${COCO_10_ANIMAL_CLASSES}" \
      --copy-images \
      --overwrite
    ;;
  *)
    echo "[ERROR] 未知模式: ${MODE}"
    echo "用法: bash script/prepare_coco_animals.sh [link|copy|coco10|coco10-copy] [coco_root] [output_root]"
    echo "示例:"
    echo "  bash script/prepare_coco_animals.sh"
    echo "  bash script/prepare_coco_animals.sh link data/coco data/coco_animals"
    echo "  bash script/prepare_coco_animals.sh copy data/coco data/coco_animals"
    echo "  bash script/prepare_coco_animals.sh coco10 data/coco data/coco_animals_10cls"
    echo "  bash script/prepare_coco_animals.sh coco10-copy data/coco data/coco_animals_10cls"
    exit 1
    ;;
esac

# 1. 默认模式：软链接，6 个共有动物类
# bash script/prepare_coco_animals.sh link data/coco data/coco_animals
# 2. 复制模式：复制图片，6 个共有动物类
# bash script/prepare_coco_animals.sh copy data/coco data/coco_animals
# 3. COCO-10 模式：软链接，10 个共有动物类
# bash script/prepare_coco_animals.sh coco10 data/coco data/coco_animals_10cls
# 4. COCO-10 复制模式：复制图片，10 个共有动物类
# bash script/prepare_coco_animals.sh coco10-copy data/coco data/coco_animals_10cls
