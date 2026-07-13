#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:-coco10}"
OUTPUT_DIR="${2:-outputs/dataset_stats}"

COCO10_CLASSES="bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe"
COCO6_CLASSES="dog,horse,elephant,cat,cow,sheep"
ANIMALS10_CLASSES="dog,horse,elephant,butterfly,chicken,cat,cow,sheep,spider,squirrel"

case "${MODE}" in
  coco10)
    python3 -m src.data.stat_dataset_distribution \
      --type recognition \
      --root data/coco_animals_10cls/recognition \
      --classes "${COCO10_CLASSES}" \
      --output-dir "${OUTPUT_DIR}" \
      --name coco_animals_10cls
    python3 -m src.data.stat_dataset_distribution \
      --type detection \
      --root data/coco_animals_10cls/detection \
      --classes "${COCO10_CLASSES}" \
      --output-dir "${OUTPUT_DIR}" \
      --name coco_animals_10cls
    ;;
  coco6)
    python3 -m src.data.stat_dataset_distribution \
      --type recognition \
      --root data/coco_animals/recognition \
      --classes "${COCO6_CLASSES}" \
      --output-dir "${OUTPUT_DIR}" \
      --name coco_animals
    python3 -m src.data.stat_dataset_distribution \
      --type detection \
      --root data/coco_animals/detection \
      --classes "${COCO6_CLASSES}" \
      --output-dir "${OUTPUT_DIR}" \
      --name coco_animals
    ;;
  animals10)
    python3 -m src.data.stat_dataset_distribution \
      --type recognition \
      --root data/animals10/recognition \
      --classes "${ANIMALS10_CLASSES}" \
      --output-dir "${OUTPUT_DIR}" \
      --name animals10
    ;;
  *)
    echo "[ERROR] 未知模式: ${MODE}"
    echo "用法: bash script/stat_dataset_distribution.sh [coco10|coco6|animals10] [output_dir]"
    echo "示例:"
    echo "  bash script/stat_dataset_distribution.sh coco10"
    echo "  bash script/stat_dataset_distribution.sh coco6 outputs/dataset_stats"
    echo "  bash script/stat_dataset_distribution.sh animals10"
    exit 1
    ;;
esac
