#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-tiny" "3.10"
install_base_deps
# tiny_detector_pro 模型依赖 timm 预训练 backbone
python -m pip install timm
echo "[INFO] tiny_detector 环境完成：conda activate animal-det-tiny"
