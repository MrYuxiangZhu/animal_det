#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-detectron2" "3.10"
install_base_deps
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
echo "[INFO] Detectron2 环境完成：conda activate animal-det-detectron2"
