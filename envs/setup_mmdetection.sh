#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-mmdet" "3.10"
install_base_deps
python -m pip install -U openmim
mim install mmengine mmcv mmdet
echo "[INFO] MMDetection 环境完成：conda activate animal-det-mmdet"
