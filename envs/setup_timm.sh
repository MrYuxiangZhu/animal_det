#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-timm" "3.10"
install_base_deps
python -m pip install timm
echo "[INFO] timm 环境完成：conda activate animal-det-timm"
