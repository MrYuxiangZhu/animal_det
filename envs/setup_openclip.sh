#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-openclip" "3.10"
install_base_deps
python -m pip install open_clip_torch
echo "[INFO] OpenCLIP 环境完成：conda activate animal-det-openclip"
