#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-wildlife" "3.10"
install_base_deps
python -m pip install PytorchWildlife || true
echo "[INFO] Pytorch-Wildlife 环境基础依赖完成。若失败，请按 Microsoft CameraTraps/Pytorch-Wildlife 官方文档安装。"
