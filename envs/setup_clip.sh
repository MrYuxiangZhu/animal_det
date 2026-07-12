#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-clip" "3.10"
install_base_deps
echo "[INFO] clip / grounding_dino 学习版环境完成：conda activate animal-det-clip"
