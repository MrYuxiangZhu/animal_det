#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-tiny" "3.10"
install_base_deps
echo "[INFO] tiny_detector 环境完成：conda activate animal-det-tiny"
