#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-birder" "3.10"
install_base_deps
python -m pip install birder || true
echo "[INFO] Birder 环境基础依赖完成。若 pip 包不可用，请 clone birder 官方仓库并按其文档安装。"
