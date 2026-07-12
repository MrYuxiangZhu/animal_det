#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-superanimal" "3.10"
install_base_deps
python -m pip install 'deeplabcut[gui]'
echo "[INFO] SuperAnimal/DeepLabCut 环境完成：conda activate animal-det-superanimal"
