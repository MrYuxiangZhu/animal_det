#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_common.sh"
create_or_update_env "animal-det-yolov5" "3.10"
install_base_deps
python -m pip install pandas seaborn requests scipy psutil thop gitpython
mkdir -p third_party
if [[ ! -d third_party/yolov5 ]]; then
  git clone https://github.com/ultralytics/yolov5 third_party/yolov5
fi
python -m pip install -r third_party/yolov5/requirements.txt
echo "[INFO] YOLOv5 环境完成：conda activate animal-det-yolov5"
