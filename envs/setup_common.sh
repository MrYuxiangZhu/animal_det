#!/usr/bin/env bash
set -euo pipefail

create_or_update_env() {
  local env_name="$1"
  local python_version="${2:-3.10}"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda 未安装或未加入 PATH。请先安装 Miniconda/Anaconda。"
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[INFO] conda 环境 ${env_name} 已存在。"
    conda activate "${env_name}"
  else
    echo "[INFO] 创建 conda 环境 ${env_name}，Python ${python_version}。"
    conda create -y -n "${env_name}" "python=${python_version}" pip
    conda activate "${env_name}"
  fi
  python -m pip install --upgrade pip
}

install_base_deps() {
  python -m pip install torch torchvision numpy pyyaml pillow matplotlib tqdm opencv-python
}
