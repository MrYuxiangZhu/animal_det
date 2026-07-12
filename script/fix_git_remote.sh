#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REMOTE_NAME="${1:-origin}"
MODE="${2:-github}"
REPO_PATH="${3:-MrYuxiangZhu/animal_det.git}"

OFFICIAL_URL="https://github.com/${REPO_PATH}"
SSH_URL="git@github.com:${REPO_PATH}"
FASTGIT_URL="https://github.com/${REPO_PATH}"

choose_url() {
  case "${MODE}" in
    github)
      echo "${OFFICIAL_URL}"
      ;;
    ssh)
      echo "${SSH_URL}"
      ;;
    https)
      echo "${OFFICIAL_URL}"
      ;;
    *)
      echo "${MODE}"
      ;;
  esac
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] 当前目录不是 Git 仓库: ${PROJECT_ROOT}"
  exit 1
fi

TARGET_URL="$(choose_url)"
CURRENT_URL="$(git remote get-url "${REMOTE_NAME}" 2>/dev/null || true)"

if [[ -z "${CURRENT_URL}" ]]; then
  echo "[INFO] remote ${REMOTE_NAME} 不存在，新增为: ${TARGET_URL}"
  git remote add "${REMOTE_NAME}" "${TARGET_URL}"
else
  echo "[INFO] 当前 ${REMOTE_NAME}: ${CURRENT_URL}"
  if [[ "${CURRENT_URL}" == *"mirror.ghproxy.com"* || "${CURRENT_URL}" == *"ghproxy"* || "${CURRENT_URL}" != "${TARGET_URL}" ]]; then
    echo "[INFO] 更新 ${REMOTE_NAME} 为: ${TARGET_URL}"
    git remote set-url "${REMOTE_NAME}" "${TARGET_URL}"
  else
    echo "[INFO] remote 已经是目标地址，无需修改。"
  fi
fi

echo "[INFO] 当前 remote 配置："
git remote -v

echo "[INFO] 测试远端连接：git ls-remote ${REMOTE_NAME} HEAD"
if git ls-remote "${REMOTE_NAME}" HEAD >/dev/null 2>&1; then
  echo "[OK] 远端连接成功。现在可以执行：git pull"
else
  echo "[WARN] 远端连接仍失败。可能原因："
  echo "  1. 当前网络无法访问 github.com；"
  echo "  2. DNS/代理配置异常；"
  echo "  3. 仓库权限或认证问题；"
  echo "  4. 如果使用 SSH，请确认 ssh key 已添加到 GitHub。"
  echo ""
  echo "你可以尝试："
  echo "  bash script/fix_git_remote.sh origin ssh"
  echo "  git config --global --unset http.proxy || true"
  echo "  git config --global --unset https.proxy || true"
  exit 2
fi
