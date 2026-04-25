#!/usr/bin/env bash
set -euo pipefail

# 全封闭 py-spy 复现脚本：
# 1. 创建独立 Python 3.11 虚拟环境；
# 2. 执行 editable install + constraints；
# 3. 用 py-spy record 直接启动第一次 `python -m dayu.cli --help`；
# 4. 记录 profile、stdout/stderr、耗时、pip freeze、系统信息；
# 5. 额外补一轮 warm importtime，便于和首次 profile 对照。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${REPO_ROOT}/workspace/tmp/diagnostics/full-cold-repro-pyspy-${TIMESTAMP}"
VENV_DIR="${RUN_DIR}/.venv"
SUMMARY_FILE="${RUN_DIR}/summary.log"
INSTALL_LOG="${RUN_DIR}/install.log"
INSTALL_TIME="${RUN_DIR}/install.time"
FREEZE_FILE="${RUN_DIR}/pip-freeze.txt"
PYTHON_INFO_FILE="${RUN_DIR}/python-info.txt"
SYSTEM_INFO_FILE="${RUN_DIR}/system-info.txt"
FIRST_RUN_STDOUT="${RUN_DIR}/first-run.stdout"
FIRST_RUN_STDERR="${RUN_DIR}/first-run.stderr"
FIRST_RUN_TIME="${RUN_DIR}/first-run.time"
FIRST_RUN_PROFILE="${RUN_DIR}/first-run.pyspy.raw"
FIRST_RUN_WRAPPER="${RUN_DIR}/first-run-wrapper.py"
SECOND_IMPORT_STDOUT="${RUN_DIR}/second-importtime.stdout"
SECOND_IMPORT_ERR="${RUN_DIR}/second-importtime.stderr"
SECOND_IMPORT_TIME="${RUN_DIR}/second-importtime.time"
PHASE="初始化"

mkdir -p "${RUN_DIR}"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
else
  echo "未找到 python3.11，请先安装 Python 3.11 再运行。" >&2
  exit 1
fi

if command -v py-spy >/dev/null 2>&1; then
  PYSPY_BIN="$(command -v py-spy)"
elif [[ -x "/opt/homebrew/bin/py-spy" ]]; then
  PYSPY_BIN="/opt/homebrew/bin/py-spy"
else
  echo "未找到 py-spy，请先安装 py-spy 再运行。" >&2
  exit 1
fi

log() {
  printf '%s\n' "$*" | tee -a "${SUMMARY_FILE}"
}

on_error() {
  local exit_code="$1"
  log "脚本在阶段【${PHASE}】失败，退出码=${exit_code}"
  log "诊断目录保留在：${RUN_DIR}"
  exit "${exit_code}"
}

trap 'on_error $?' ERR

log "输出目录: ${RUN_DIR}"
log "仓库目录: ${REPO_ROOT}"
log "使用 Python: ${PYTHON_BIN}"
log "使用 py-spy: ${PYSPY_BIN}"
log "诊断虚拟环境目录: ${VENV_DIR}"

cd "${REPO_ROOT}"

{
  echo "repo_root=${REPO_ROOT}"
  echo "timestamp=${TIMESTAMP}"
  echo "python_bin=${PYTHON_BIN}"
  echo "pyspy_bin=${PYSPY_BIN}"
  echo "git_head=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "pwd=${PWD}"
} >"${SYSTEM_INFO_FILE}"

{
  sw_vers || true
  echo "---"
  uname -a || true
  echo "---"
  df -h "${REPO_ROOT}" || true
  echo "---"
  mount | grep "$(df "${REPO_ROOT}" | tail -n 1 | awk '{print $1}')" || true
} >>"${SYSTEM_INFO_FILE}" 2>&1

PHASE="创建独立虚拟环境"
log "创建独立虚拟环境..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

{
  "${VENV_DIR}/bin/python" --version
  "${VENV_DIR}/bin/pip" --version
  "${PYSPY_BIN}" --version
} >"${PYTHON_INFO_FILE}" 2>&1

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_PYTHON_VERSION_WARNING=1

PHASE="sudo 鉴权"
log "先执行 sudo -v，后续仅在 py-spy 阶段提权..."
sudo -v

PHASE="安装依赖"
log "开始安装依赖（editable install + constraints）..."
/usr/bin/time -lp -o "${INSTALL_TIME}" \
  "${VENV_DIR}/bin/pip" install \
  -e ".[test,dev,browser,web]" \
  -c "${REPO_ROOT}/constraints/lock-macos-arm64-py311.txt" \
  2>&1 | tee "${INSTALL_LOG}"

PHASE="记录 pip freeze"
log "安装完成，记录 pip freeze..."
"${VENV_DIR}/bin/pip" freeze >"${FREEZE_FILE}"

cat >"${FIRST_RUN_WRAPPER}" <<'PY'
"""为 py-spy 预留附着窗口的首次运行包装器。"""

from __future__ import annotations

import runpy
import sys
import time


def main() -> int:
    """等待短暂窗口后启动真正的 dayu.cli 模块。

    Args:
        无。

    Returns:
        进程退出码 0。

    Raises:
        SystemExit: 由目标模块自行触发时继续向上传播。
    """

    time.sleep(3.0)
    sys.argv = ["python -m dayu.cli", "--help"]
    runpy.run_module("dayu.cli", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

PHASE="首次 py-spy 运行"
log "开始第一次执行：sudo py-spy record -- first-run-wrapper.py ..."
/usr/bin/time -lp -o "${FIRST_RUN_TIME}" \
  sudo "${PYSPY_BIN}" record \
  --format raw \
  --full-filenames \
  --threads \
  --output "${FIRST_RUN_PROFILE}" \
  -- \
  "${VENV_DIR}/bin/python" "${FIRST_RUN_WRAPPER}" \
  >"${FIRST_RUN_STDOUT}" 2>"${FIRST_RUN_STDERR}"

PHASE="第二次 importtime"
log "开始第二次执行：python -X importtime -m dayu.cli --help..."
/usr/bin/time -lp -o "${SECOND_IMPORT_TIME}" \
  "${VENV_DIR}/bin/python" -X importtime -m dayu.cli --help \
  >"${SECOND_IMPORT_STDOUT}" 2>"${SECOND_IMPORT_ERR}"

log ""
log "关键信息："
log "  安装耗时文件: ${INSTALL_TIME}"
log "  首次 py-spy 运行耗时文件: ${FIRST_RUN_TIME}"
log "  首次 py-spy profile: ${FIRST_RUN_PROFILE}"
log "  首次运行 stdout: ${FIRST_RUN_STDOUT}"
log "  首次运行 stderr: ${FIRST_RUN_STDERR}"
log "  第二次 importtime 耗时文件: ${SECOND_IMPORT_TIME}"
log "  第二次 importtime stderr: ${SECOND_IMPORT_ERR}"
log "  pip freeze: ${FREEZE_FILE}"
log ""
log "请把整个目录回传给我：${RUN_DIR}"
