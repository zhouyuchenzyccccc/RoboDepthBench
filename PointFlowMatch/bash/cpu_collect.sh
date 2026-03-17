#!/usr/bin/env bash
set -euo pipefail

# ========= 参数 =========
TASK_NAME="${1:-unplug_charger}"
DATA_BASE="${2:-/inspire/qb-ilm/project/wuliqifa/public/zyc/dataset/pfp_state_recon}"
CONDA_ENV="${CONDA_ENV:-rdbench}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/root/CoppeliaSim}"

TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
VALID_EPISODES="${VALID_EPISODES:-10}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"

# ========= 环境 =========
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

CURRENT_USER="${USER:-$(id -un 2>/dev/null || whoami)}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-${CURRENT_USER}}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

# 关键：软件渲染
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export QT_X11_NO_MITSHM=1
unset QT_DEBUG_PLUGINS

# ========= 依赖检查 =========
if ! command -v Xvfb >/dev/null 2>&1; then
  echo "[Error] Xvfb not found. Please install: apt-get update && apt-get install -y xvfb"
  exit 1
fi

if ! command -v glxinfo >/dev/null 2>&1; then
  echo "[Warn] glxinfo not found. Install mesa-utils if you want renderer diagnostics."
fi

# ========= 启动 Xvfb =========
echo "[Info] Starting Xvfb on ${XVFB_DISPLAY}"
LOCK_FILE="/tmp/.X${XVFB_DISPLAY#:}-lock"
rm -f "${LOCK_FILE}"
pkill -f "Xvfb ${XVFB_DISPLAY}" || true

Xvfb "${XVFB_DISPLAY}" -screen 0 1280x1024x24 +extension GLX +extension RANDR +extension RENDER -noreset &
XVFB_PID=$!
trap 'kill ${XVFB_PID} >/dev/null 2>&1 || true' EXIT
sleep 2

export DISPLAY="${XVFB_DISPLAY}"

if command -v glxinfo >/dev/null 2>&1; then
  echo "[Info] OpenGL renderer (expected llvmpipe/swrast in software mode):"
  glxinfo -B | egrep "OpenGL vendor|OpenGL renderer|OpenGL core profile version" || true
fi

# ========= 采集 =========
TRAIN_OUT="${DATA_BASE}/${TASK_NAME}/train"
VALID_OUT="${DATA_BASE}/${TASK_NAME}/valid"

echo "[Collect] train -> ${TRAIN_OUT}"
python scripts/collect_demos.py \
  --config-name=collect_demos_train \
  save_data=True \
  env_config.vis=False \
  env_config.headless=True \
  env_config.task_name="${TASK_NAME}" \
  num_episodes="${TRAIN_EPISODES}" \
  output_data_dir="${TRAIN_OUT}"

echo "[Collect] valid -> ${VALID_OUT}"
python scripts/collect_demos.py \
  --config-name=collect_demos_valid \
  save_data=True \
  env_config.vis=False \
  env_config.headless=True \
  env_config.task_name="${TASK_NAME}" \
  num_episodes="${VALID_EPISODES}" \
  output_data_dir="${VALID_OUT}"

echo "[Done] Data saved:"
echo "  ${TRAIN_OUT}"
echo "  ${VALID_OUT}"