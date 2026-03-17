#!/usr/bin/env bash
set -euo pipefail

# ========= 参数 =========
TASK_NAME="${1:-unplug_charger}"
DATA_BASE="${2:-/inspire/qb-ilm/project/wuliqifa/public/zyc/dataset/pfp_state_recon}"
DISPLAY_ID="${3:-:0}"

CONDA_ENV="${CONDA_ENV:-rdbench}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/root/CoppeliaSim}"

TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
VALID_EPISODES="${VALID_EPISODES:-10}"

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

# 关键：GPU 渲染，不要强制软件栈
export QT_QPA_PLATFORM=xcb
unset LIBGL_ALWAYS_SOFTWARE
unset MESA_GL_VERSION_OVERRIDE
unset MESA_GLSL_VERSION_OVERRIDE
unset QT_DEBUG_PLUGINS

# 多 GPU 机器常见设置
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
export QT_X11_NO_MITSHM=1

export DISPLAY="${DISPLAY_ID}"

# ========= 依赖与可用性检查 =========
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[Error] nvidia-smi not found. NVIDIA runtime is unavailable."
  exit 1
fi

if ! command -v glxinfo >/dev/null 2>&1; then
  echo "[Error] glxinfo not found. Install mesa-utils first."
  exit 1
fi

echo "[Check] nvidia-smi"
nvidia-smi >/dev/null

echo "[Check] OpenGL renderer on DISPLAY=${DISPLAY}"
GL_INFO="$(glxinfo -B 2>/dev/null || true)"
if [ -z "${GL_INFO}" ]; then
  echo "[Error] Unable to query OpenGL info from DISPLAY=${DISPLAY}."
  echo "        Ensure an NVIDIA-capable X display exists and is accessible in this container."
  exit 1
fi
echo "${GL_INFO}" | egrep "OpenGL vendor|OpenGL renderer|OpenGL core profile version" || true

if ! echo "${GL_INFO}" | grep -qi "OpenGL renderer string:.*NVIDIA"; then
  echo "[Error] Current renderer is not NVIDIA GPU (likely llvmpipe/swrast)."
  echo "        Please provide a GPU-backed X display, e.g. :0 or :1."
  exit 1
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
