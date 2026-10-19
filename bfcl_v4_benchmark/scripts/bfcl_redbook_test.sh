#!/bin/bash
# 直接指定模型路径进行 bfcl 测试
# 用法:
#   export MODEL_PATH=/your/model/path
#   export REDBOOK_TEST_MODE=prompt   # 或 fc
#   export DT=20250228
#   bash test_bfcl_direct.sh
# 可选:
#   export VLLM_GPU=0,1,...
#   export VLLM_PORT=8399
#   export VLLM_MAX_MODEL_LEN=65536
#   export VLLM_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1"

set -x

# ===== 0) 校验必需变量 =====
: "${MODEL_PATH:?必须 export MODEL_PATH}"
: "${DT:?必须 export DT，例如 20250228}"

REDBOOK_TEST_MODE="${REDBOOK_TEST_MODE:-prompt}"

case "${REDBOOK_TEST_MODE}" in
  prompt|fc)
    ;;
  *)
    echo "[ERROR] REDBOOK_TEST_MODE=$REDBOOK_TEST_MODE，支持: prompt / fc"
    exit 1
    ;;
esac

# ===== 1) 基本路径/默认值 =====
VLLM_PORT="${VLLM_PORT:-8399}"
VLLM_GPU="${VLLM_GPU:-0,1,2,3,4,5,6,7}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1}"

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_LOG_DIR="/tmp/vllm_logs"
BENCH_LOG_ROOT="${BENCH_LOG_ROOT:-${WORKDIR}/outputs}"

mkdir -p "$VLLM_LOG_DIR"

DEBUG_SLEEP_TIME="${DEBUG_SLEEP_TIME:-1200}"

# ===== 1.1) 根据 MODEL_PATH + DT 计算 BFCL 输出目录 =====
MODEL_PATH_CLEAN="${MODEL_PATH%/}"
MODEL_NAME="$(basename "$MODEL_PATH_CLEAN")"
MODEL_FAMILY="$(basename "$(dirname "$MODEL_PATH_CLEAN")")"

BFCL_PROJECT_ROOT="${BENCH_LOG_ROOT}/${MODEL_FAMILY}/${MODEL_NAME}/compass_outputs/bfcl_bench_${DT}"
BFCL_RESULT_DIR="${BFCL_PROJECT_ROOT}/result"
BFCL_SCORE_DIR="${BFCL_PROJECT_ROOT}/score"

mkdir -p "$BFCL_PROJECT_ROOT" "$BFCL_RESULT_DIR" "$BFCL_SCORE_DIR"

export BFCL_PROJECT_ROOT

echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] MODEL_NAME=$MODEL_NAME"
echo "[INFO] MODEL_FAMILY=$MODEL_FAMILY"
echo "[INFO] BFCL_PROJECT_ROOT=$BFCL_PROJECT_ROOT"
echo "[INFO] BFCL_RESULT_DIR=$BFCL_RESULT_DIR"
echo "[INFO] BFCL_SCORE_DIR=$BFCL_SCORE_DIR"
echo "[INFO] VLLM_GPU=$VLLM_GPU  VLLM_PORT=$VLLM_PORT"
echo "[INFO] VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN"
echo "[INFO] VLLM_EXTRA_ARGS=$VLLM_EXTRA_ARGS"
echo "[INFO] REDBOOK_TEST_MODE=$REDBOOK_TEST_MODE"
echo "[INFO] DT=$DT"

# ===== 2) 环境准备 =====
if [ ! -d "/mnt/nlp-ali" ] && [ -n "${DATA_ROOT:-}" ]; then
    mkdir -p /mnt/nlp-ali
    cd /mnt/nlp-ali
    ln -s "${DATA_ROOT}" usr
    ln -s "${DATA_ROOT}" nas
    cd /mnt/nlp-ali/usr
fi

# ===== 2.1) 可选：把 WORKDIR 下的 .env 暴露给 BFCL_PROJECT_ROOT =====
if [ -f "${WORKDIR}/.env" ] && [ ! -e "${BFCL_PROJECT_ROOT}/.env" ]; then
  ln -s "${WORKDIR}/.env" "${BFCL_PROJECT_ROOT}/.env"
fi

# ===== 3) 工具函数 =====
VLLM_PID=""
VLLM_LOG_FILE=""
SERVED_NAME=""
BFCL_MODEL_NAME=""

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && ps -p "$VLLM_PID" > /dev/null 2>&1; then
    echo "[CLEANUP] 终止 vLLM (PID=$VLLM_PID)"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
  fi
}
trap cleanup EXIT

wait_for_debug() {
  local reason="$1"
  echo ""
  echo "================================================================"
  echo "[DEBUG] $reason"
  echo "[DEBUG] 系统将等待 ${DEBUG_SLEEP_TIME} 秒 ($(( DEBUG_SLEEP_TIME / 60 )) 分钟) 供你调试"
  echo "[DEBUG] vLLM 日志文件: $VLLM_LOG_FILE"
  echo "[DEBUG] 查看日志命令: tail -f $VLLM_LOG_FILE"
  echo "================================================================"
  local remaining=$DEBUG_SLEEP_TIME
  while [ $remaining -gt 0 ]; do
    echo "[DEBUG] 剩余: $(( remaining / 60 )) 分 $(( remaining % 60 )) 秒..."
    if [ $remaining -ge 60 ]; then
      sleep 60
      remaining=$(( remaining - 60 ))
    else
      sleep $remaining
      remaining=0
    fi
  done
  echo "[DEBUG] 调试等待结束，脚本退出"
}

start_vllm() {
  local MODEL="$1"
  local suffix
  local suffix_clean
  local timestamp
  local -a extra_args=()

  suffix="$(basename "$(dirname "$MODEL")")_$(basename "$MODEL")"
  suffix_clean="$(echo "$suffix" | tr '/' '_')"

  SERVED_NAME="RedBook_${suffix_clean}_${DT}"

  if [ "${REDBOOK_TEST_MODE}" = "fc" ]; then
    BFCL_MODEL_NAME="${SERVED_NAME}-FC"
  else
    BFCL_MODEL_NAME="${SERVED_NAME}"
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  VLLM_LOG_FILE="${VLLM_LOG_DIR}/vllm_${suffix_clean}_${timestamp}.log"

  if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
  fi

  echo "[INFO] SERVED_NAME=$SERVED_NAME"
  echo "[INFO] BFCL_MODEL_NAME=$BFCL_MODEL_NAME"
  echo "[INFO] VLLM_LOG_FILE=$VLLM_LOG_FILE"
  echo "[INFO] vLLM extra args: ${extra_args[*]}"

  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
  vllm serve "$MODEL" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --tensor-parallel-size 8 \
    --served-model-name "${SERVED_NAME}" \
    "${extra_args[@]}" \
    > "$VLLM_LOG_FILE" 2>&1 &
  VLLM_PID=$!

  echo "[INFO] vLLM PID=$VLLM_PID，等待启动..."

  local MAX_TRIES=600
  local CHECK_INTERVAL=20
  for ((try=0; try<MAX_TRIES; try++)); do
    sleep "${CHECK_INTERVAL}"
    if ! ps -p "$VLLM_PID" > /dev/null 2>&1; then
      echo "[ERROR] vLLM 进程已退出!"
      tail -n 100 "$VLLM_LOG_FILE"
      wait_for_debug "vLLM 进程意外退出"
      exit 1
    fi
    if curl --noproxy localhost,127.0.0.1 -fs "http://localhost:${VLLM_PORT}/health" &>/dev/null; then
      echo "[INFO] ✓ vLLM 就绪 (耗时约 $((try * CHECK_INTERVAL)) 秒)"
      tail -n 30 "$VLLM_LOG_FILE"
      return 0
    fi
    echo "[INFO] 等待 vLLM... (已等 $(((try+1)*CHECK_INTERVAL))s)"
    tail -n 5 "$VLLM_LOG_FILE" 2>/dev/null
  done

  echo "[ERROR] vLLM 启动超时"
  tail -n 200 "$VLLM_LOG_FILE"
  wait_for_debug "vLLM 启动超时"
  kill -9 "${VLLM_PID}" || true
  exit 1
}

stop_vllm() {
  if [ -n "${VLLM_PID:-}" ]; then
    echo "[INFO] 关闭 vLLM (PID=$VLLM_PID)"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
    VLLM_PID=""
  fi
  [ -n "${VLLM_LOG_FILE:-}" ] && echo "[INFO] vLLM 日志: $VLLM_LOG_FILE"
}

run_bfcl() {
  cd "$WORKDIR"
  deactivate
  source "${BFCL_VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export NO_PROXY="*"
  export no_proxy="*"
  export REDBOOK_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  export REDBOOK_API_KEY="${REDBOOK_API_KEY:-EMPTY}"

  echo "[INFO] 开始生成..."
  echo "[INFO] SERVED_NAME=$SERVED_NAME"
  echo "[INFO] BFCL_MODEL_NAME=$BFCL_MODEL_NAME"
  echo "[INFO] REDBOOK_BASE_URL=$REDBOOK_BASE_URL"
  echo "[INFO] BFCL_PROJECT_ROOT=$BFCL_PROJECT_ROOT"

  bfcl generate \
    --model "${BFCL_MODEL_NAME}" \
    --num-threads 16 \
    --test-category single_turn,multi_turn,memory \
    --result-dir result

  echo "[INFO] 开始评估..."
  bfcl evaluate \
    --model "${BFCL_MODEL_NAME}" \
    --test-category single_turn,multi_turn,memory \
    --result-dir result \
    --score-dir score

  echo "[INFO] 测试完成!"
  echo "[INFO] result 输出目录: $BFCL_RESULT_DIR"
  echo "[INFO] score  输出目录: $BFCL_SCORE_DIR"
}

# ===== 4) 主流程 =====
start_vllm "$MODEL_PATH"
run_bfcl
stop_vllm

echo "✅ 全部完成"
echo "📁 vLLM 日志: $VLLM_LOG_DIR"
echo "📁 BFCL result: $BFCL_RESULT_DIR"
echo "📁 BFCL score : $BFCL_SCORE_DIR"
