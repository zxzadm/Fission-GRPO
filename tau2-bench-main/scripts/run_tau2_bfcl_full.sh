#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 轮询 Key（完整）
###############################################################################
QS_TOKENS=()

if [[ -n "${QS_TOKENS_CSV:-}" ]]; then
  IFS=',' read -r -a QS_TOKENS <<<"${QS_TOKENS_CSV}"
fi

if [[ -n "${QS_API_KEY:-}" ]]; then
  QS_TOKENS=("${QS_API_KEY}")
fi

###############################################################################
# 可配参数（按需修改）
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU2_ROOT="${TAU2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BFCL_ROOT="${BFCL_ROOT:-$(cd "${SCRIPT_DIR}/../../bfcl_v4_benchmark" && pwd)}"

RUN_TAU2="${RUN_TAU2:-1}"              # 1=跑 tau2, 0=跳过
RUN_BFCL="${RUN_BFCL:-1}"              # 1=跑 BFCL, 0=跳过
KEY_RETRY_LIMIT="${KEY_RETRY_LIMIT:-12}"  # 单个命令失败时，最多换 key 重试次数

# tau2 设置（test split）
TAU2_AGENT_LLM="${TAU2_AGENT_LLM:-openai/qwen-plus}"
TAU2_USER_LLM="${TAU2_USER_LLM:-openai/qwen-plus}"
TAU2_API_BASE="${TAU2_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
TAU2_TASK_SPLIT="${TAU2_TASK_SPLIT:-test}"
TAU2_NUM_TRIALS="${TAU2_NUM_TRIALS:-1}"
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-200}"
TAU2_MAX_ERRORS="${TAU2_MAX_ERRORS:-10}"
TAU2_MAX_CONCURRENCY="${TAU2_MAX_CONCURRENCY:-1}"
TAU2_AGENT_LLM_ARGS="${TAU2_AGENT_LLM_ARGS:-{\"temperature\":0.0,\"api_base\":\"${TAU2_API_BASE}\"}}"
TAU2_USER_LLM_ARGS="${TAU2_USER_LLM_ARGS:-{\"temperature\":0.0,\"api_base\":\"${TAU2_API_BASE}\"}}"

# BFCL 设置（multi_turn）
BFCL_MODEL="${BFCL_MODEL:-qwen3-32b-FC}"
BFCL_TEST_CATEGORY="${BFCL_TEST_CATEGORY:-multi_turn}"
BFCL_NUM_THREADS="${BFCL_NUM_THREADS:-1}"
BFCL_RESULT_DIR="${BFCL_RESULT_DIR:-result}"
BFCL_SCORE_DIR="${BFCL_SCORE_DIR:-score}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

###############################################################################
# 工具函数
###############################################################################
require_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "[FATAL] 目录不存在: $d" >&2
    exit 1
  fi
}

mask_token() {
  local t="$1"
  local n=${#t}
  if (( n <= 12 )); then
    echo "$t"
  else
    echo "${t:0:8}...${t:n-4:4}"
  fi
}

validate_tokens() {
  local ok=1
  for t in "${QS_TOKENS[@]}"; do
    if [[ ! "$t" =~ ^(QST|MAAS)[0-9a-f]{32}$ ]]; then
      echo "[WARN] token 格式异常: $t" >&2
      ok=0
    fi
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[FATAL] token 列表里存在格式异常，请先修正。" >&2
    exit 1
  fi
}

tau2_cmd() {
  if command -v tau2 >/dev/null 2>&1; then
    tau2 "$@"
  else
    PYTHONPATH=src python -m tau2.cli "$@"
  fi
}

bfcl_cmd() {
  if command -v bfcl >/dev/null 2>&1; then
    bfcl "$@"
  else
    python -m bfcl_eval "$@"
  fi
}

KEY_INDEX=0
TOTAL_KEYS=${#QS_TOKENS[@]}

next_token() {
  local token="${QS_TOKENS[$KEY_INDEX]}"
  KEY_INDEX=$(( (KEY_INDEX + 1) % TOTAL_KEYS ))
  printf "%s" "$token"
}

export_all_key_envs() {
  local token="$1"
  export QS_API_KEY="$token"
  export DASHSCOPE_API_KEY="$token"
  export OPENAI_API_KEY="$token"
  export QWEN_API_KEY="$token"
}

run_with_retry_rotate() {
  local max_retries="$1"
  shift

  local attempt=1
  local rc=0
  while (( attempt <= max_retries )); do
    local token
    token="$(next_token)"
    export_all_key_envs "$token"
    echo "[INFO] Attempt ${attempt}/${max_retries}, key=$(mask_token "$token"): $*"

    if "$@"; then
      return 0
    fi

    rc=$?
    echo "[WARN] 命令失败（exit=${rc}），准备切换下一个 key 重试..."
    attempt=$((attempt + 1))
    sleep 2
  done

  return "$rc"
}

run_tau2() {
  require_dir "$TAU2_ROOT"
  cd "$TAU2_ROOT"

  echo "[INFO] 检查 tau2 数据目录..."
  run_with_retry_rotate "$KEY_RETRY_LIMIT" tau2_cmd check-data

  local domains=("telecom" "airline" "retail")
  for domain in "${domains[@]}"; do
    local save_name="qs_${RUN_TAG}_${domain}_${TAU2_TASK_SPLIT}"
    echo "[INFO] 开始 tau2 domain=${domain}, save_to=${save_name}"

    run_with_retry_rotate "$KEY_RETRY_LIMIT" \
      tau2_cmd run \
      --domain "$domain" \
      --agent llm_agent \
      --user user_simulator \
      --agent-llm "$TAU2_AGENT_LLM" \
      --user-llm "$TAU2_USER_LLM" \
      --agent-llm-args "$TAU2_AGENT_LLM_ARGS" \
      --user-llm-args "$TAU2_USER_LLM_ARGS" \
      --task-split-name "$TAU2_TASK_SPLIT" \
      --num-trials "$TAU2_NUM_TRIALS" \
      --max-steps "$TAU2_MAX_STEPS" \
      --max-errors "$TAU2_MAX_ERRORS" \
      --max-concurrency "$TAU2_MAX_CONCURRENCY" \
      --save-to "$save_name"

    echo "[INFO] tau2 ${domain} 完成"
  done

  echo "[INFO] tau2 全部完成。结果目录: ${TAU2_ROOT}/data/tau2/simulations"
}

run_bfcl() {
  require_dir "$BFCL_ROOT"
  cd "$BFCL_ROOT"
  export BFCL_PROJECT_ROOT="$BFCL_ROOT"

  echo "[INFO] 开始 BFCL generate: model=${BFCL_MODEL}, category=${BFCL_TEST_CATEGORY}"
  run_with_retry_rotate "$KEY_RETRY_LIMIT" \
    bfcl_cmd generate \
    --model "$BFCL_MODEL" \
    --test-category "$BFCL_TEST_CATEGORY" \
    --num-threads "$BFCL_NUM_THREADS" \
    --result-dir "$BFCL_RESULT_DIR"

  echo "[INFO] 开始 BFCL evaluate: model=${BFCL_MODEL}, category=${BFCL_TEST_CATEGORY}"
  run_with_retry_rotate "$KEY_RETRY_LIMIT" \
    bfcl_cmd evaluate \
    --model "$BFCL_MODEL" \
    --test-category "$BFCL_TEST_CATEGORY" \
    --result-dir "$BFCL_RESULT_DIR" \
    --score-dir "$BFCL_SCORE_DIR"

  echo "[INFO] BFCL 完成。结果目录: ${BFCL_ROOT}/${BFCL_RESULT_DIR} 与 ${BFCL_ROOT}/${BFCL_SCORE_DIR}"
}

###############################################################################
# 主流程
###############################################################################
validate_tokens
if [[ "${#QS_TOKENS[@]}" -eq 0 ]]; then
  echo "[FATAL] 请通过 QS_API_KEY 或 QS_TOKENS_CSV 提供至少一个 token。" >&2
  exit 1
fi
echo "[INFO] key 数量: ${TOTAL_KEYS}"
echo "[INFO] RUN_TAG=${RUN_TAG}"

if [[ "$RUN_TAU2" == "1" ]]; then
  run_tau2
fi

if [[ "$RUN_BFCL" == "1" ]]; then
  run_bfcl
fi

echo "[DONE] 全流程执行完成。"
