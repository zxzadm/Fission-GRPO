#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_tau2_three_domains_qs.sh smoke
#   ./scripts/run_tau2_three_domains_qs.sh full
#
# Optional env vars:
#   AGENT_MODE (default: remote; options: remote | vllm)
#   USER_MODE (default: remote; options: remote | vllm)
#   API_BASE (required for remote API mode)
#   AGENT_MODEL (remote default: openai/glm-4.6; vllm default: openai/$AGENT_VLLM_SERVED_MODEL)
#   USER_MODEL (remote default: openai/glm-5; vllm default: openai/$USER_VLLM_SERVED_MODEL)
#   AGENT_TEMPERATURE (default: 0.9)
#   USER_TEMPERATURE (default: 0.9)
#   AGENT_MAX_TOKENS (default: 4096)
#   USER_MAX_TOKENS (default: 4096)
#   AGENT_LLM_EXTRA_BODY_JSON (optional; JSON object passed as extra_body in agent LLM requests)
#   USER_LLM_EXTRA_BODY_JSON (optional; JSON object passed as extra_body in user LLM requests)
#   NUM_TRIALS (default: 1)
#   MAX_STEPS (default: 200)
#   MAX_ERRORS (default: 10)
#   MAX_CONCURRENCY (default: 3)
#   TASK_SPLIT (default: base)
#   NUM_TASKS (smoke mode default: 5, full mode ignored)
#   KEY_RETRY_LIMIT (default: 1; run-level fallback retry only)
#   RETRY_SLEEP_SECS (default: 2)
#   PARALLEL_DOMAINS (default: 0; 1 means run retail/airline/telecom in parallel)
#   RETRY_RESUME (default: 1; 1 means retry uses same save_to and resume)
#   AUTO_RESUME_ON_RETRY (default: 1; auto-answer yes to resume prompts)
#   REQUEST_KEY_POOL_ENABLED (default: 1)
#   REQUEST_PER_KEY_MAX_INFLIGHT (default: 1)
#   REQUEST_MAX_RETRIES (default: 6)
#   REQUEST_BASE_DELAY (default: 1.0)
#   REQUEST_MAX_DELAY (default: 12.0)
#   REQUEST_JITTER (default: 0.5)
#   DT (optional; used in tau2_bench_<DT> output directory naming)
#   BENCH_OUTPUT_BASE_DIR (default: <repo>/outputs)
#   BENCH_TARGET_MODEL_PATH (optional; defaults to AGENT_VLLM_MODEL_PATH when present)
#   BENCH_TARGET_MODEL_NAME (optional; overrides basename-derived model directory name)
#   BENCH_RESULT_ROOT_DIR (optional; overrides the full output root for this run)
#   SIM_OUTPUT_DIR (optional; overrides BENCH_RESULT_ROOT_DIR/simulations)
#   AUTO_EXPORT_RUN_SUMMARY (default: 1; auto-write one CSV per run under <BENCH_RESULT_ROOT_DIR>/summaries)
#   RUN_SUMMARY_DIR (default: <BENCH_RESULT_ROOT_DIR>/summaries)
#   AGENT_REQUEST_KEY_POOL_ENABLED (default: 1 for remote, 0 for vllm)
#   TAU2_CONTINUE_ON_CONTEXT_WINDOW_ERROR (default: 1; mark overlong task failed and continue)
#   TAU2_CONTINUE_ON_TASK_ERROR (default: 1; mark any task runtime error as failed and continue)
#   QS_TOKENS_CSV (optional, comma-separated key list; overrides built-in list)
#   QS_API_KEY (optional, single key override for quick debugging)
#   AGENT_VLLM_AUTOSTART (default: 1 when AGENT_MODE=vllm, else ignored)
#   AGENT_VLLM_MODEL_PATH (required when AGENT_MODE=vllm and AGENT_VLLM_AUTOSTART=1)
#   AGENT_VLLM_PORT (default: 8588)
#   AGENT_VLLM_BASE_URL (default: http://127.0.0.1:${AGENT_VLLM_PORT}/v1)
#   AGENT_VLLM_SERVED_MODEL (default: vllm-agent)
#   AGENT_VLLM_API_KEY (default: EMPTY)
#   AGENT_VLLM_GPU (default: 0,1,2,3,4,5,6,7; auto-splits to 0,1,2,3 when USER_MODE=vllm too)
#   AGENT_TENSOR_PARALLEL (default: 8; auto-splits to 4 when USER_MODE=vllm too)
#   AGENT_VLLM_MAX_MODEL_LEN (default: falls back to VLLM_MAX_MODEL_LEN)
#   AGENT_MODEL_TAG (optional; if empty and AGENT_MODE=vllm, auto-derived)
#   AGENT_VLLM_EXTRA_ARGS (optional; extra args appended only to agent-side `vllm serve`)
#   USER_VLLM_AUTOSTART (default: 1 when USER_MODE=vllm, else ignored)
#   USER_VLLM_MODEL_PATH (required when USER_MODE=vllm and USER_VLLM_AUTOSTART=1)
#   USER_VLLM_PORT (default: 8589)
#   USER_VLLM_BASE_URL (default: http://127.0.0.1:${USER_VLLM_PORT}/v1)
#   USER_VLLM_SERVED_MODEL (default: vllm-user)
#   USER_VLLM_API_KEY (default: EMPTY)
#   USER_VLLM_GPU (default: 0,1,2,3,4,5,6,7; auto-splits to 4,5,6,7 when AGENT_MODE=vllm too)
#   USER_TENSOR_PARALLEL (default: 8; auto-splits to 4 when AGENT_MODE=vllm too)
#   USER_MODEL_TAG (optional; if empty and USER_MODE=vllm, auto-derived)
#   USER_VLLM_EXTRA_ARGS (optional; extra args appended only to user-side `vllm serve`)
#   VLLM_STARTUP_TIMEOUT_SECS (default: 1200)
#   VLLM_CHECK_INTERVAL_SECS (default: 5)
#   VLLM_LOG_DIR (default: /tmp/vllm_logs)
#   VLLM_GPU_MEMORY_UTILIZATION (default: 0.90)
#   VLLM_MAX_MODEL_LEN (default: 65536)
#   VLLM_MAX_NUM_SEQS (default: 16)
#   VLLM_MAX_NUM_BATCHED_TOKENS (default: 196608)
#   VLLM_ROPE_SCALING_JSON (optional)
#   VLLM_EXTRA_ARGS (optional, extra args appended to `vllm serve`)
#   Legacy compatibility: existing VLLM_* and TENSOR_PARALLEL env vars still map to the agent-side vLLM config

QS_TOKENS=()

if [[ -n "${QS_TOKENS_CSV:-}" ]]; then
  IFS=',' read -r -a QS_TOKENS <<<"${QS_TOKENS_CSV}"
fi

if [[ -n "${QS_API_KEY:-}" ]]; then
  QS_TOKENS=("${QS_API_KEY}")
fi

MODE="${1:-smoke}" # smoke | full

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Invalid mode: ${MODE}. Use: smoke | full" >&2
  exit 1
fi

if [[ "${#QS_TOKENS[@]}" -eq 0 ]]; then
  echo "No QS keys configured." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TAU2_VENV_ACTIVATE="${TAU2_VENV_ACTIVATE:-${ROOT_DIR}/.venv/bin/activate}"

API_BASE="${API_BASE:-}"
AGENT_MODE="${AGENT_MODE:-remote}"
USER_MODE="${USER_MODE:-remote}"
AGENT_MODEL="${AGENT_MODEL:-}"
USER_MODEL="${USER_MODEL:-}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.9}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.9}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-4096}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-4096}"
AGENT_LLM_EXTRA_BODY_JSON="${AGENT_LLM_EXTRA_BODY_JSON:-}"
USER_LLM_EXTRA_BODY_JSON="${USER_LLM_EXTRA_BODY_JSON:-}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
TASK_SPLIT="${TASK_SPLIT:-base}"
RETRY_SLEEP_SECS="${RETRY_SLEEP_SECS:-2}"
PARALLEL_DOMAINS="${PARALLEL_DOMAINS:-0}"
KEY_RETRY_LIMIT="${KEY_RETRY_LIMIT:-1}"
RETRY_RESUME="${RETRY_RESUME:-1}"
AUTO_RESUME_ON_RETRY="${AUTO_RESUME_ON_RETRY:-1}"
REQUEST_KEY_POOL_ENABLED="${REQUEST_KEY_POOL_ENABLED:-1}"
REQUEST_PER_KEY_MAX_INFLIGHT="${REQUEST_PER_KEY_MAX_INFLIGHT:-1}"
REQUEST_MAX_RETRIES="${REQUEST_MAX_RETRIES:-6}"
REQUEST_BASE_DELAY="${REQUEST_BASE_DELAY:-1.0}"
REQUEST_MAX_DELAY="${REQUEST_MAX_DELAY:-12.0}"
REQUEST_JITTER="${REQUEST_JITTER:-0.5}"
AUTO_EXPORT_RUN_SUMMARY="${AUTO_EXPORT_RUN_SUMMARY:-1}"
DT="${DT:-}"
DEFAULT_BENCH_OUTPUT_BASE_DIR="${ROOT_DIR}/outputs"
BENCH_OUTPUT_BASE_DIR="${BENCH_OUTPUT_BASE_DIR:-${DEFAULT_BENCH_OUTPUT_BASE_DIR}}"
BENCH_TARGET_MODEL_PATH="${BENCH_TARGET_MODEL_PATH:-}"
BENCH_TARGET_MODEL_NAME="${BENCH_TARGET_MODEL_NAME:-}"
BENCH_RESULT_ROOT_DIR="${BENCH_RESULT_ROOT_DIR:-}"
SIM_OUTPUT_DIR="${SIM_OUTPUT_DIR:-}"
RUN_SUMMARY_DIR="${RUN_SUMMARY_DIR:-}"
AGENT_REQUEST_KEY_POOL_ENABLED="${AGENT_REQUEST_KEY_POOL_ENABLED:-}"
TAU2_CONTINUE_ON_CONTEXT_WINDOW_ERROR="${TAU2_CONTINUE_ON_CONTEXT_WINDOW_ERROR:-1}"
TAU2_CONTINUE_ON_TASK_ERROR="${TAU2_CONTINUE_ON_TASK_ERROR:-1}"

DEFAULT_USER_VLLM_MODEL_PATH=""
DEFAULT_VLLM_TOOL_EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser hermes'

AGENT_VLLM_AUTOSTART="${AGENT_VLLM_AUTOSTART:-${VLLM_AUTOSTART:-1}}"
AGENT_VLLM_MODEL_PATH="${AGENT_VLLM_MODEL_PATH:-${VLLM_MODEL_PATH:-}}"
AGENT_VLLM_PORT="${AGENT_VLLM_PORT:-${VLLM_PORT:-8588}}"
AGENT_VLLM_BASE_URL="${AGENT_VLLM_BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:${AGENT_VLLM_PORT}/v1}}"
AGENT_VLLM_SERVED_MODEL="${AGENT_VLLM_SERVED_MODEL:-${VLLM_SERVED_MODEL:-vllm-agent}}"
AGENT_VLLM_API_KEY="${AGENT_VLLM_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
AGENT_VLLM_GPU="${AGENT_VLLM_GPU:-${VLLM_GPU:-}}"
AGENT_TENSOR_PARALLEL="${AGENT_TENSOR_PARALLEL:-${TENSOR_PARALLEL:-}}"
AGENT_VLLM_MAX_MODEL_LEN="${AGENT_VLLM_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-65536}}"
AGENT_MODEL_TAG="${AGENT_MODEL_TAG:-${MODEL_TAG:-}}"
AGENT_VLLM_EXTRA_ARGS="${AGENT_VLLM_EXTRA_ARGS:-${DEFAULT_VLLM_TOOL_EXTRA_ARGS} ${VLLM_EXTRA_ARGS:-}}"

USER_VLLM_AUTOSTART="${USER_VLLM_AUTOSTART:-1}"
USER_VLLM_MODEL_PATH="${USER_VLLM_MODEL_PATH:-${DEFAULT_USER_VLLM_MODEL_PATH}}"
USER_VLLM_PORT="${USER_VLLM_PORT:-8589}"
USER_VLLM_BASE_URL="${USER_VLLM_BASE_URL:-http://127.0.0.1:${USER_VLLM_PORT}/v1}"
USER_VLLM_SERVED_MODEL="${USER_VLLM_SERVED_MODEL:-vllm-user}"
USER_VLLM_API_KEY="${USER_VLLM_API_KEY:-EMPTY}"
USER_VLLM_GPU="${USER_VLLM_GPU:-}"
USER_TENSOR_PARALLEL="${USER_TENSOR_PARALLEL:-}"
USER_MODEL_TAG="${USER_MODEL_TAG:-}"
USER_VLLM_EXTRA_ARGS="${USER_VLLM_EXTRA_ARGS:-${DEFAULT_VLLM_TOOL_EXTRA_ARGS} ${VLLM_EXTRA_ARGS:-}}"

VLLM_STARTUP_TIMEOUT_SECS="${VLLM_STARTUP_TIMEOUT_SECS:-1200}"
VLLM_CHECK_INTERVAL_SECS="${VLLM_CHECK_INTERVAL_SECS:-5}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-/tmp/vllm_logs}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-196608}"
VLLM_ROPE_SCALING_JSON="${VLLM_ROPE_SCALING_JSON:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

if [[ "${AGENT_MODE}" == "remote" ]]; then
  AGENT_MODEL="${AGENT_MODEL:-openai/glm-4.6}"
elif [[ "${AGENT_MODE}" == "vllm" ]]; then
  AGENT_MODEL="${AGENT_MODEL:-openai/${AGENT_VLLM_SERVED_MODEL}}"
  if [[ -z "${AGENT_REQUEST_KEY_POOL_ENABLED}" ]]; then
    AGENT_REQUEST_KEY_POOL_ENABLED="0"
  fi
else
  echo "[FATAL] AGENT_MODE must be one of: remote, vllm" >&2
  exit 1
fi
if [[ "${AGENT_MODE}" == "remote" && -z "${AGENT_REQUEST_KEY_POOL_ENABLED}" ]]; then
  AGENT_REQUEST_KEY_POOL_ENABLED="1"
fi

if [[ "${USER_MODE}" == "remote" ]]; then
  USER_MODEL="${USER_MODEL:-openai/glm-5}"
elif [[ "${USER_MODE}" == "vllm" ]]; then
  USER_MODEL="${USER_MODEL:-openai/${USER_VLLM_SERVED_MODEL}}"
else
  echo "[FATAL] USER_MODE must be one of: remote, vllm" >&2
  exit 1
fi

apply_dual_vllm_defaults() {
  local enable_local_split="${DUAL_VLLM_SPLIT_LOCAL_NODE:-1}"
  if [[ "${AGENT_MODE}" == "vllm" && "${USER_MODE}" == "vllm" && "${enable_local_split}" == "1" ]]; then
    if [[ -z "${AGENT_VLLM_GPU}" || "${AGENT_VLLM_GPU}" == "0,1,2,3,4,5,6,7" ]]; then
      AGENT_VLLM_GPU="0,1,2,3"
    fi
    USER_VLLM_GPU="${USER_VLLM_GPU:-4,5,6,7}"
    if [[ -z "${AGENT_TENSOR_PARALLEL}" || "${AGENT_TENSOR_PARALLEL}" == "8" ]]; then
      AGENT_TENSOR_PARALLEL="4"
    fi
    USER_TENSOR_PARALLEL="${USER_TENSOR_PARALLEL:-4}"
  else
    if [[ "${AGENT_MODE}" == "vllm" ]]; then
      AGENT_VLLM_GPU="${AGENT_VLLM_GPU:-0,1,2,3,4,5,6,7}"
      AGENT_TENSOR_PARALLEL="${AGENT_TENSOR_PARALLEL:-8}"
    fi
    if [[ "${USER_MODE}" == "vllm" ]]; then
      USER_VLLM_GPU="${USER_VLLM_GPU:-0,1,2,3,4,5,6,7}"
      USER_TENSOR_PARALLEL="${USER_TENSOR_PARALLEL:-8}"
    fi
  fi
}

sanitize_model_tag() {
  local raw="$1"
  raw="$(echo "${raw}" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//;s/_*$//')"
  raw="${raw:0:48}"
  if [[ -z "${raw}" ]]; then
    raw="vllm_model"
  fi
  printf "%s" "${raw}"
}

build_model_tags() {
  if [[ "${AGENT_MODE}" == "vllm" && -z "${AGENT_MODEL_TAG}" ]]; then
    if [[ -n "${AGENT_VLLM_MODEL_PATH}" ]]; then
      AGENT_MODEL_TAG="$(sanitize_model_tag "$(basename "${AGENT_VLLM_MODEL_PATH}")")"
    else
      AGENT_MODEL_TAG="$(sanitize_model_tag "${AGENT_VLLM_SERVED_MODEL}")"
    fi
  fi
  if [[ "${USER_MODE}" == "vllm" && -z "${USER_MODEL_TAG}" ]]; then
    if [[ -n "${USER_VLLM_MODEL_PATH}" ]]; then
      USER_MODEL_TAG="$(sanitize_model_tag "$(basename "${USER_VLLM_MODEL_PATH}")")"
    else
      USER_MODEL_TAG="$(sanitize_model_tag "${USER_VLLM_SERVED_MODEL}")"
    fi
  fi
}

resolve_bench_target_model_name() {
  if [[ -n "${BENCH_TARGET_MODEL_NAME}" ]]; then
    printf "%s" "$(sanitize_model_tag "${BENCH_TARGET_MODEL_NAME}")"
    return 0
  fi

  if [[ -z "${BENCH_TARGET_MODEL_PATH}" && -n "${AGENT_VLLM_MODEL_PATH}" ]]; then
    BENCH_TARGET_MODEL_PATH="${AGENT_VLLM_MODEL_PATH}"
  fi

  if [[ -n "${BENCH_TARGET_MODEL_PATH}" ]]; then
    printf "%s" "$(sanitize_model_tag "$(basename "${BENCH_TARGET_MODEL_PATH}")")"
    return 0
  fi

  if [[ -n "${AGENT_MODEL_TAG}" ]]; then
    printf "%s" "${AGENT_MODEL_TAG}"
    return 0
  fi

  printf "%s" "$(sanitize_model_tag "${AGENT_MODEL}")"
}

configure_output_dirs() {
  local bench_model_name
  bench_model_name="$(resolve_bench_target_model_name)"

  if [[ -z "${DT}" ]]; then
    DT="${RUN_TAG}"
  fi

  if [[ -z "${BENCH_RESULT_ROOT_DIR}" ]]; then
    BENCH_RESULT_ROOT_DIR="${BENCH_OUTPUT_BASE_DIR}/${bench_model_name}/compass_outputs/tau2_bench_${DT}"
  fi

  if [[ -z "${SIM_OUTPUT_DIR}" ]]; then
    SIM_OUTPUT_DIR="${BENCH_RESULT_ROOT_DIR}/simulations"
  fi

  if [[ -z "${RUN_SUMMARY_DIR}" ]]; then
    RUN_SUMMARY_DIR="${BENCH_RESULT_ROOT_DIR}/summaries"
  fi

  mkdir -p "${SIM_OUTPUT_DIR}" "${RUN_SUMMARY_DIR}"
}

is_local_base_url() {
  local base_url="$1"
  [[ "${base_url}" =~ ^https?://(127\.0\.0\.1|localhost)(:|/|$) ]]
}

extract_base_url_host() {
  local base_url="$1"
  local without_scheme="${base_url#http://}"
  without_scheme="${without_scheme#https://}"
  printf "%s" "${without_scheme%%[:/]*}"
}

append_hosts_to_no_proxy() {
  local np_existing="${NO_PROXY:-${no_proxy:-}}"
  local combined="${np_existing}"
  local host
  for host in "$@"; do
    [[ -n "${host}" ]] || continue
    if [[ -n "${combined}" ]]; then
      combined="${combined},${host}"
    else
      combined="${host}"
    fi
  done
  export NO_PROXY="${combined}"
  export no_proxy="${NO_PROXY}"
}

append_base_url_host_to_no_proxy() {
  local base_url="$1"
  local host
  host="$(extract_base_url_host "${base_url}")"
  append_hosts_to_no_proxy "${host}"
}

args_contain_flag() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

validate_vllm_layout() {
  if [[ "${AGENT_MODE}" == "vllm" && "${USER_MODE}" == "vllm" ]]; then
    # Only check local port/GPU conflicts when both vLLM servers are actually
    # launched on this node. In two-node mode the user-side vLLM is remote.
    if [[ "${AGENT_VLLM_AUTOSTART}" != "1" || "${USER_VLLM_AUTOSTART}" != "1" ]]; then
      return 0
    fi
    if [[ "${AGENT_VLLM_PORT}" == "${USER_VLLM_PORT}" ]]; then
      echo "[FATAL] AGENT_VLLM_PORT and USER_VLLM_PORT must be different when both run in vllm mode" >&2
      exit 1
    fi
    local agent_gpu user_gpu
    for agent_gpu in ${AGENT_VLLM_GPU//,/ }; do
      for user_gpu in ${USER_VLLM_GPU//,/ }; do
        if [[ "${agent_gpu}" == "${user_gpu}" ]]; then
          echo "[FATAL] AGENT_VLLM_GPU and USER_VLLM_GPU overlap on GPU ${agent_gpu}" >&2
          exit 1
        fi
      done
    done
  fi
}

if [[ "${MODE}" == "smoke" ]]; then
  NUM_TASKS="${NUM_TASKS:-5}"
else
  NUM_TASKS=""
fi

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
  local t
  for t in "${QS_TOKENS[@]}"; do
    if [[ ! "$t" =~ ^(QST|MAAS)[0-9a-f]{32}$ ]]; then
      echo "[WARN] token format unexpected: $(mask_token "$t")" >&2
      ok=0
    fi
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[FATAL] token list contains invalid format. Please check keys." >&2
    exit 1
  fi
}

join_tokens_csv() {
  local IFS=","
  echo "$*"
}

tau2_cmd() {
  if command -v tau2 >/dev/null 2>&1; then
    tau2 "$@"
  else
    PYTHONPATH=src python -m tau2.cli "$@"
  fi
}

activate_tau2_env() {
  deactivate 2>/dev/null || true
  if [[ ! -f "${TAU2_VENV_ACTIVATE}" ]]; then
    echo "[FATAL] tau2 virtualenv activate script not found: ${TAU2_VENV_ACTIVATE}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${TAU2_VENV_ACTIVATE}"
}

AGENT_VLLM_PID=""
AGENT_VLLM_LOG_FILE=""
USER_VLLM_PID=""
USER_VLLM_LOG_FILE=""

start_vllm_server_if_needed() {
  local role="$1"
  local mode="$2"
  local autostart="$3"
  local model_path="$4"
  local port="$5"
  local base_url="$6"
  local served_model="$7"
  local gpu="$8"
  local tensor_parallel="$9"
  local max_model_len="${10}"
  local extra_args_raw="${11}"
  local pid_var="${12}"
  local log_var="${13}"

  if [[ "${mode}" != "vllm" ]]; then
    return 0
  fi
  if [[ "${autostart}" != "1" ]]; then
    echo "[INFO] ${role}_MODE=vllm and ${role}_VLLM_AUTOSTART=0, will use existing endpoint: ${base_url}"
    return 0
  fi
  if [[ -z "${model_path}" ]]; then
    echo "[FATAL] ${role}_VLLM_MODEL_PATH is required when ${role}_MODE=vllm and ${role}_VLLM_AUTOSTART=1" >&2
    exit 1
  fi
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[FATAL] 'vllm' command not found in PATH." >&2
    exit 1
  fi

  mkdir -p "${VLLM_LOG_DIR}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${VLLM_LOG_DIR}/vllm_tau2_${role,,}_${ts}.log"

  echo "[INFO] starting ${role,,} vLLM..."
  echo "[INFO] ${role,,}_model_path=${model_path}"
  echo "[INFO] ${role,,}_served_model=${served_model}"
  echo "[INFO] ${role,,}_base_url=${base_url}"
  echo "[INFO] ${role,,}_gpu=${gpu}"
  echo "[INFO] ${role,,}_tensor_parallel=${tensor_parallel}"
  echo "[INFO] ${role,,}_max_model_len=${max_model_len}"
  echo "[INFO] ${role,,}_log=${log_file}"

  local extra_args
  extra_args=()
  if [[ -n "${VLLM_ROPE_SCALING_JSON}" ]]; then
    extra_args+=(--rope-scaling "${VLLM_ROPE_SCALING_JSON}")
  fi
  if [[ -n "${extra_args_raw}" ]]; then
    # shellcheck disable=SC2206
    local parsed_extra=( ${extra_args_raw} )
    extra_args+=("${parsed_extra[@]}")
  fi

  local max_model_len_args
  max_model_len_args=()
  if ! args_contain_flag "--max-model-len" "${extra_args[@]}"; then
    max_model_len_args=(--max-model-len "${max_model_len}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  vllm serve "${model_path}" \
  --port "${port}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  "${max_model_len_args[@]}" \
  --tensor-parallel-size "${tensor_parallel}" \
  --served-model-name "${served_model}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  "${extra_args[@]}" \
  >"${log_file}" 2>&1 &
  local pid=$!

  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT_SECS )); do
    sleep "${VLLM_CHECK_INTERVAL_SECS}"
    waited=$(( waited + VLLM_CHECK_INTERVAL_SECS ))

    if ! ps -p "${pid}" >/dev/null 2>&1; then
      echo "[FATAL] ${role,,} vLLM process exited unexpectedly. See log: ${log_file}" >&2
      tail -n 120 "${log_file}" || true
      exit 1
    fi

    if curl --noproxy localhost,127.0.0.1 -fs "${base_url%/v1}/health" >/dev/null 2>&1; then
      printf -v "${pid_var}" "%s" "${pid}"
      printf -v "${log_var}" "%s" "${log_file}"
      echo "[INFO] ${role,,} vLLM is healthy after ${waited}s"
      return 0
    fi
  done

  echo "[FATAL] ${role,,} vLLM startup timeout after ${VLLM_STARTUP_TIMEOUT_SECS}s. Log: ${log_file}" >&2
  tail -n 120 "${log_file}" || true
  exit 1
}

stop_vllm_server_if_needed() {
  local role="$1"
  local pid="$2"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    echo "[INFO] stopping ${role,,} vLLM (pid=${pid})"
    kill "${pid}" || true
    wait "${pid}" || true
  fi
}

cleanup() {
  stop_vllm_server_if_needed "USER" "${USER_VLLM_PID}"
  stop_vllm_server_if_needed "AGENT" "${AGENT_VLLM_PID}"
}

trap cleanup EXIT

TOTAL_KEYS=${#QS_TOKENS[@]}
if (( KEY_RETRY_LIMIT > TOTAL_KEYS )); then
  KEY_RETRY_LIMIT="${TOTAL_KEYS}"
fi
QS_TOKENS_JOINED="$(join_tokens_csv "${QS_TOKENS[@]}")"
token_for_attempt() {
  local base_index="$1"
  local attempt="$2"
  local idx=$(( (base_index + attempt - 1) % TOTAL_KEYS ))
  printf "%s" "${QS_TOKENS[$idx]}"
}

append_extra_body_json() {
  local var_name="$1"
  local extra_body_json="$2"
  if [[ -z "${extra_body_json}" ]]; then
    return 0
  fi
  local current="${!var_name}"
  current="${current%\}}"
  printf -v "${var_name}" '%s,"extra_body":%s}' "${current}" "${extra_body_json}"
}

build_llm_args() {
  local fallback_key="$1"
  local agent_api_base="${API_BASE}"
  local agent_api_key="${fallback_key}"
  local user_api_base="${API_BASE}"
  local user_api_key="${fallback_key}"
  if [[ "${AGENT_MODE}" == "vllm" ]]; then
    agent_api_base="${AGENT_VLLM_BASE_URL}"
    agent_api_key="${AGENT_VLLM_API_KEY}"
  fi
  if [[ "${USER_MODE}" == "vllm" ]]; then
    user_api_base="${USER_VLLM_BASE_URL}"
    user_api_key="${USER_VLLM_API_KEY}"
  fi

  if [[ "${AGENT_MODE}" == "remote" ]]; then
    printf -v AGENT_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s","api_key":"%s","api_keys_csv":"%s","request_key_pool_enabled":%s,"request_per_key_max_inflight":%s,"request_max_retries":%s,"request_base_delay":%s,"request_max_delay":%s,"request_jitter":%s}' \
      "${AGENT_TEMPERATURE}" "${AGENT_MAX_TOKENS}" "${agent_api_base}" "${agent_api_key}" "${QS_TOKENS_JOINED}" \
      "${AGENT_REQUEST_KEY_POOL_ENABLED}" "${REQUEST_PER_KEY_MAX_INFLIGHT}" "${REQUEST_MAX_RETRIES}" "${REQUEST_BASE_DELAY}" "${REQUEST_MAX_DELAY}" "${REQUEST_JITTER}"
  else
    printf -v AGENT_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s","api_key":"%s"}' \
      "${AGENT_TEMPERATURE}" "${AGENT_MAX_TOKENS}" "${agent_api_base}" "${agent_api_key}"
  fi
  append_extra_body_json AGENT_LLM_ARGS "${AGENT_LLM_EXTRA_BODY_JSON}"

  if [[ "${USER_MODE}" == "remote" ]]; then
    printf -v USER_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s","api_key":"%s","api_keys_csv":"%s","request_key_pool_enabled":%s,"request_per_key_max_inflight":%s,"request_max_retries":%s,"request_base_delay":%s,"request_max_delay":%s,"request_jitter":%s}' \
      "${USER_TEMPERATURE}" "${USER_MAX_TOKENS}" "${user_api_base}" "${user_api_key}" "${QS_TOKENS_JOINED}" \
      "${REQUEST_KEY_POOL_ENABLED}" "${REQUEST_PER_KEY_MAX_INFLIGHT}" "${REQUEST_MAX_RETRIES}" "${REQUEST_BASE_DELAY}" "${REQUEST_MAX_DELAY}" "${REQUEST_JITTER}"
  else
    printf -v USER_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s","api_key":"%s"}' \
      "${USER_TEMPERATURE}" "${USER_MAX_TOKENS}" "${user_api_base}" "${user_api_key}"
  fi
  append_extra_body_json USER_LLM_ARGS "${USER_LLM_EXTRA_BODY_JSON}"
}

run_one_domain_with_key() {
  local domain="$1"
  local token="$2"
  local save_path="$3"
  local attempt="$4"

  export OPENAI_API_KEY="${token}"
  export TAU2_CONTINUE_ON_CONTEXT_WINDOW_ERROR="${TAU2_CONTINUE_ON_CONTEXT_WINDOW_ERROR}"
  export TAU2_CONTINUE_ON_TASK_ERROR="${TAU2_CONTINUE_ON_TASK_ERROR}"
  build_llm_args "${token}"

  # vLLM endpoints are direct OpenAI-compatible servers and should not traverse
  # a corporate HTTP proxy, including cross-node private IPs.
  if [[ "${AGENT_MODE}" == "vllm" ]]; then
    append_base_url_host_to_no_proxy "${AGENT_VLLM_BASE_URL}"
  fi
  if [[ "${USER_MODE}" == "vllm" ]]; then
    append_base_url_host_to_no_proxy "${USER_VLLM_BASE_URL}"
  fi

  local cmd=(
    tau2_cmd run
    --domain "${domain}"
    --agent llm_agent
    --user user_simulator
    --agent-llm "${AGENT_MODEL}"
    --user-llm "${USER_MODEL}"
    --agent-llm-args "${AGENT_LLM_ARGS}"
    --user-llm-args "${USER_LLM_ARGS}"
    --task-split-name "${TASK_SPLIT}"
    --num-trials "${NUM_TRIALS}"
    --max-steps "${MAX_STEPS}"
    --max-errors "${MAX_ERRORS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --save-to "${save_path}"
  )

  if [[ "${MODE}" == "smoke" ]]; then
    cmd+=(--num-tasks "${NUM_TASKS}")
  fi

  if [[ "${AUTO_RESUME_ON_RETRY}" == "1" && "${attempt}" -gt 1 && -f "${save_path}" ]]; then
    printf 'y\ny\ny\ny\n' | "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_domain_with_retry_rotate() {
  local domain="$1"
  local base_index="$2"
  local attempt=1
  local rc=0
  local base_save_to
  local save_to
  local save_path

  base_save_to="qs_${RUN_TAG}_${domain}_${MODE}"
  if [[ "${AGENT_MODE}" == "vllm" ]]; then
    base_save_to="${base_save_to}_${AGENT_MODEL_TAG}"
  fi
  if [[ "${USER_MODE}" == "vllm" ]]; then
    base_save_to="${base_save_to}_user-${USER_MODEL_TAG}"
  fi

  while (( attempt <= KEY_RETRY_LIMIT )); do
    local token
    token="$(token_for_attempt "${base_index}" "${attempt}")"

    save_to="${base_save_to}"
    if [[ "${RETRY_RESUME}" != "1" && "${attempt}" -gt 1 ]]; then
      save_to="${save_to}_retry${attempt}"
    fi

    save_path="${SIM_OUTPUT_DIR}/${save_to}.json"

    echo "[INFO] domain=${domain} attempt=${attempt}/${KEY_RETRY_LIMIT} fallback_key=$(mask_token "${token}") save_to=${save_to}"
    echo "[INFO] domain=${domain} save_path=${save_path}"

    if run_one_domain_with_key "${domain}" "${token}" "${save_path}" "${attempt}"; then
      rc=0
    else
      rc=$?
    fi
    if [[ "${rc}" -eq 0 ]]; then
      echo "[INFO] domain=${domain} completed"
      return 0
    fi

    echo "[WARN] domain=${domain} failed (exit=${rc}), fallback run-level retry..."
    attempt=$((attempt + 1))
    sleep "${RETRY_SLEEP_SECS}"
  done

  return "${rc}"
}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DOMAINS=("retail" "airline" "telecom")

build_summary_model_label() {
  local mode="$1"
  local model="$2"
  local model_tag="$3"
  if [[ "${mode}" == "vllm" && -n "${model_tag}" ]]; then
    printf "%s" "${model_tag}"
  else
    printf "%s" "$(sanitize_model_tag "${model}")"
  fi
}

emit_run_summary() {
  if [[ "${AUTO_EXPORT_RUN_SUMMARY}" != "1" ]]; then
    return 0
  fi

  local agent_label user_label output_path
  agent_label="$(build_summary_model_label "${AGENT_MODE}" "${AGENT_MODEL}" "${AGENT_MODEL_TAG}")"
  user_label="$(build_summary_model_label "${USER_MODE}" "${USER_MODEL}" "${USER_MODEL_TAG}")"
  output_path="${RUN_SUMMARY_DIR}/qs_${RUN_TAG}_${MODE}_${agent_label}_user-${user_label}.csv"

  mkdir -p "${RUN_SUMMARY_DIR}"
  if PYTHONPATH=src python "scripts/export_run_summary_csv.py" \
    --run-tag "${RUN_TAG}" \
    --sim-dir "${SIM_OUTPUT_DIR}" \
    --output "${output_path}" \
    --mode "${MODE}" \
    --agent-label "${agent_label}" \
    --user-label "${user_label}"; then
    echo "[INFO] run_summary_csv=${output_path}"
  else
    echo "[WARN] failed to export run summary CSV for run_tag=${RUN_TAG}" >&2
  fi
}

validate_tokens
apply_dual_vllm_defaults
validate_vllm_layout
build_model_tags
configure_output_dirs

start_vllm_server_if_needed \
  "AGENT" \
  "${AGENT_MODE}" \
  "${AGENT_VLLM_AUTOSTART}" \
  "${AGENT_VLLM_MODEL_PATH}" \
  "${AGENT_VLLM_PORT}" \
  "${AGENT_VLLM_BASE_URL}" \
  "${AGENT_VLLM_SERVED_MODEL}" \
  "${AGENT_VLLM_GPU}" \
  "${AGENT_TENSOR_PARALLEL}" \
  "${AGENT_VLLM_MAX_MODEL_LEN}" \
  "${AGENT_VLLM_EXTRA_ARGS}" \
  "AGENT_VLLM_PID" \
  "AGENT_VLLM_LOG_FILE"
start_vllm_server_if_needed \
  "USER" \
  "${USER_MODE}" \
  "${USER_VLLM_AUTOSTART}" \
  "${USER_VLLM_MODEL_PATH}" \
  "${USER_VLLM_PORT}" \
  "${USER_VLLM_BASE_URL}" \
  "${USER_VLLM_SERVED_MODEL}" \
  "${USER_VLLM_GPU}" \
  "${USER_TENSOR_PARALLEL}" \
  "${VLLM_MAX_MODEL_LEN}" \
  "${USER_VLLM_EXTRA_ARGS}" \
  "USER_VLLM_PID" \
  "USER_VLLM_LOG_FILE"

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] mode=${MODE}, task_split=${TASK_SPLIT}, num_trials=${NUM_TRIALS}"
echo "[INFO] agent_model=${AGENT_MODEL}, user_model=${USER_MODEL}"
echo "[INFO] agent_mode=${AGENT_MODE}, user_mode=${USER_MODE}"
echo "[INFO] dt=${DT}"
echo "[INFO] bench_output_base_dir=${BENCH_OUTPUT_BASE_DIR}"
echo "[INFO] bench_result_root_dir=${BENCH_RESULT_ROOT_DIR}"
echo "[INFO] sim_output_dir=${SIM_OUTPUT_DIR}"
echo "[INFO] run_summary_dir=${RUN_SUMMARY_DIR}"
if [[ -n "${BENCH_TARGET_MODEL_PATH}" ]]; then
  echo "[INFO] bench_target_model_path=${BENCH_TARGET_MODEL_PATH}"
fi
if [[ "${AGENT_MODE}" == "vllm" ]]; then
  echo "[INFO] agent_api_base=${AGENT_VLLM_BASE_URL} served_model=${AGENT_VLLM_SERVED_MODEL}"
  echo "[INFO] agent_gpu=${AGENT_VLLM_GPU} tensor_parallel=${AGENT_TENSOR_PARALLEL}"
  echo "[INFO] agent_max_model_len=${AGENT_VLLM_MAX_MODEL_LEN}"
  echo "[INFO] agent_model_tag=${AGENT_MODEL_TAG}"
else
  echo "[INFO] agent_api_base=${API_BASE}"
fi
if [[ "${USER_MODE}" == "vllm" ]]; then
  echo "[INFO] user_api_base=${USER_VLLM_BASE_URL} served_model=${USER_VLLM_SERVED_MODEL}"
  echo "[INFO] user_gpu=${USER_VLLM_GPU} tensor_parallel=${USER_TENSOR_PARALLEL}"
  echo "[INFO] user_model_tag=${USER_MODEL_TAG}"
else
  echo "[INFO] user_api_base=${API_BASE}"
fi
echo "[INFO] keys=${TOTAL_KEYS}, key_retry_limit=${KEY_RETRY_LIMIT}, parallel_domains=${PARALLEL_DOMAINS}"
echo "[INFO] retry_resume=${RETRY_RESUME}, auto_resume_on_retry=${AUTO_RESUME_ON_RETRY}"
echo "[INFO] request_pool_user=${REQUEST_KEY_POOL_ENABLED}, request_pool_agent=${AGENT_REQUEST_KEY_POOL_ENABLED}"
echo "[INFO] per_key_inflight=${REQUEST_PER_KEY_MAX_INFLIGHT}, request_max_retries=${REQUEST_MAX_RETRIES}, backoff=${REQUEST_BASE_DELAY}/${REQUEST_MAX_DELAY}, jitter=${REQUEST_JITTER}"
echo "[INFO] continue_on_context_window_error=${TAU2_CONTINUE_ON_CONTEXT_WINDOW_ERROR}"
echo "[INFO] continue_on_task_error=${TAU2_CONTINUE_ON_TASK_ERROR}"

activate_tau2_env
tau2_cmd check-data

if [[ "${PARALLEL_DOMAINS}" == "1" ]]; then
  pids=()
  idx=0
  for domain in "${DOMAINS[@]}"; do
    run_domain_with_retry_rotate "${domain}" "${idx}" &
    pids+=("$!")
    idx=$((idx + 1))
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    emit_run_summary
    echo "[FATAL] one or more domains failed" >&2
    exit 1
  fi
else
  failed=0
  idx=0
  for domain in "${DOMAINS[@]}"; do
    if ! run_domain_with_retry_rotate "${domain}" "${idx}"; then
      failed=1
      break
    fi
    idx=$((idx + 1))
  done
  if [[ "${failed}" -ne 0 ]]; then
    emit_run_summary
    echo "[FATAL] one or more domains failed" >&2
    exit 1
  fi
fi

emit_run_summary
echo "[DONE] Finished. Simulation results are in: ${SIM_OUTPUT_DIR}"
echo "[DONE] Run summary CSV is in: ${RUN_SUMMARY_DIR}"
