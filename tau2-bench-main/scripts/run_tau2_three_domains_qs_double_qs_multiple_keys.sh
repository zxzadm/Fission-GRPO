#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_tau2_three_domains_qs.sh smoke
#   ./scripts/run_tau2_three_domains_qs.sh full
#
# Optional env vars:
#   API_BASE (required for remote API mode)
#   AGENT_MODEL (default: openai/glm-4.6)
#   USER_MODEL (default: openai/glm-5)
#   AGENT_TEMPERATURE (default: 0.9)
#   USER_TEMPERATURE (default: 0.9)
#   AGENT_MAX_TOKENS (default: 4096)
#   USER_MAX_TOKENS (default: 4096)
#   NUM_TRIALS (default: 1)
#   MAX_STEPS (default: 200)
#   MAX_ERRORS (default: 10)
#   MAX_CONCURRENCY (default: 1)
#   TASK_SPLIT (default: base)
#   NUM_TASKS (smoke mode default: 5, full mode ignored)
#   KEY_RETRY_LIMIT (default: number of keys)
#   RETRY_SLEEP_SECS (default: 2)
#   PARALLEL_DOMAINS (default: 0; 1 means run retail/airline/telecom in parallel)
#   QS_TOKENS_CSV (optional, comma-separated key list; overrides built-in list)
#   QS_API_KEY (optional, single key override for quick debugging)

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

API_BASE="${API_BASE:-}"
AGENT_MODEL="${AGENT_MODEL:-openai/glm-4.6}"
USER_MODEL="${USER_MODEL:-openai/glm-4.7}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.9}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.9}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-4096}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-4096}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
TASK_SPLIT="${TASK_SPLIT:-base}"
RETRY_SLEEP_SECS="${RETRY_SLEEP_SECS:-2}"
PARALLEL_DOMAINS="${PARALLEL_DOMAINS:-0}"
KEY_RETRY_LIMIT="${KEY_RETRY_LIMIT:-${#QS_TOKENS[@]}}"

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

tau2_cmd() {
  if command -v tau2 >/dev/null 2>&1; then
    tau2 "$@"
  else
    PYTHONPATH=src python -m tau2.cli "$@"
  fi
}

TOTAL_KEYS=${#QS_TOKENS[@]}
token_for_attempt() {
  local base_index="$1"
  local attempt="$2"
  local idx=$(( (base_index + attempt - 1) % TOTAL_KEYS ))
  printf "%s" "${QS_TOKENS[$idx]}"
}

build_llm_args() {
  local token="$1"
  printf -v AGENT_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s","api_key":"%s"}' \
    "${AGENT_TEMPERATURE}" "${AGENT_MAX_TOKENS}" "${API_BASE}" "${token}"
  printf -v USER_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s","api_key":"%s"}' \
    "${USER_TEMPERATURE}" "${USER_MAX_TOKENS}" "${API_BASE}" "${token}"
}

run_one_domain_with_key() {
  local domain="$1"
  local token="$2"
  local save_to="$3"

  export OPENAI_API_KEY="${token}"
  build_llm_args "${token}"

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
    --save-to "${save_to}"
  )

  if [[ "${MODE}" == "smoke" ]]; then
    cmd+=(--num-tasks "${NUM_TASKS}")
  fi

  "${cmd[@]}"
}

run_domain_with_retry_rotate() {
  local domain="$1"
  local base_index="$2"
  local attempt=1
  local rc=0
  local save_to

  while (( attempt <= KEY_RETRY_LIMIT )); do
    local token
    token="$(token_for_attempt "${base_index}" "${attempt}")"
    save_to="qs_${RUN_TAG}_${domain}_${MODE}"
    if (( attempt > 1 )); then
      save_to="${save_to}_retry${attempt}"
    fi

    echo "[INFO] domain=${domain} attempt=${attempt}/${KEY_RETRY_LIMIT} key=$(mask_token "${token}") save_to=${save_to}"

    if run_one_domain_with_key "${domain}" "${token}" "${save_to}"; then
      echo "[INFO] domain=${domain} completed"
      return 0
    fi

    rc=$?
    echo "[WARN] domain=${domain} failed (exit=${rc}), rotating key and retrying..."
    attempt=$((attempt + 1))
    sleep "${RETRY_SLEEP_SECS}"
  done

  return "${rc}"
}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DOMAINS=("retail" "airline" "telecom")

validate_tokens

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] mode=${MODE}, task_split=${TASK_SPLIT}, num_trials=${NUM_TRIALS}"
echo "[INFO] agent_model=${AGENT_MODEL}, user_model=${USER_MODEL}"
echo "[INFO] api_base=${API_BASE}"
echo "[INFO] keys=${TOTAL_KEYS}, key_retry_limit=${KEY_RETRY_LIMIT}, parallel_domains=${PARALLEL_DOMAINS}"

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
    echo "[FATAL] one or more domains failed" >&2
    exit 1
  fi
else
  idx=0
  for domain in "${DOMAINS[@]}"; do
    run_domain_with_retry_rotate "${domain}" "${idx}"
    idx=$((idx + 1))
  done
fi

echo "[DONE] Finished. Results are in: ${ROOT_DIR}/data/simulations"
