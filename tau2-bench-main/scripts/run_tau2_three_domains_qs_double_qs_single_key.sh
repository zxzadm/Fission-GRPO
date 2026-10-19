#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   QS_API_KEY=... ./scripts/run_tau2_three_domains_qs.sh smoke
#   QS_API_KEY=... ./scripts/run_tau2_three_domains_qs.sh full
#
# Optional env vars:
#   API_BASE (required for remote API mode)
#   AGENT_MODEL (default: openai/qwen3-next-80b-a3b-instruct)
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

MODE="${1:-smoke}" # smoke | full

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Invalid mode: ${MODE}. Use: smoke | full" >&2
  exit 1
fi

QS_API_KEY="${QS_API_KEY:-}"

if [[ -z "${QS_API_KEY}" ]]; then
  echo "QS_API_KEY is empty. Please set a valid key." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

API_BASE="${API_BASE:-}"
AGENT_MODEL="${AGENT_MODEL:-openai/glm-4.6}"
USER_MODEL="${USER_MODEL:-openai/glm-5}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.9}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.9}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-4096}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-4096}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TASK_SPLIT="${TASK_SPLIT:-base}"

if [[ "${MODE}" == "smoke" ]]; then
  NUM_TASKS="${NUM_TASKS:-5}"
else
  NUM_TASKS=""
fi

printf -v AGENT_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s"}' \
  "${AGENT_TEMPERATURE}" "${AGENT_MAX_TOKENS}" "${API_BASE}"
printf -v USER_LLM_ARGS '{"temperature":%s,"max_tokens":%s,"api_base":"%s"}' \
  "${USER_TEMPERATURE}" "${USER_MAX_TOKENS}" "${API_BASE}"

export OPENAI_API_KEY="${QS_API_KEY}"

tau2_cmd() {
  if command -v tau2 >/dev/null 2>&1; then
    tau2 "$@"
  else
    PYTHONPATH=src python -m tau2.cli "$@"
  fi
}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
DOMAINS=("retail" "airline" "telecom")

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] mode=${MODE}, task_split=${TASK_SPLIT}, num_trials=${NUM_TRIALS}"
echo "[INFO] agent_model=${AGENT_MODEL}, user_model=${USER_MODEL}"
echo "[INFO] api_base=${API_BASE}"

tau2_cmd check-data

for domain in "${DOMAINS[@]}"; do
  save_to="qs_${RUN_TAG}_${domain}_${MODE}"
  echo "[INFO] Running domain=${domain}, save_to=${save_to}"

  cmd=(
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
done

echo "[DONE] Finished. Results are in: ${ROOT_DIR}/data/tau2/simulations"
