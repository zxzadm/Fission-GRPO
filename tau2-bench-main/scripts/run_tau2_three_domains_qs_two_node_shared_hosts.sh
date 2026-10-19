#!/usr/bin/env bash
set -euo pipefail

# Shared-hostfile two-node workflow.
#
# Assumption:
# - The same script is launched on both nodes of one multi-node job.
# - Node role is decided by platform rank env vars when available.
# - If rank env vars are absent, both nodes register into a shared `nodes.txt`
#   and the lexicographically first node becomes rank 0.
# - Rank 0 runs agent vLLM + benchmark.
# - Rank 1 runs user vLLM.
# - Both nodes can read/write the same SHARED_RUN_BASE_DIR on a shared filesystem.
# - Each launch uses a fresh timestamped run directory under SHARED_RUN_BASE_DIR.
#
# Usage example:
#   SHARED_RUN_BASE_DIR=/path/to/shared/run_dir \
#   AGENT_VLLM_MODEL_PATH=/path/to/agent/model \
#   ./scripts/run_tau2_three_domains_qs_two_node_shared_hosts.sh smoke
#
# Required env vars:
#   AGENT_VLLM_MODEL_PATH     Agent model path on rank 0 node
#
# Important optional env vars:
#   SHARED_RUN_BASE_DIR       Shared parent directory visible from both nodes
#                             (default: <repo>/shared_runs)
#   SHARED_RUN_DIR            Backward-compatible alias for SHARED_RUN_BASE_DIR
#   LOCAL_IP_OVERRIDE         Explicit node IP to advertise into hosts.txt
#   SESSION_JOIN_WINDOW_SECS  Max age for another node to join the same fresh run dir (default: 1200)
#   SHARED_RUN_GROUP_KEY      Optional extra namespace to isolate concurrent jobs
#   EXPECTED_NODES            Number of nodes expected in this job (default: 2)
#   RANK / NODE_RANK / SLURM_NODEID / OMPI_COMM_WORLD_RANK
#                             Optional platform rank env vars; if absent, auto-elected
#   USER_VLLM_MODEL_PATH      User model path on rank 1 node
#   USER_VLLM_SERVED_MODEL    User served model name (default: vllm-user)
#   USER_REMOTE_PORT          User vLLM port on rank 1 node (default: 8588)
#   USER_VLLM_GPU             User GPUs on rank 1 node (default: 0,1,2,3,4,5,6,7)
#   USER_TENSOR_PARALLEL      User TP on rank 1 node (default: 8)
#   USER_VLLM_EXTRA_ARGS      Extra args for user-side vLLM
#   AGENT_VLLM_EXTRA_ARGS     Passed through to the inner script for agent-side vLLM
#   AGENT_VLLM_PORT           Agent vLLM port on rank 0 node (default inherited by inner script)
#   AGENT_VLLM_GPU            Agent GPUs on rank 0 node (default inherited by inner script)
#   AGENT_TENSOR_PARALLEL     Agent TP on rank 0 node (default inherited by inner script)
#   VLLM_STARTUP_TIMEOUT_SECS Health-check timeout (default: 1200)
#   VLLM_CHECK_INTERVAL_SECS  Health-check interval (default: 5)
#   DT / BENCH_OUTPUT_BASE_DIR / BENCH_TARGET_MODEL_PATH / BENCH_TARGET_MODEL_NAME
#                             Passed through to the inner benchmark script to
#                             control unified result output directories.
#
# Output files inside SHARED_RUN_BASE_DIR/run_<datetime>[_N]:
#   nodes.txt                 Node registration file for role election
#   hosts.txt                 Service discovery file
#   done.flag                 Created by rank 0 when benchmark finishes
#   logs/                     User-side vLLM logs

MODE="${1:-smoke}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/run_tau2_three_domains_qs.sh"

DEFAULT_SHARED_RUN_BASE_DIR="${ROOT_DIR}/shared_runs"
SHARED_RUN_BASE_DIR="${SHARED_RUN_BASE_DIR:-${SHARED_RUN_DIR:-${DEFAULT_SHARED_RUN_BASE_DIR}}}"
EXPECTED_NODES="${EXPECTED_NODES:-2}"
SESSION_JOIN_WINDOW_SECS="${SESSION_JOIN_WINDOW_SECS:-1200}"
SHARED_RUN_GROUP_KEY="${SHARED_RUN_GROUP_KEY:-default}"

PLATFORM_RANK=""
RANK=""
RUN_ID=""
DEFAULT_USER_VLLM_MODEL_PATH=""
DEFAULT_VLLM_TOOL_EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser hermes'
USER_VLLM_MODEL_PATH="${USER_VLLM_MODEL_PATH:-${DEFAULT_USER_VLLM_MODEL_PATH}}"
USER_VLLM_SERVED_MODEL="${USER_VLLM_SERVED_MODEL:-vllm-user}"
USER_REMOTE_PORT="${USER_REMOTE_PORT:-8588}"
USER_VLLM_GPU="${USER_VLLM_GPU:-0,1,2,3,4,5,6,7}"
USER_TENSOR_PARALLEL="${USER_TENSOR_PARALLEL:-8}"
USER_VLLM_API_KEY="${USER_VLLM_API_KEY:-EMPTY}"
USER_VLLM_EXTRA_ARGS="${USER_VLLM_EXTRA_ARGS:-${DEFAULT_VLLM_TOOL_EXTRA_ARGS} ${VLLM_EXTRA_ARGS:-}}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-196608}"
VLLM_STARTUP_TIMEOUT_SECS="${VLLM_STARTUP_TIMEOUT_SECS:-1200}"
VLLM_CHECK_INTERVAL_SECS="${VLLM_CHECK_INTERVAL_SECS:-5}"

SHARED_RUN_DIR=""
NODES_FILE=""
NODES_LOCK_FILE=""
HOSTS_FILE=""
LOCK_FILE=""
DONE_FLAG=""
RUN_ID_FILE=""
LOG_DIR=""
SESSION_STATE_FILE=""
SESSION_GROUP_FILE=""
LAST_JOIN_FILE=""
CREATED_AT_FILE=""
LOCAL_IP_SOURCE=""

LOCAL_HOSTNAME="$(hostname)"
LOCAL_IP=""

USER_PID=""

quote_shell() {
  printf "%q" "$1"
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

is_private_ipv4() {
  local ip="$1"
  [[ "${ip}" =~ ^10\. ]] && return 0
  [[ "${ip}" =~ ^192\.168\. ]] && return 0
  [[ "${ip}" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && return 0
  return 1
}

is_usable_ipv4() {
  local ip="$1"
  [[ "${ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  [[ "${ip}" =~ ^127\. ]] && return 1
  [[ "${ip}" =~ ^169\.254\. ]] && return 1
  [[ "${ip}" == "0.0.0.0" ]] && return 1
  return 0
}

pick_best_ipv4() {
  local ip
  for ip in "$@"; do
    if is_usable_ipv4 "${ip}" && is_private_ipv4 "${ip}"; then
      printf "%s" "${ip}"
      return 0
    fi
  done
  for ip in "$@"; do
    if is_usable_ipv4 "${ip}"; then
      printf "%s" "${ip}"
      return 0
    fi
  done
  return 1
}

resolve_local_ip() {
  if [[ -n "${LOCAL_IP_OVERRIDE:-}" ]]; then
    LOCAL_IP="${LOCAL_IP_OVERRIDE}"
    LOCAL_IP_SOURCE="LOCAL_IP_OVERRIDE"
    return 0
  fi

  local candidates=()
  local ip
  for ip in $(hostname -I 2>/dev/null || true); do
    candidates+=("${ip}")
  done
  if [[ "${#candidates[@]}" -gt 0 ]]; then
    if LOCAL_IP="$(pick_best_ipv4 "${candidates[@]}")"; then
      LOCAL_IP_SOURCE="hostname -I"
      return 0
    fi
  fi

  candidates=()
  while read -r ip _; do
    candidates+=("${ip}")
  done < <(getent hosts "${LOCAL_HOSTNAME}" 2>/dev/null || true)
  if [[ "${#candidates[@]}" -gt 0 ]]; then
    if LOCAL_IP="$(pick_best_ipv4 "${candidates[@]}")"; then
      LOCAL_IP_SOURCE="getent hosts"
      return 0
    fi
  fi

  return 1
}

if ! resolve_local_ip; then
  echo "[FATAL] failed to resolve local IP" >&2
  exit 1
fi

initialize_run_paths() {
  NODES_FILE="${SHARED_RUN_DIR}/nodes.txt"
  NODES_LOCK_FILE="${NODES_FILE}.lock"
  HOSTS_FILE="${SHARED_RUN_DIR}/hosts.txt"
  LOCK_FILE="${HOSTS_FILE}.lock"
  DONE_FLAG="${SHARED_RUN_DIR}/done.flag"
  RUN_ID_FILE="${SHARED_RUN_DIR}/run_id.txt"
  LOG_DIR="${SHARED_RUN_DIR}/logs"
  SESSION_STATE_FILE="${SHARED_RUN_DIR}/session.state"
  SESSION_GROUP_FILE="${SHARED_RUN_DIR}/session.group"
  LAST_JOIN_FILE="${SHARED_RUN_DIR}/last_join_at.txt"
  CREATED_AT_FILE="${SHARED_RUN_DIR}/created_at.txt"
}

ensure_parent_dir() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
}

ensure_run_dir_ready() {
  if [[ -z "${SHARED_RUN_DIR}" ]]; then
    echo "[FATAL] SHARED_RUN_DIR is empty" >&2
    exit 1
  fi
  mkdir -p "${SHARED_RUN_DIR}" "${LOG_DIR}"
  ensure_parent_dir "${NODES_FILE}"
  ensure_parent_dir "${HOSTS_FILE}"
  ensure_parent_dir "${RUN_ID_FILE}"
  ensure_parent_dir "${DONE_FLAG}"
  ensure_parent_dir "${SESSION_STATE_FILE}"
}

acquire_shared_run_dir() {
  mkdir -p "${SHARED_RUN_BASE_DIR}"

  local selected_dir
  selected_dir="$(
    (
      flock -x 200

      local now
      now="$(date +%s)"

      local candidate=""
      local candidate_last_join=0
      local dir
      shopt -s nullglob
      for dir in "${SHARED_RUN_BASE_DIR}"/run_*; do
        [[ -d "${dir}" ]] || continue

        local state="open"
        if [[ -f "${dir}/session.state" ]]; then
          state="$(head -n1 "${dir}/session.state" 2>/dev/null || true)"
        fi
        [[ "${state}" == "open" ]] || continue

        local group="default"
        if [[ -f "${dir}/session.group" ]]; then
          group="$(head -n1 "${dir}/session.group" 2>/dev/null || true)"
        fi
        [[ "${group}" == "${SHARED_RUN_GROUP_KEY}" ]] || continue

        local last_join=0
        if [[ -f "${dir}/last_join_at.txt" ]]; then
          last_join="$(head -n1 "${dir}/last_join_at.txt" 2>/dev/null || true)"
        fi
        [[ "${last_join}" =~ ^[0-9]+$ ]] || last_join=0
        if (( last_join == 0 || now - last_join > SESSION_JOIN_WINDOW_SECS )); then
          continue
        fi

        local count=0
        if [[ -f "${dir}/nodes.txt" ]]; then
          count="$(awk 'NF >= 2 {print $1}' "${dir}/nodes.txt" | sort -u | wc -l | tr -d ' ')"
        fi
        [[ "${count}" =~ ^[0-9]+$ ]] || count=0
        if (( count >= EXPECTED_NODES )); then
          continue
        fi

        if (( last_join >= candidate_last_join )); then
          candidate="${dir}"
          candidate_last_join="${last_join}"
        fi
      done
      shopt -u nullglob

      if [[ -z "${candidate}" ]]; then
        local dt
        dt="$(date +%Y%m%d_%H%M%S)"
        candidate="${SHARED_RUN_BASE_DIR}/run_${dt}"

        local suffix=1
        while [[ -e "${candidate}" ]]; do
          candidate="${SHARED_RUN_BASE_DIR}/run_${dt}_${suffix}"
          suffix=$((suffix + 1))
        done

        mkdir -p "${candidate}" "${candidate}/logs"
        if [[ ! -d "${candidate}" ]]; then
          echo "[FATAL] failed to create shared run dir ${candidate}" >&2
          exit 1
        fi
        printf "open\n" > "${candidate}/session.state"
        printf "%s\n" "${SHARED_RUN_GROUP_KEY}" > "${candidate}/session.group"
        printf "%s\n" "${now}" > "${candidate}/created_at.txt"
      fi

      mkdir -p "${candidate}" "${candidate}/logs"
      if [[ ! -d "${candidate}" ]]; then
        echo "[FATAL] shared run dir disappeared before initialization: ${candidate}" >&2
        exit 1
      fi
      touch "${candidate}/nodes.txt"
      grep -v "^${LOCAL_HOSTNAME}[[:space:]]" "${candidate}/nodes.txt" > "${candidate}/nodes.txt.tmp" || true
      mv "${candidate}/nodes.txt.tmp" "${candidate}/nodes.txt"
      echo "${LOCAL_HOSTNAME} ${LOCAL_IP}" >> "${candidate}/nodes.txt"
      printf "%s\n" "${now}" > "${candidate}/last_join_at.txt"
      printf "%s\n" "${candidate}"
    ) 200>"${SHARED_RUN_BASE_DIR}/.session.lock"
  )"

  SHARED_RUN_DIR="${selected_dir}"
  initialize_run_paths
  ensure_run_dir_ready
}

detect_platform_rank() {
  local candidate_vars=(
    RANK
    NODE_RANK
    SLURM_NODEID
    OMPI_COMM_WORLD_RANK
    PMI_RANK
    MV2_COMM_WORLD_RANK
    WORLD_RANK
  )
  local var value
  for var in "${candidate_vars[@]}"; do
    value="${!var:-}"
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
      PLATFORM_RANK="${value}"
      return 0
    fi
  done
  return 1
}

register_node_for_election() {
  ensure_run_dir_ready
  ensure_parent_dir "${NODES_LOCK_FILE}"
  (
    flock -x 200
    ensure_run_dir_ready
    touch "${NODES_FILE}"
    grep -v "^${LOCAL_HOSTNAME}[[:space:]]" "${NODES_FILE}" > "${NODES_FILE}.tmp" || true
    mv "${NODES_FILE}.tmp" "${NODES_FILE}"
    echo "${LOCAL_HOSTNAME} ${LOCAL_IP}" >> "${NODES_FILE}"
    printf "%s\n" "$(date +%s)" > "${LAST_JOIN_FILE}"
  ) 200>"${NODES_LOCK_FILE}"
}

wait_for_nodes_and_elect_rank() {
  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT_SECS )); do
    register_node_for_election
    if [[ -f "${NODES_FILE}" ]]; then
      local count
      count="$(awk 'NF >= 2 {print $1}' "${NODES_FILE}" | sort -u | wc -l | tr -d ' ')"
      if [[ "${count}" -ge "${EXPECTED_NODES}" ]]; then
        local sorted_file
        sorted_file="${NODES_FILE}.sorted"
        awk 'NF >= 2 {print $1, $2}' "${NODES_FILE}" | sort -u > "${sorted_file}"
        local idx=0
        local host ip
        while read -r host ip; do
          if [[ "${host}" == "${LOCAL_HOSTNAME}" ]]; then
            RANK="${idx}"
            return 0
          fi
          idx=$((idx + 1))
        done < "${sorted_file}"
      fi
    fi
    sleep "${VLLM_CHECK_INTERVAL_SECS}"
    waited=$(( waited + VLLM_CHECK_INTERVAL_SECS ))
  done
  return 1
}

resolve_rank() {
  if detect_platform_rank; then
    RANK="${PLATFORM_RANK}"
    return 0
  fi
  wait_for_nodes_and_elect_rank
}

wait_for_run_id() {
  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT_SECS )); do
    if [[ -f "${RUN_ID_FILE}" ]]; then
      local line
      line="$(head -n1 "${RUN_ID_FILE}" 2>/dev/null || true)"
      if [[ -n "${line}" ]]; then
        RUN_ID="${line}"
        return 0
      fi
    fi
    sleep "${VLLM_CHECK_INTERVAL_SECS}"
    waited=$(( waited + VLLM_CHECK_INTERVAL_SECS ))
  done
  return 1
}

register_host() {
  local run_id="$1"
  local role="$2"
  local ip="$3"
  local port="$4"
  ensure_run_dir_ready
  ensure_parent_dir "${LOCK_FILE}"
  (
    flock -x 200
    ensure_run_dir_ready
    touch "${HOSTS_FILE}"
    grep -v "^${run_id}[[:space:]]${role}[[:space:]]" "${HOSTS_FILE}" > "${HOSTS_FILE}.tmp" || true
    mv "${HOSTS_FILE}.tmp" "${HOSTS_FILE}"
    echo "${run_id} ${role} ${ip} ${port}" >> "${HOSTS_FILE}"
  ) 200>"${LOCK_FILE}"
}

wait_for_role_entry() {
  local run_id="$1"
  local role="$2"
  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT_SECS )); do
    if [[ -f "${HOSTS_FILE}" ]]; then
      local line
      line="$(awk -v run_id="${run_id}" -v target="${role}" '$1 == run_id && $2 == target { print $0; exit }' "${HOSTS_FILE}" 2>/dev/null || true)"
      if [[ -n "${line}" ]]; then
        printf "%s" "${line}"
        return 0
      fi
    fi
    sleep "${VLLM_CHECK_INTERVAL_SECS}"
    waited=$(( waited + VLLM_CHECK_INTERVAL_SECS ))
  done
  return 1
}

wait_for_local_url() {
  local url="$1"
  local waited=0
  while (( waited < VLLM_STARTUP_TIMEOUT_SECS )); do
    if curl --noproxy localhost,127.0.0.1 -fs "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${VLLM_CHECK_INTERVAL_SECS}"
    waited=$(( waited + VLLM_CHECK_INTERVAL_SECS ))
  done
  return 1
}

cleanup_user() {
  if [[ -n "${USER_PID}" ]] && ps -p "${USER_PID}" >/dev/null 2>&1; then
    kill "${USER_PID}" || true
    wait "${USER_PID}" || true
  fi
}

close_rank0_session() {
  if [[ "${RANK:-}" == "0" && -n "${SESSION_STATE_FILE:-}" ]]; then
    ensure_parent_dir "${SESSION_STATE_FILE}"
    printf "closed\n" > "${SESSION_STATE_FILE}"
  fi
}

acquire_shared_run_dir
trap close_rank0_session EXIT

if ! resolve_rank; then
  echo "[FATAL] failed to determine local rank" >&2
  exit 1
fi

echo "[INFO] shared_run_base_dir=${SHARED_RUN_BASE_DIR}"
echo "[INFO] shared_run_dir=${SHARED_RUN_DIR}"
echo "[INFO] local_ip_source=${LOCAL_IP_SOURCE}"
if [[ -n "${DT:-}" ]]; then
  echo "[INFO] dt=${DT}"
fi
echo "[INFO] hostname=${LOCAL_HOSTNAME} ip=${LOCAL_IP} rank=${RANK}"

if [[ "${RANK}" == "0" ]]; then
  ensure_run_dir_ready
  rm -f "${DONE_FLAG}"
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$_${LOCAL_HOSTNAME}}"
  printf "%s\n" "${RUN_ID}" > "${RUN_ID_FILE}"
else
  if ! wait_for_run_id; then
    echo "[FATAL] rank=${RANK} timed out waiting for ${RUN_ID_FILE}" >&2
    exit 1
  fi
fi

if [[ "${RANK}" == "1" ]]; then
USER_LOG_FILE="${LOG_DIR}/user_vllm_${LOCAL_IP}_${USER_REMOTE_PORT}.log"
  echo "[INFO] rank=1 starting user vLLM on ${LOCAL_IP}:${USER_REMOTE_PORT}"
  extra_args=()
  if [[ -n "${USER_VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    parsed_extra=( ${USER_VLLM_EXTRA_ARGS} )
    extra_args+=("${parsed_extra[@]}")
  fi

  max_model_len_args=()
  if ! args_contain_flag "--max-model-len" "${extra_args[@]}"; then
    max_model_len_args=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
  fi

  CUDA_VISIBLE_DEVICES="${USER_VLLM_GPU}" \
  vllm serve "${USER_VLLM_MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${USER_REMOTE_PORT}" \
    --served-model-name "${USER_VLLM_SERVED_MODEL}" \
    --tensor-parallel-size "${USER_TENSOR_PARALLEL}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    "${max_model_len_args[@]}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    "${extra_args[@]}" \
    > "${USER_LOG_FILE}" 2>&1 &
  USER_PID=$!
  trap cleanup_user EXIT

  if ! wait_for_local_url "http://127.0.0.1:${USER_REMOTE_PORT}/health"; then
    echo "[FATAL] rank=1 user vLLM failed to become healthy. log=${USER_LOG_FILE}" >&2
    exit 1
  fi

  register_host "${RUN_ID}" "user" "${LOCAL_IP}" "${USER_REMOTE_PORT}"
  echo "[INFO] rank=1 registered user endpoint ${LOCAL_IP}:${USER_REMOTE_PORT} into ${HOSTS_FILE} for run_id=${RUN_ID}"

  while [[ ! -f "${DONE_FLAG}" ]]; do
    if ! ps -p "${USER_PID}" >/dev/null 2>&1; then
      echo "[FATAL] rank=1 user vLLM exited unexpectedly. log=${USER_LOG_FILE}" >&2
      exit 1
    fi
    sleep 30
  done
  echo "[INFO] rank=1 detected done flag, exiting"
  exit 0
fi

if [[ "${RANK}" != "0" && "${RANK}" != "1" ]]; then
  echo "[FATAL] only rank 0 and rank 1 are supported for this workflow. got rank=${RANK}" >&2
  exit 1
fi

if [[ -z "${AGENT_VLLM_MODEL_PATH:-}" ]]; then
  echo "[FATAL] AGENT_VLLM_MODEL_PATH is required on rank 0" >&2
  exit 1
fi

echo "[INFO] rank=0 waiting for user endpoint in ${HOSTS_FILE}"
USER_LINE="$(wait_for_role_entry "${RUN_ID}" "user")" || {
  echo "[FATAL] timed out waiting for user endpoint in ${HOSTS_FILE}" >&2
  exit 1
}

USER_HOST_IP="$(echo "${USER_LINE}" | awk '{print $3}')"
USER_HOST_PORT="$(echo "${USER_LINE}" | awk '{print $4}')"
if [[ -z "${USER_HOST_IP}" || -z "${USER_HOST_PORT}" ]]; then
  echo "[FATAL] malformed user entry in ${HOSTS_FILE}: ${USER_LINE}" >&2
  exit 1
fi

echo "[INFO] rank=0 discovered user endpoint ${USER_HOST_IP}:${USER_HOST_PORT}"
if ! curl --connect-timeout 2 -fs "http://${USER_HOST_IP}:${USER_HOST_PORT}/health" >/dev/null 2>&1; then
  echo "[FATAL] rank=0 cannot reach user endpoint http://${USER_HOST_IP}:${USER_HOST_PORT}/health" >&2
  echo "[INFO] shared hosts file: ${HOSTS_FILE}" >&2
  exit 1
fi

echo "[INFO] rank=0 launching benchmark on node A"
set +e
cd "${ROOT_DIR}"
DUAL_VLLM_SPLIT_LOCAL_NODE=0 \
AGENT_MODE=vllm \
AGENT_VLLM_MODEL_PATH="${AGENT_VLLM_MODEL_PATH}" \
VLLM_MODEL_PATH="${AGENT_VLLM_MODEL_PATH}" \
USER_MODE=vllm \
USER_VLLM_AUTOSTART=0 \
USER_VLLM_BASE_URL="http://${USER_HOST_IP}:${USER_HOST_PORT}/v1" \
USER_MODEL="${USER_MODEL:-openai/${USER_VLLM_SERVED_MODEL}}" \
USER_VLLM_API_KEY="${USER_VLLM_API_KEY}" \
bash "${INNER_SCRIPT}" "${MODE}"
RC=$?
set -e

touch "${DONE_FLAG}"
exit "${RC}"
