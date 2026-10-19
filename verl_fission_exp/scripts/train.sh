#!/bin/bash
# VERL GRPO Training Script
# 功能：单节点 RLHF 训练

set -euo pipefail

if [[ "${TRACE_TRAIN_SH:-0}" == "1" ]]; then
    set -x
fi

# 环境准备：创建统一的目录结构
if [ ! -d "/mnt/nlp-ali" ] && [ -n "${DATA_ROOT:-}" ]; then
    mkdir -p /mnt/nlp-ali
    cd /mnt/nlp-ali
    ln -s "${DATA_ROOT}" usr
    ln -s "${DATA_ROOT}" nas
    cd /mnt/nlp-ali/usr
fi

# ==============================
# 基础配置
# ==============================
project_folder="${PROJECT_FOLDER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ==============================
# Wandb / Proxy / API 配置
# ==============================
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="qwen_rlhf_verl_multi_turn"
# Set https_proxy/http_proxy externally if needed.

export XHS_MAAS_API_KEY="${XHS_MAAS_API_KEY:-}"
export XHS_MAAS_BASE_URL="${XHS_MAAS_BASE_URL:-https://your-openai-compatible-endpoint.example.com/v1}"
export XHS_MAAS_MODEL="gpt-4.1"

# ==============================
# 训练环境变量（融合自你截图中的“环境配置”）
# 如果外部已经传入，则优先使用外部值；否则使用这里的默认值
# ==============================
export datetime="${datetime:-20260417101866}"
export ori_checkpoint_name="${ori_checkpoint_name:-Qwen3-8B}"

# 注意：这里的路径在你的截图里被截断了，需要你替换成真实完整模型路径
export ori_checkpoint_path="${ori_checkpoint_path:-/path/to/base/model}"

export lr="${lr:-1e-6}"
export gpu_memory_utilization="${gpu_memory_utilization:-0.7}"
export temperature="${temperature:-0.95}"
export top_k="${top_k:-50}"

# correction 相关配置
# 0表示范围在-3，3之间，1表示范围在-1到1之间
export CORRECTMAX1="${CORRECTMAX1:-0}"
export corr_acc_threshold="${corr_acc_threshold:-0.6}"
export dedup_enable="${dedup_enable:-True}"
export corr_min_interval_steps="${corr_min_interval_steps:-4}"

# ==============================
# 时间戳打印
# ==============================
echo "datetime = ${datetime}"
echo "system time = $(date +"%Y%m%d%H%M%S")"

# ==============================
# 输出目录
# ==============================
output_dir="${OUTPUT_DIR:-${project_folder}/outputs/${ori_checkpoint_name}_rlhf_grpo_${datetime}}"
log_dir="${LOG_DIR:-${project_folder}/logs/${ori_checkpoint_name}_rlhf_grpo_${datetime}}"

# 如果输出路径是一个目录，而且目录非空，也直接退出
if [ -d "$output_dir" ]; then
    if [ "$(ls -A "$output_dir")" ]; then
        echo "[INFO] 检测到已有非空输出目录：$output_dir"
        echo "[INFO] 认为训练已完成/无需重复，脚本退出。"
        exit 0
    fi
fi

mkdir -p "$output_dir"
mkdir -p "$log_dir"

# ==============================
# 加载 verl 所需环境
# ==============================
# source /mnt/nlp-ali/usr/envs/anaconda3/bin/activate verl_zxz

cd "$project_folder"
pip install -e .

cd "$project_folder/scripts"

export BASE_MODEL="${ori_checkpoint_path}"
export EXPERIMENT_NAME="grpo_qwen3_8b_multi_turn_data_1325sample_${datetime}"
export VLLM_USE_V1=1

TRAIN_FILES_JSON="${TRAIN_FILES_JSON:-[\"/path/to/train.parquet\"]}"
VAL_FILES_JSON="${VAL_FILES_JSON:-[\"/path/to/test.parquet\"]}"

ROW_COUNT=$(python3 - <<'PY' "$TRAIN_FILES_JSON"
import sys, json
import pyarrow.parquet as pq

files = json.loads(sys.argv[1])
total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
print(total)
PY
)

echo "[info] train row count = ${ROW_COUNT}"

SAVE_FREQ=$(( ROW_COUNT / 8 ))
if [ "$SAVE_FREQ" -le 0 ]; then
    SAVE_FREQ=1
fi
echo "[info] computed save_freq = ${SAVE_FREQ}"

export HYDRA_FULL_ERROR=1
export RAY_DISABLE_DASHBOARD=1

log_file=${log_dir}/${EXPERIMENT_NAME}_train_log.log

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES_JSON" \
    data.val_files="$VAL_FILES_JSON" \
    data.train_batch_size=8 \
    data.return_raw_chat=True \
    data.max_prompt_length=12800 \
    data.max_response_length=4096 \
    +algorithm.correction.enable=True \
    +algorithm.correction.dedup_enable=${dedup_enable} \
    +algorithm.correction.acc_threshold=${corr_acc_threshold} \
    +algorithm.correction.min_interval_steps=${corr_min_interval_steps} \
    +algorithm.correction.batch_size=8 \
    +algorithm.correction.fixed_error_message='error: you made a mistake. please try again.' \
    actor_rollout_ref.rollout.max_model_len=18000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH:-${project_folder}/verl/utils/reward_score/multi_turn_reward_v2_dynamic_v3_no_0_strict.py}" \
    custom_reward_function.name=compute_score \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.rollout_data_dir=${log_dir}/$EXPERIMENT_NAME \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=100000 \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.default_local_dir=${output_dir} \
    2>&1 | tee "$log_file"

train_exit_code=$?

echo "========== Training Completed =========="
echo "Exit code: $train_exit_code"
echo "Model saved to: $output_dir"
echo "Logs saved to: $log_dir"

exit $train_exit_code
