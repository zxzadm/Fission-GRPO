# Fission-GRPO

This directory contains the main training entrypoint for the code release of our ACL 2026 paper:

**Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors**

Fission-GRPO is a training framework for improving multi-turn tool-use robustness. Instead of treating execution errors only as negative rewards, the framework converts failed trajectories into corrective supervision inside the RL loop. In practice, the training pipeline:

- runs standard GRPO exploration on tool-use trajectories,
- identifies failed or low-quality rollouts,
- appends diagnostic error feedback to construct correction examples,
- replays those examples as additional correction batches.

This repository version is intended to make the training procedure understandable and reproducible for follow-up research on robust tool use and execution-error recovery.

## Main Entry

The main script in this directory is:

- `train.sh`

It launches RL training through `python3 -m verl.trainer.main_ppo` and enables the correction stream used by Fission-GRPO.

## Overview of the Training Pipeline

The public training entry follows the same high-level idea as the paper:

1. The policy is optimized with GRPO on the original multi-turn tool-use data.
2. Incorrect trajectories are intercepted by the correction pipeline.
3. A diagnostic error message is appended after the model's incorrect response.
4. These augmented trajectories are buffered and replayed as correction batches.

This design lets the model learn not only which trajectories are bad, but also how to recover from its own execution mistakes.

## Error Feedback Modes

Fission-GRPO supports two practical ways to provide correction feedback.

### 1. Fixed fallback feedback

The simplest mode uses a fixed fallback error string:

```text
error: you made a mistake. please try again.
```

This is useful when:

- you want a stable baseline,
- you only want to verify that the correction pipeline is functioning,
- you do not have a separate error simulator model.

### 2. External LLM as the error simulator

If you do not have a dedicated SFT error simulator, you can use an OpenAI-compatible API model to generate corrective feedback on the fly.

The current code reads:

- `XHS_MAAS_BASE_URL`
- `XHS_MAAS_API_KEY`
- `BFCL_SIMULATOR_API_KEY`
- `XHS_MAAS_MODEL`

The effective logic is:

- use `XHS_MAAS_API_KEY` if available,
- otherwise fall back to `BFCL_SIMULATOR_API_KEY`,
- if the external feedback model is unavailable or fails, revert to the fixed fallback message.

This means a general-purpose API model can be used as the feedback model during training, which is useful for public reproduction when the original simulator checkpoint is not released.

## What You Need Before Running

At minimum, you should provide:

- a base model checkpoint path,
- training and validation parquet files in the expected verl RL format,
- an output directory,
- a log directory,
- a valid reward function path.

In practice, the following environment variables are the ones most users will need to set explicitly:

- `PROJECT_FOLDER`
- `ori_checkpoint_path`
- `TRAIN_FILES_JSON`
- `VAL_FILES_JSON`
- `OUTPUT_DIR`
- `LOG_DIR`
- `CUSTOM_REWARD_FUNCTION_PATH`

Optional but often useful:

- `WANDB_API_KEY`
- `XHS_MAAS_BASE_URL`
- `XHS_MAAS_API_KEY`
- `BFCL_SIMULATOR_API_KEY`
- `XHS_MAAS_MODEL`

## Quick Start

```bash
export PROJECT_FOLDER=/path/to/verl_fission_exp
export ori_checkpoint_path=/path/to/base/model
export TRAIN_FILES_JSON='["/path/to/train.parquet"]'
export VAL_FILES_JSON='["/path/to/test.parquet"]'
export OUTPUT_DIR=/path/to/output_dir
export LOG_DIR=/path/to/log_dir
export CUSTOM_REWARD_FUNCTION_PATH=/path/to/reward_file.py

# Optional: external error simulator
export XHS_MAAS_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
export XHS_MAAS_API_KEY=...
export XHS_MAAS_MODEL=gpt-4.1

bash train.sh
```

## Correction-Related Notes

The current training script enables the correction stream by default. A few points are important for understanding its behavior:

- correction examples are collected from failed or low-quality trajectories,
- correction samples are deduplicated before being replayed,
- replay is triggered only after enough unique correction samples have accumulated,
- correction samples and intermediate correction parquet files are written under the training output directory,
- if LLM-generated feedback is unavailable, training continues with the fixed fallback message.

This fallback behavior is intentional: the correction mechanism should degrade gracefully rather than block the full RL run.

## Important Practical Notes

- The public `train.sh` contains placeholder paths. It is not intended to run as-is without setting your own model and data paths.
- You should explicitly verify `CUSTOM_REWARD_FUNCTION_PATH` in your environment before launching training.
- If you use `trainer.logger=['console','wandb']`, you should normally set `WANDB_API_KEY`.
- If correction samples are too few or become too repetitive after deduplication, a correction step may be skipped. This is expected behavior.
- The external feedback model is optional. If it is unavailable, the trainer falls back to the fixed message instead of aborting the run.

## Scope of This Release

This directory is meant to document and expose the training-side implementation of Fission-GRPO, centered around the public `train.sh` entrypoint. It is primarily intended for:

- reproducing the general training workflow,
- understanding how the correction stream is integrated into GRPO training,
- adapting the framework to new tool-use datasets or new error-feedback backends.

## Citation

If you find this code useful, please cite:

```bibtex
@article{zhang2026robust,
  title={Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors},
  author={Zhang, Zhiwei and Zhao, Fei and Wang, Rui and Wang, Zezhong and Liang, Bin and Wang, Jiakang and Hu, Yao and Cao, Shaosheng and Wong, Kam-Fai},
  journal={arXiv preprint arXiv:2601.15625},
  year={2026}
}
```
