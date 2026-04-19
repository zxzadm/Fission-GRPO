# Fission-GRPO: Robust Tool Use via Learning to Recover from Execution Errors

<p align="center">
  <a href="https://arxiv.org/abs/2601.15625"><img src="https://img.shields.io/badge/arXiv-2601.15625-b31b1b.svg" alt="arXiv"></a>
  <a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2026-blue.svg" alt="ACL 2026"></a>
</p>

> [!IMPORTANT]
> This repository is being prepared for the official open-source release of **Fission-GRPO**.  
> The codebase and training data are currently under internal review and will be uploaded once the approval process is completed.

This repository contains the official project page for the ACL 2026 paper:

> **Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors**  
> Zhiwei Zhang, Fei Zhao, Rui Wang, Zezhong Wang, Bin Liang, Jiakang Wang, Yao Hu, Shaosheng Cao, Kam-Fai Wong
---

## Overview

**Fission-GRPO** is a dynamic training framework for multi-turn tool-use robustness. Unlike standard RL approaches that treat execution errors purely as negative reward signals, Fission-GRPO converts failed trajectories into corrective supervision *within* the RL loop — enabling the model to learn not only which actions are wrong, but concretely how to recover from its own mistakes.

A key advantage of this framework is its **dynamic data generation** capability: rather than relying on a fixed supervision set, Fission-GRPO continuously mines new correction examples from the base dataset *throughout training*, producing a self-expanding stream of corrective trajectories that keeps the training signal fresh and diverse as the model evolves.

The training pipeline proceeds as follows:

1. The policy is optimized via GRPO on multi-turn tool-use rollouts.
2. Failed or low-quality trajectories are intercepted by the correction pipeline.
3. A diagnostic error message is appended to each failed trajectory, constructing a correction example.
4. These augmented trajectories are buffered, deduplicated, and replayed as additional correction batches.

---

## Repository Structure

The main entry point for training is:

```
train.sh          # Main RL training launch script
```

Training is launched via `python3 -m verl.trainer.main_ppo` with the correction stream enabled by default.

---

## Error Feedback Modes

Fission-GRPO supports two modes for providing corrective feedback signals during training.

### Mode 1: Fixed Fallback Feedback

The simplest mode uses a fixed error string as the feedback signal:

```
error: you made a mistake. please try again.
```

This is suitable for establishing a stable baseline or verifying that the correction pipeline is functioning correctly, without requiring any external model.

### Mode 2: LLM-Based Error Simulator

For richer feedback, Fission-GRPO supports an LLM-based error simulator that generates dynamic, context-aware corrective messages on the fly via any OpenAI-compatible API endpoint.

> **Note:** A dedicated SFT-trained error simulator, fine-tuned specifically to produce diagnostic feedback for tool-use failures, yields the strongest performance. If such a checkpoint is not available, a general-purpose LLM API prompted with an appropriate system prompt can serve as a practical substitute and still produce meaningful improvements over the fixed fallback.

The feedback model is configured through the following environment variables:

| Variable | Description |
|---|---|
| `XHS_MAAS_BASE_URL` | Base URL for the OpenAI-compatible API endpoint |
| `XHS_MAAS_API_KEY` | Primary API key |
| `BFCL_SIMULATOR_API_KEY` | Fallback API key if primary is unavailable |
| `XHS_MAAS_MODEL` | Model identifier (e.g., `gpt-4.1`) |

If the external feedback model is unavailable or fails at any point, the framework gracefully reverts to the fixed fallback message, ensuring that training is never blocked.

---

## Prerequisites

Before launching training, ensure the following are available:

- A base model checkpoint
- Training and validation data in verl RL parquet format
- An output directory and a log directory
- A valid reward function file

### Required Environment Variables

| Variable | Description |
|---|---|
| `PROJECT_FOLDER` | Root project directory |
| `ori_checkpoint_path` | Path to the base model checkpoint |
| `TRAIN_FILES_JSON` | JSON list of training parquet file paths |
| `VAL_FILES_JSON` | JSON list of validation parquet file paths |
| `OUTPUT_DIR` | Directory for model outputs and checkpoints |
| `LOG_DIR` | Directory for training logs |
| `CUSTOM_REWARD_FUNCTION_PATH` | Path to the reward function `.py` file |

### Optional Environment Variables

| Variable | Description |
|---|---|
| `WANDB_API_KEY` | Weights & Biases API key (required if using wandb logging) |
| `XHS_MAAS_BASE_URL` | API base URL for the LLM-based error simulator |
| `XHS_MAAS_API_KEY` | Primary API key for the error simulator |
| `BFCL_SIMULATOR_API_KEY` | Fallback API key for the error simulator |
| `XHS_MAAS_MODEL` | Model name for the error simulator |

---

## Quick Start

```bash
export PROJECT_FOLDER=/path/to/verl_fission_exp
export ori_checkpoint_path=/path/to/base/model
export TRAIN_FILES_JSON='["/path/to/train.parquet"]'
export VAL_FILES_JSON='["/path/to/test.parquet"]'
export OUTPUT_DIR=/path/to/output_dir
export LOG_DIR=/path/to/log_dir
export CUSTOM_REWARD_FUNCTION_PATH=/path/to/reward_file.py

# Optional: LLM-based error simulator
export XHS_MAAS_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
export XHS_MAAS_API_KEY=your_api_key
export XHS_MAAS_MODEL=gpt-4.1

bash train.sh
```

---

## Notes on the Correction Pipeline

- Correction examples are collected from failed or low-quality rollouts and deduplicated before replay.
- A correction batch is only triggered once a sufficient number of unique correction examples has accumulated; skipped steps when this threshold is not met are expected behavior.
- Intermediate correction parquet files are written under `OUTPUT_DIR` for inspection and reproducibility.
- The pipeline degrades gracefully: if LLM-based feedback is unavailable at any point, the fixed fallback is used and training continues uninterrupted.

---

## Scope of This Release

This codebase is intended to support:

- Reproducing the training workflow described in the paper
- Understanding the integration of the correction stream within GRPO training
- Adapting the framework to new tool-use benchmarks or alternative error-feedback backends

> **Important:** The provided `train.sh` contains placeholder paths and is not intended to run as-is. Please set all required environment variables before launching training. In particular, verify that `CUSTOM_REWARD_FUNCTION_PATH` resolves correctly in your environment.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026robust,
  title={Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors},
  author={Zhang, Zhiwei and Zhao, Fei and Wang, Rui and Wang, Zezhong and Liang, Bin and
          Wang, Jiakang and Hu, Yao and Cao, Shaosheng and Wong, Kam-Fai},
  journal={arXiv preprint arXiv:2601.15625},
  year={2026}
}
```
