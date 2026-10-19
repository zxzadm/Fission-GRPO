# $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment

## 备注：

# Tau2 测试脚本说明

这份说明主要总结仓库里常用的测试脚本、关键参数、结果输出位置，以及几个最重要的指标，方便在日常调试和正式评测时快速查阅。

## 入口脚本

- `scripts/run_tau2_three_domains_qs.sh`
  单机脚本，依次运行 `retail`、`airline`、`telecom` 三个领域。
- `scripts/run_tau2_three_domains_qs_two_node_shared_hosts.sh`
  双机脚本。
  `rank=0` 负责启动 benchmark 和 agent vLLM。
  `rank=1` 负责启动 user vLLM，并通过共享目录注册服务地址。
- 一般测试都是在两个节点下启动，其中一个节点负责部署 agent LLM，另一个节点负责部署 user LLM，不走外部 API，方便进行大批量测试。当前常见的 user 侧部署模型为 `Qwen3-235B-A22B-Instruct-2507`。

## 运行模式

- `smoke`
  快速测试模式。不传位置参数时默认就是这个模式。
- `full`
  全量测试模式。

示例：

```bash
bash scripts/run_tau2_three_domains_qs.sh smoke
bash scripts/run_tau2_three_domains_qs_two_node_shared_hosts.sh full
```

## 关键参数

- `NUM_TRIALS`
  每个 task 运行的 trial 次数。默认是 `1`。如果要测试 `pass^k` 指标，需要把 `NUM_TRIALS` 设置为大于等于 `k`。
- `MAX_CONCURRENCY`
  单个领域内部的 task 并发数。默认是 `3`。
- `PARALLEL_DOMAINS`
  `0` 表示按 `retail -> airline -> telecom` 串行运行。
  `1` 表示三个领域并行运行。
- `TASK_SPLIT`
  默认是 `base`。
- `NUM_TASKS`
  只在 `smoke` 模式下生效。默认是 `5`。
- `MAX_STEPS`
  默认是 `200`。
- `MAX_ERRORS`
  默认是 `10`。
- `KEY_RETRY_LIMIT`
  领域级重试次数。默认是 `1`。
- `RETRY_RESUME`
  默认是 `1`，重试时会复用同一个 `save_to`。
- `DT`
  用于控制结果目录名中的 `tau2_bench_<DT>`。
- `AGENT_VLLM_MODEL_PATH`
  agent 模型路径。脚本会从这个路径取最后一级目录名，作为结果输出目录中的模型名。
- `BENCH_OUTPUT_BASE_DIR`
  统一结果输出根目录，默认是：

```text
<BENCH_OUTPUT_BASE_DIR>
```

- `AGENT_VLLM_EXTRA_ARGS`
  仅传给 agent 侧 `vllm serve` 的额外参数。
- `USER_VLLM_EXTRA_ARGS`
  仅传给 user 侧 `vllm serve` 的额外参数。

推荐的稳定配置：

```bash
NUM_TRIALS=3 MAX_CONCURRENCY=5 PARALLEL_DOMAINS=0 bash scripts/run_tau2_three_domains_qs_two_node_shared_hosts.sh full
```

## 常用 vLLM 参数

- `VLLM_MAX_MODEL_LEN`
  脚本默认值是 `65536`，但有些模型只支持 `40960`。
  如果 vLLM 启动时报 `max_model_len` 校验错误，需要显式调小，例如：

```bash
VLLM_MAX_MODEL_LEN=40960
```

- `AGENT_VLLM_MODEL_PATH`
  agent 走本地 vLLM 时需要配置的模型路径。
- `USER_VLLM_MODEL_PATH`
  user 侧模型路径。
- `AGENT_VLLM_EXTRA_ARGS`
  只作用于 agent 侧 vLLM，例如可以单独给 agent 增加 rope scaling 或更长上下文。
- `USER_VLLM_EXTRA_ARGS`
  只作用于 user 侧 vLLM，不会和 agent 共用。
- `LOCAL_IP_OVERRIDE`
  多网卡机器上如果自动选 IP 选错了，可以用这个变量强制指定本机地址。

## 双机共享目录

双机脚本每次运行都会在 `SHARED_RUN_BASE_DIR` 下新建一个独立的运行目录，里面通常会有：

- `nodes.txt`
- `hosts.txt`
- `run_id.txt`
- `done.flag`
- `logs/user_vllm_<ip>_<port>.log`

脚本启动时会打印：

- `shared_run_base_dir=...`
- `shared_run_dir=...`
- `local_ip_source=...`
- `dt=...`

双机模式出问题时，优先看这些信息。

## 结果文件

每个领域的原始结果 JSON 会写到：

```text
<BENCH_OUTPUT_BASE_DIR>/<模型名>/compass_outputs/tau2_bench_<DT>/simulations/
```

命名格式通常是：

```text
qs_<timestamp>_<domain>_<mode>_<agent_tag>_user-<user_tag>.json
```

例如：

```text
qs_20260312_144225_airline_smoke_Qwen3_8B_user-Qwen3-235B-A22B-Instruct-2507.json
```

其中：

- `<模型名>` 默认从 `AGENT_VLLM_MODEL_PATH` 的最后一级目录名提取
- `<DT>` 由环境变量 `DT` 控制

例如当：

```bash
DT=20260323
AGENT_VLLM_MODEL_PATH=/path/to/your/model
```

结果会写到：

```text
<BENCH_OUTPUT_BASE_DIR>/Qwen3-8B/compass_outputs/tau2_bench_20260323/simulations/
```

## 自动汇总 CSV

- 备注：该功能为后加入功能，原始 `tau2-bench` 并没有该实现，主要是为了方便统一查看评测结果和指标。

每次 `scripts/run_tau2_three_domains_qs.sh` 跑完后，会自动调用：

```text
scripts/export_run_summary_csv.py
```

然后把本次 run 的汇总 CSV 写到：

```text
<BENCH_OUTPUT_BASE_DIR>/<模型名>/compass_outputs/tau2_bench_<DT>/summaries/
```

汇总文件名通常类似：

```text
qs_<timestamp>_<mode>_<agent_tag>_user-<user_tag>.csv
```

这份 CSV 是“一次 run 一行”的汇总表，会把三个领域的结果展开，常见字段包括：

- `*_tasks`
- `*_simulations`
- `*_success_count`
- `*_success_rate`
- `*_avg_reward`
- `*_avg_agent_cost`
- `*_avg_user_cost`
- `*_pass_hat_k`
- `*_termination_summary`
- `*_file`

如果不想自动导出，可以设置：

```bash
AUTO_EXPORT_RUN_SUMMARY=0
```

如果想改汇总目录，可以设置：

```bash
RUN_SUMMARY_DIR=/your/custom/summary/dir
```

## 指标说明

当前框架里最核心的几个指标是：

- `avg_reward`
  一个结果文件中所有 simulation 的 reward 平均值。
- `pass_hat_k`
  先按 task 计算，再在 task 维度上求平均。
  成功的定义是 `reward` 约等于 `1.0`。
- `avg_agent_cost`
  agent 侧平均 cost。

关于 `pass_hat_k` 需要注意：

- 它依赖 `NUM_TRIALS`。
- `NUM_TRIALS=1` 时只会有 `pass_hat_1`。
- `NUM_TRIALS=3` 时才可能有 `pass_hat_1`、`pass_hat_2`、`pass_hat_3`。
- 不要直接比较不同 `NUM_TRIALS` 运行下的 `pass_hat_3` 之类的高阶指标。

## 手动查看结果

- 交互式查看结果：

```bash
tau2 view
```

- 汇总多次历史运行结果：

```bash
python scripts/summarize_runs_to_csv.py
```

## 实用建议

- 调试稳定性时，优先使用 `PARALLEL_DOMAINS=0`。
- vLLM 服务没有完全稳定前，不要一开始就把 `MAX_CONCURRENCY` 调太高。
- 如果要比较不同模型的 `pass_hat_k`，必须固定 `NUM_TRIALS`。
- 如果终端显示“跑完了”但结果全是 0，先不要直接相信终端摘要，优先检查 per-domain JSON 和双机共享目录下的日志。



## 下面为原readme.md的内容

[![python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](http://img.shields.io/badge/cs.AI-arXiv%3A2506.07982-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.07982)
[![blog](https://img.shields.io/badge/blog-tau2--bench-green)](https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/sierra.svg?style=social&label=Follow%20%40SierraPlatform)](https://x.com/SierraPlatform/status/1932464265207889974)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/sierra_last-year-we-introduced-%F0%9D%9C%8F-bench-a-benchmark-activity-7338229693898231809-F8L4?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAdc8goBmhEsiEo1_t_XSJbAnY4_zMfAWcE)
[![Leaderboard](https://img.shields.io/badge/🏆_Live_Leaderboard-taubench.com-brightgreen?style=flat)](https://taubench.com)

<div align="center">
<img src="figs/overview.png" width="95%" alt="System Overview"><br>
<em>Figure 1: τ²-bench allows users to interact with the agent and the environment</em>
</div>

<div align="center">
<img src="figs/traj.png" width="95%" alt="Trajectory"><br>
<em>Figure 2: Trajectory of a conversation between an agent and a user</em>
</div>

## 🆕 What's New

### 🤖 Reinforcement Learning Support (New!)
τ²-bench now supports RL training with a Gymnasium-compatible interface:

- **🏋️ Train RL Agents**: Use the gym interface to train agents with popular RL frameworks. 
- **🎮 Play as Agent or User**: Interactive mode lets you control either the agent or the user in conversations
- **📊 Train/Test Splits**: To help support experiments around training Agents and evaluating them, all domains include standardized task splits for proper train/test evaluation.

> **⚠️ IMPORTANT FOR BACKWARD COMPATIBILITY**: If you are just evaluating an agent (not training), you **MUST** use the `base` task split to evaluate on the complete task set that matches the original τ²-bench structure. This ensures your results are comparable to previous evaluations and maintains consistency with the established benchmark. (If you don't specify a task split, it will default to `base`.)
- **🔧 Gymnasium Compatible**: Standard gym interface works with existing RL tools and libraries

[**→ See Gym Documentation**](src/tau2/gym/README.md) | [**→ Try CLI Play Mode**](#interactive-play-mode)

### 🏆 Live Leaderboard (v0.2.0)
The τ²-bench leaderboard is now live at **[taubench.com](https://taubench.com)**! 

- **📊 Interactive Rankings**: Compare model performance across all domains
- **📱 Mobile-Friendly**: View results on any device  
- **🔍 Detailed Analysis**: Explore trajectories and conversation flows
- **📥 Easy Submission**: Submit your results directly through the interface

[**→ Visit the Leaderboard**](https://taubench.com) | [**→ Submit Your Results**](#leaderboard-submission)

## Overview

$\tau^2$-bench implements a simulation framework for evaluating customer service agents across various domains.

**$\tau^2$-bench is the new iteration of the original $\tau$-bench**, featuring code fixes and an additional telecom domain.

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance
- Optionally: A set of tools that the user simulator can use

Domains are:
- `mock`
- `airline`
- `retail`
- `telecom`

All the information that an agent developer needs to build an agent for a domain can be accessed through the domain's API docs. See [View domain documentation](#view-domain-documentation) for more details.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

2. Create a new environment (optional)

$\tau^2$-bench requires Python 3.10 or higher. You may create and activate a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install tau2

```bash
pip install -e .
```

This will enable you to run the `tau2` command.

**Note:** If you use `pip install .` (without `-e`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:

```bash
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

**Check your data directory setup:**

After installation, you can verify that your data directory is correctly configured by running:

```bash
tau2 check-data
```

This command will check if the data directory exists and print instructions if it is missing.

To remove all the generated files and the virtual environment, run:
```bash
make clean
```

## Quick Start

### Setup LLM API keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

To provide your API keys, copy `.env.example` as `.env` and edit it to include your API keys.

### Run agent evaluation

To run a test evaluation on only 5 tasks with 1 trial per task, run:

```bash
tau2 run \ 
--domain airline \
--agent-llm gpt-4.1 \
--user-llm gpt-4.1 \
--num-trials 1 \
--num-tasks 5
```

Results will be saved in `data/tau2/simulations/`.

> **💡 Tip**: For full agent evaluation that matches the original τ²-bench methodology, remove `--num-tasks` and use `--task-split base` to evaluate on the complete task set.

## Command Line Interface

The `tau2` command provides a unified interface for all functionality:

### Running Benchmark 
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --task-ids <task_ids> \
  --max-concurrency <concurrent_sims> \
  ...
```

### Interactive Play Mode
```bash
tau2 play
```
Experience τ²-bench from either perspective! The play mode allows you to:
- **Play as Agent**: Manually control the agent's responses and tool calls
- **Play as User**: Control the user while an LLM agent handles requests (available in domains with user tools like telecom)
- **Understand tasks** by walking through scenarios step-by-step
- **Test strategies** before implementing them in code
- **Choose task splits** to practice on training data or test on held-out tasks

This is perfect for:
- Getting familiar with domain policies and tools from both perspectives
- Debugging task scenarios and conversation flows
- Developing intuition for agent strategies
- Testing user behavior and agent responses
- Training yourself before training your model!

See the [Gym Documentation](src/tau2/gym/README.md) for more details on using the gymnasium interface programmatically, including the `AgentGymEnv` (play as agent) and `UserGymEnv` (play as user).

### Viewing Results
```bash
tau2 view
```
This tool allows you to:
- Browse simulation files (in `data/tau2/simulations/`)
- View agent performance metrics
- View a particular simulation
- View task details

### View domain documentation
```bash
tau2 domain <domain>
```
Visit http://127.0.0.1:8004/redoc to see the domain policy and API documentation.

![domain_viewer1](figs/domain_viewer.png)

### Check data configuration
```bash
tau2 check-data
```
This command checks if your data directory is properly configured and all required files are present.

## Leaderboard Submission

To submit your agent results to the τ²-bench leaderboard, you need to prepare a valid submission package that meets specific requirements.

### Requirements for Valid Submissions

Your trajectory runs must follow these constraints:

1. **Complete domain coverage**: Include results for all three domains:
   - `retail`
   - `airline` 
   - `telecom`

2. **Consistent model configuration**: All trajectory files must use:
   - The same agent LLM with identical arguments across all domains
   - The same user simulator LLM with identical arguments across all domains

3. **One result per domain**: Each domain should appear exactly once in your submission

4. **All tasks completed**: Run evaluation on all tasks within each domain (don't use `--task-ids` or `--num-tasks` filters)

> **📝 Note**: For consistency with the original τ²-bench evaluation methodology, use the `base` task split when evaluating your agent to ensure you're testing on the complete, standard task set.

### Preparing Your Submission

#### Step 1: Run Evaluations
First, run your agent evaluation on all domains with consistent settings:

```bash
# Example: Run complete evaluation for all domains
tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_airline  
tau2 run --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_telecom
```

**Important**: Use identical `--agent-llm`, `--user-llm`, and their arguments across all runs.

#### Step 2: Prepare Submission Package
Use the submission preparation tool to create your leaderboard submission:

```bash
tau2 submit prepare data/tau2/simulations/my_model_*.json --output ./my_submission
```

This command will:
- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email)
- Create a structured submission directory with:
  - `submission.json`: Metadata and metrics
  - `trajectories/`: Your trajectory files

#### Step 3: Validate Your Submission
Before submitting, validate your submission package:

```bash
tau2 submit validate ./my_submission
```

This will verify:
- All required files are present
- Trajectory files are valid
- Domain coverage is complete
- Model configurations are consistent

### Additional Options

#### Skip Verification (if needed)
```bash
tau2 submit prepare data/tau2/simulations/my_model_*.json --output ./my_submission --no-verify
```

#### Verify Individual Trajectory Files
```bash
tau2 submit verify-trajs data/tau2/simulations/my_model_*.json
```

### Submitting to the Leaderboard

Once your submission package is prepared and validated:

1. Review the generated `submission.json` file
2. Follow the submission guidelines in [web/leaderboard/public/submissions/README.md](web/leaderboard/public/submissions/README.md) to create a Pull Request
3. Keep your `trajectories/` directory for reference

The leaderboard will display your model's Pass^k success rates (k=1,2,3,4) across all domains.

## Experiments

### Experimental Code Directory

The `@experiments/` directory contains experimental features and research code that extends beyond the core tau2 benchmark. This directory is designed for community contributions of innovative approaches, prototypes, and new features that are not part of the core evaluation framework.

- **Purpose**: Research code and experimental features
- **Location**: `src/experiments/`
- **Usage**: Each experimental component has its own README with documentation
- **Status**: Experimental code is provided as-is and may not be fully tested or supported

For more details, see the [experiments README](src/experiments/README.md).

### Running Ablation Studies (No User, or Agent with Oracle Plan)
`telecom` domain enables running ablation studies.

1. Running an LLM in `no-user` mode. In this mode, the LLM is given all the tools and the information upfront.
Just choose `llm_agent_solo` as the agent and `dummy_user` as the user.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --agent-llm gpt-4.1 \
  --user dummy_user \
  ...
```

2. Running an LLM in `oracle-plan` mode. In this mode, the LLM is given an oracle plan ahead of time alleviating the need for action planning.
Just choose `llm_agent_gt` as the agent.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

### Running Telecom Domain with Workflow Policy
To test the impact of policy format, we provide an additional "workflow" policy for the telecom domain.
To run using this policy, use the `telecom-workflow` domain.

```bash
tau2 run \
  --domain telecom-workflow \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

## Domains

For all the details see the domains [README](src/tau2/domains/README.md).

### Basics

- Code is located in `src/tau2/domains/`
- Data is located in `data/tau2/domains/`
- Each domain has its own configuration and task definitions

#### View domain-specific policy and API docs:
Run the following command to see the domain policy and API documentation.
```bash
tau2 env <domain>
```

Then visit http://127.0.0.1:8004/redoc

### Environment CLI (beta)

An interactive command-line interface for directly querying and testing domain environments. Features:
- Interactive query interface with domain-specific tools
- Support for multiple domains (airline, mock, etc.)
- Session management with history

To use:
```bash
make env-cli
```

Available commands:
- `:q` - quit the program
- `:d` - change domain
- `:n` - start new session (clears history)

Example usage:
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to airline domain.

Query (:n new session, :d change domain, :q quit)> What flights are available from SF to LA tomorrow?
Assistant: Let me check the flight availability for you...
[Flight details will appear here]
```

The Environment CLI is useful for:
- Testing domain tools and queries
- Debugging environment responses
- Exploring available domain functionality
- Quick domain interaction without starting the full server stack


## Run tests
To run the test suite use the command

```sh
make test
```

## Config

To configure the framework, see the [config](src/tau2/config.py) file.

### LLM Calls caching
LLM call caching is disabled by default.

To enable LLM calls caching:
    - Make sure `redis` is running.
    - Update the redis config in `config.py` if necessary.
    - Set `LLM_CACHE_ENABLED` to `True` in `config.py`


## Evaluate Your Own Agent
For local or remote agent evaluation, see our [agent developer guide](src/tau2/agent/README.md).

## Contributing

We welcome contributions to τ²-bench! Whether you're fixing bugs, adding new features, creating new domains, or contributing experimental research code, please see our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines on:

- **Opening issues** before starting work
- **Branch naming conventions** and development workflow  
- **Code quality standards** and testing requirements
- **Pull request guidelines** for clean, reviewable contributions
- **Domain and experimental contributions** specific guidelines

For experimental features and research code, check out the [`@experiments/`](src/experiments/) directory.

## Orchestration Sequence Diagram

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant A as Agent
    participant U as UserSimulator
    participant E as Environment

    Note over O: Initialize(task)
    rect rgb(100, 150, 150)
        O->>A: get_init_state_info(message_history)
        A->>O: agent_state_info
        O->>U: get_init_state_info(message_history)
        U->>O: user_state_info
        O->>E: set_state(initialization_data, initialization_actions, message_history)
    end
    Note over O: Start simulation
    loop Pass messages between Agent, User, and Environment

        alt Agent/Env to User
            rect rgb(200, 150, 150)
            O->>U: generate_next_message(msg, user_state_info)
            U-->>O: (user_msg, user_state_info)
            end
            Note over O: Check if user_msg is STOP
        else User/Env to Agent
            rect rgb(100, 200, 100)
            O->>A: generate_next_message(msg, agent_state_info)
            A-->>O: (assistant_msg, agent_state_info)
            Note over O: Check if too many errors
            end
        else User/Agent to Environment
            rect rgb(150, 150, 200)
            O->>E: get_response(tool_call)
            E-->>O: tool_message
            end
        end
        Note over O: Check if max turns reached.
    end
    Note over O: Return simulation run
```

## Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```
