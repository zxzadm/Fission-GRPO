# Fission-GRPO: Robust Tool Use via Learning to Recover from Execution Errors

<p align="center">
  <a href="https://arxiv.org/abs/2601.15625"><img src="https://img.shields.io/badge/arXiv-2601.15625-b31b1b.svg" alt="arXiv"></a>
  <a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2026-blue.svg" alt="ACL 2026"></a>
</p>

> [!IMPORTANT]
> This repository is being prepared for the official open-source release of **Fission-GRPO**.  
> The codebase, training data, model checkpoints, and evaluation scripts are currently under internal review and will be uploaded once the approval process is completed.

This repository currently serves as the **official project page** for the ACL 2026 paper:

> **Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors**  
> Zhiwei Zhang, Fei Zhao, Rui Wang, Zezhong Wang, Bin Liang, Jiakang Wang, Yao Hu, Shaosheng Cao, Kam-Fai Wong

---

## Overview

**Fission-GRPO** is a training framework for improving **multi-turn tool-use robustness**.  
Instead of treating execution errors as purely negative reward signals, Fission-GRPO converts failed trajectories into **on-policy corrective supervision** inside the RL loop, enabling the model to learn not only which actions are wrong, but also how to recover from its own mistakes.

The framework is motivated by a key limitation of standard RL for tool use: when a tool call fails, the model typically receives only a sparse penalty, without any direct supervision about how to correct the failure. Fission-GRPO addresses this by transforming failure cases into new corrective training instances during exploration.

At a high level, the framework works as follows:

1. The policy is optimized with GRPO on multi-turn tool-use trajectories.
2. Failed or low-quality trajectories are intercepted by a correction pipeline.
3. Diagnostic feedback is attached to the failed context to form a corrective example.
4. These corrective examples are replayed as additional on-policy updates, helping the model learn recovery behavior.

---

## Key Idea

The central idea of **fission** is to turn a single execution failure into a source of additional supervision.

Rather than relying on a fixed offline error-correction dataset, Fission-GRPO continuously mines new correction examples from the model's current failure modes during training. This produces a dynamic stream of corrective trajectories that stays aligned with the evolving policy distribution.

This design has two important benefits:

- **On-policy alignment:** the model learns from the types of errors it is currently making.
- **Corrective supervision:** failures are not only penalized, but also converted into learning opportunities for recovery.

---

## Method Framework

Fission-GRPO consists of three stages:

### 1. Standard Exploration
The policy interacts with the environment under the GRPO objective and generates multi-turn tool-use trajectories.

### 2. Error Identification and Feedback Construction
Failed or low-quality trajectories are filtered and transformed into corrective contexts by appending diagnostic feedback.

### 3. Fission-Based Corrective Update
These corrective contexts are replayed as additional updates, allowing the model to explicitly learn recovery strategies from its own execution errors.

---

## Release Status

At this stage, this repository serves as the official project page.

The full open-source release is currently under internal review. Once approval is completed, we plan to update this repository with:

- training code
- data processing scripts
- synthetic training data
- error simulator checkpoints
- evaluation scripts
- example configs and launch scripts
- documentation for reproduction

---

## Planned Contents

The full release is intended to support the following use cases:

- reproducing the training workflow described in the paper
- understanding how the corrective stream is integrated into GRPO training
- studying the role of dynamic feedback in multi-turn error recovery
- adapting the framework to new tool-use benchmarks or alternative feedback backends

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


Contact

If you have questions about the paper or the upcoming release, please feel free to open an issue in this repository.


