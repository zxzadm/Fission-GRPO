# Fission-GRPO: Robust Tool Use via Learning to Recover from Execution Errors

<p align="center">
  <a href="https://arxiv.org/abs/2601.15625"><img src="https://img.shields.io/badge/arXiv-2601.15625-b31b1b.svg" alt="arXiv"></a>
  <a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2026-blue.svg" alt="ACL 2026"></a>
</p>

> [!IMPORTANT]
> This repository is being prepared for the official open-source release of **Fission-GRPO**.  
> The codebase and training data are currently under internal review and will be uploaded once the approval process is completed.

This repository currently serves as the **official project page** for the ACL 2026 paper:

> **Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors**  
> Zhiwei Zhang, Fei Zhao, Rui Wang, Zezhong Wang, Bin Liang, Jiakang Wang, Yao Hu, Shaosheng Cao, Kam-Fai Wong

---

## Overview

**Fission-GRPO** is a training framework for improving **multi-turn tool-use robustness**.  
Instead of treating execution errors as purely negative rewards, it converts failed trajectories into **on-policy corrective supervision**, enabling the model to learn how to recover from its own mistakes during training.

At a high level, the framework works in three stages:

1. **Standard Exploration:** the policy is optimized with GRPO on multi-turn tool-use trajectories.  
2. **Error Identification:** failed or low-quality trajectories are intercepted and augmented with diagnostic feedback.  
3. **Fission-Based Update:** these corrective contexts are replayed as additional updates, helping the model learn recovery behavior.

---

## Release Status

The full open-source release is currently under internal review. Once approved, we plan to update this repository with:

- training code
- data processing scripts
- synthetic training data
- error simulator checkpoints
- evaluation scripts
- example configs and launch scripts

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


