# MWM：ADNI与MIMIC实验

全部未完成MWM训练已暂停，不会自动恢复。主方法为latent2x both；GPU用于恢复同参数量baseline。

每组使用独立单卡，无梯度累积。ADNI主比较batch128，新增A/B的batch32对照；MIMIC主版本batch128，MIMIC A/B保持暂停。

| 作业 | batch | 状态 | 阶段 | GPU |
|---|---:|---|---|---:|
| adni_main | 128 | complete | tested | 0 |
| adni_A | 128 | complete | tested | 1 |
| adni_B | 128 | complete | tested | 3 |
| adni_A_batch32 | 32 | paused_by_user | user_paused | 0 |
| adni_B_batch32 | 32 | paused_by_user | user_paused | 1 |
| mimic_main | 128 | paused_by_user | user_paused | 2 |
| mimic_A | 128 | paused_by_user | user_paused | — |
| mimic_B | 128 | paused_by_user | user_paused | — |

- [ADNI结果](../../../paper_odeae/results/mwm/summary.md)
- [MIMIC结果](../../../mimic_experiment/results/mwm/summary.md)

[队列](queue.json) · [CUDA检查](architecture_checks.json) · [原始固定配置](manifest.json) · [新增对照配置](additional_jobs.json)
