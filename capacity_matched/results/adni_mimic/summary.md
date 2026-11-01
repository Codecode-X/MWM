# ADNI与MIMIC：三种基线的同预算实验

ADNI按44,875参数匹配，MIMIC按43,280参数匹配。六组新实验按空闲GPU排队，所有任务均为单卡训练，使用ode环境和SwanLab。

| 作业 | 状态 | 阶段 | 物理GPU |
|---|---|---|---:|
| adni_ode_ae | complete | tested | 0 |
| mimic_ode_ae | complete | tested | 0 |
| adni_ode_rnn | complete | tested | 1 |
| mimic_ode_rnn | running | training | 1 |
| adni_latent_ode | running | training | 3 |
| mimic_latent_ode | running | training | 0 |

- [ADNI：原容量与同预算指标](../../../paper_odeae/results/baselines_matched_latent2x/summary.md)
- [MIMIC：原容量与同预算指标](../../../mimic_experiment/results/baselines_matched_latent2x/summary.md)

[设计与参数分配](../../README.zh-CN.md) · [队列](queue.json) · [六组CUDA预检查](architecture_checks.json)
