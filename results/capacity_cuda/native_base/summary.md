# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_transformer | 5 | 126767 | 0.9680 ± 0.0610 | 0.9600 ± 0.0723 | 0.1337 ± 0.0508 | 0.9200 ± 0.0298 | 0.8692 ± 0.0674 | 0.9333 ± 0.0913 | 0.8400 ± 0.1673 |
| latent_ode | 5 | 32727 | 0.4920 ± 0.1942 | 0.4342 ± 0.1589 | 0.2230 ± 0.0014 | 0.4667 ± 0.0943 | 0.2981 ± 0.2338 | 0.2500 ± 0.1667 | 0.4400 ± 0.4336 |
| ode_rnn | 5 | 15877 | 0.9320 ± 0.0879 | 0.9075 ± 0.1079 | 0.2234 ± 0.0060 | 0.8267 ± 0.0760 | 0.6854 ± 0.1599 | 0.8433 ± 0.1507 | 0.6000 ± 0.2000 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
