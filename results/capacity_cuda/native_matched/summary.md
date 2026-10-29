# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent_ode | 5 | 126766 | 0.5240 ± 0.0713 | 0.4514 ± 0.0758 | 0.2269 ± 0.0052 | 0.4667 ± 0.1944 | 0.3504 ± 0.0715 | 0.4315 ± 0.3264 | 0.4400 ± 0.2191 |
| ode_rnn | 5 | 126997 | 0.9720 ± 0.0390 | 0.9602 ± 0.0571 | 0.2140 ± 0.0091 | 0.8533 ± 0.0869 | 0.7398 ± 0.1520 | 0.9200 ± 0.1789 | 0.6400 ± 0.1673 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
