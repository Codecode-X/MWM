# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent_ode | 5 | 126766 | 0.4200 ± 0.1643 | 0.4096 ± 0.1947 | 0.2253 ± 0.0030 | 0.4933 ± 0.0894 | 0.2887 ± 0.1933 | 0.2483 ± 0.1659 | 0.3600 ± 0.2608 |
| ode_rnn | 5 | 126997 | 0.7720 ± 0.0934 | 0.6556 ± 0.1294 | 0.2199 ± 0.0017 | 0.6933 ± 0.0596 | 0.2000 ± 0.1826 | 0.6000 ± 0.5477 | 0.1200 ± 0.1095 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
