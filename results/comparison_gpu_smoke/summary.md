# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_transformer | 2 | 126767 | 0.4800 ± 0.0849 | 0.3831 ± 0.0653 | 0.2357 ± 0.0003 | 0.4000 ± 0.0000 | 0.4706 ± 0.0000 | 0.3333 ± 0.0000 | 0.8000 ± 0.0000 |
| latent_ode | 2 | 32727 | 0.4700 ± 0.0990 | 0.4655 ± 0.1969 | 0.2356 ± 0.0063 | 0.4000 ± 0.0943 | 0.3571 ± 0.1010 | 0.2778 ± 0.0786 | 0.5000 ± 0.1414 |
| ode_rnn | 2 | 15877 | 0.7900 ± 0.2404 | 0.7364 ± 0.2919 | 0.2351 ± 0.0024 | 0.8000 ± 0.0943 | 0.6500 ± 0.2121 | 0.7333 ± 0.0943 | 0.6000 ± 0.2828 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
