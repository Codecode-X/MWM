# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_transformer | 5 | 126767 | 0.9600 ± 0.0894 | 0.9126 ± 0.1955 | 0.0810 ± 0.0686 | 0.9200 ± 0.1193 | 0.8700 ± 0.1857 | 0.9200 ± 0.1789 | 0.8400 ± 0.2191 |
| latent_ode | 5 | 32727 | 0.3200 ± 0.1449 | 0.3020 ± 0.0553 | 0.2263 ± 0.0033 | 0.3600 ± 0.1116 | 0.2508 ± 0.1791 | 0.1944 ± 0.1406 | 0.3600 ± 0.2608 |
| ode_rnn | 5 | 15877 | 0.8280 ± 0.0832 | 0.7909 ± 0.1078 | 0.2191 ± 0.0013 | 0.7600 ± 0.0760 | 0.3976 ± 0.2832 | 0.8000 ± 0.4472 | 0.2800 ± 0.2280 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
