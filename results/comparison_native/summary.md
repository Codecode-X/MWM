# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_transformer | 5 | 126767 | 0.9120 ± 0.0867 | 0.8821 ± 0.1072 | 0.1308 ± 0.0632 | 0.8267 ± 0.0760 | 0.6521 ± 0.2158 | 0.9100 ± 0.1245 | 0.5600 ± 0.2608 |
| latent_ode | 5 | 32727 | 0.3480 ± 0.1474 | 0.3770 ± 0.1569 | 0.2230 ± 0.0003 | 0.4133 ± 0.1193 | 0.2083 ± 0.2244 | 0.1558 ± 0.1625 | 0.3200 ± 0.3633 |
| ode_rnn | 5 | 15877 | 0.8120 ± 0.2480 | 0.8078 ± 0.2124 | 0.2238 ± 0.0056 | 0.8400 ± 0.1211 | 0.7110 ± 0.2602 | 0.8250 ± 0.2437 | 0.6800 ± 0.3347 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
