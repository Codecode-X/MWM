# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_transformer | 5 | 126767 | 0.9360 ± 0.1322 | 0.9347 ± 0.1282 | 0.1007 ± 0.0703 | 0.8800 ± 0.1366 | 0.7111 ± 0.3478 | 1.0000 ± 0.0000 | 0.6400 ± 0.4099 |
| latent_ode | 5 | 32727 | 0.4160 ± 0.1438 | 0.3846 ± 0.1452 | 0.2252 ± 0.0023 | 0.5333 ± 0.1247 | 0.1067 ± 0.1535 | 0.0971 ± 0.1364 | 0.1200 ± 0.1789 |
| ode_rnn | 5 | 15877 | 0.8160 ± 0.0921 | 0.7667 ± 0.1328 | 0.2197 ± 0.0009 | 0.7467 ± 0.0558 | 0.3810 ± 0.2586 | 0.7500 ± 0.4330 | 0.2800 ± 0.2280 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
