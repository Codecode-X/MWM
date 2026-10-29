# Three-model classification comparison

Mean ± sample standard deviation (ddof=1) across independent training seeds, fixed patient split.
Threshold selected on validation F1. AUROC/AP do not depend on the threshold.

| Model | Seeds | Parameters | AUROC | AP | Brier | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_transformer | 4 | 126767 | 0.5250 ± 0.2009 | 0.5546 ± 0.2092 | 0.2373 ± 0.0040 | 0.4000 ± 0.0544 | 0.4819 ± 0.0723 | 0.3377 ± 0.0434 | 0.8500 ± 0.1915 |

This estimates training randomness on one fixed split; it is not cross-validation or a clinical generalization estimate.
Native auxiliary losses and parameter counts differ; use --objective bce for a shared classification-loss ablation.
