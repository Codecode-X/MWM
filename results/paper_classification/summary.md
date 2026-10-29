# Synthetic ICU experiments

Validation MSE selects the checkpoint. MSE is in training-set min-max units.

| Experiment | Best epoch | Initial val MSE | Best val MSE | Test MSE | Test AUROC | Test AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| results/paper_classification/latent_ode | 19 | 0.275829 | 0.016674 | 0.017582 | 0.7200 | 0.6190 |
| results/paper_classification/ode_rnn | 19 | 0.300488 | 0.020061 | 0.019840 | 1.0000 | 1.0000 |
