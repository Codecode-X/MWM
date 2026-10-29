# Synthetic ICU experiments

Validation MSE selects the checkpoint. MSE is in training-set min-max units.

| Experiment | Best epoch | Initial val MSE | Best val MSE | Test MSE | Test AUROC | Test AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| results/paper/latent_ode_extrapolation | 11 | 0.272257 | 0.017897 | 0.020271 | — | — |
| results/paper/latent_ode_interpolation | 19 | 0.275828 | 0.016686 | 0.017853 | — | — |
| results/paper/ode_rnn_extrapolation | 20 | 0.323320 | 0.017114 | 0.020470 | — | — |
| results/paper/ode_rnn_interpolation | 19 | 0.297583 | 0.020438 | 0.021403 | — | — |
| results/paper_classification/latent_ode | 19 | 0.275829 | 0.016674 | 0.017582 | 0.7200 | 0.6190 |
| results/paper_classification/ode_rnn | 19 | 0.300488 | 0.020061 | 0.019840 | 1.0000 | 1.0000 |
