# Synthetic ICU experiments

Validation MSE selects the checkpoint. MSE is in training-set min-max units.

| Experiment | Best epoch | Initial val MSE | Best val MSE | Test MSE | Test AUROC | Test AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| results/gpu_paper_default_check/latent_ode_extrapolation | 1 | 0.296434 | 0.182662 | 0.188953 | — | — |
| results/gpu_paper_default_check/latent_ode_interpolation | 1 | 0.300288 | 0.184735 | 0.184277 | — | — |
| results/gpu_paper_default_check/ode_rnn_extrapolation | 1 | 0.323320 | 0.202785 | 0.206888 | — | — |
| results/gpu_paper_default_check/ode_rnn_interpolation | 1 | 0.297583 | 0.180888 | 0.179137 | — | — |
