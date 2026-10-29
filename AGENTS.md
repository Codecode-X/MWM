# Project execution requirements

- User requirement: all subsequent model experiments must run on the available four NVIDIA RTX 4090 GPUs. Do not train, validate, evaluate checkpoints, or run model diagnostics on CPU, and do not silently fall back to CPU.
- Use CUDA by default. Run independent model/seed jobs across physical GPUs 0, 1, 2, 3; prefer one worker per GPU unless task requirements justify otherwise.
- Keep CPU use to orchestration, data preparation, file I/O, statistical summaries, and small mathematical/unit checks. These are not substitutes for GPU model experiments.
- Record actual tensor devices and physical GPU assignments in experiment artifacts. Keep historical CPU results clearly labeled and separate from new GPU results.
- Use the existing Conda `ode` environment. Typical comparison invocation: `DEVICE=cuda JOBS=4 EPOCHS=20 OUTPUT=results/comparison_cuda bash scripts/run_comparison.sh --gpus 0 1 2 3`.
