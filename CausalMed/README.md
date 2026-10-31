# CausalMed：Level 3 连续时间反事实基准

γ=1动作条件模型实验包括latent2x both、ODE-AE、确定性ODE-AE和HealthFlux-KL，执行相同五策略反事实推演。前两组已完成；新增确定性ODE-AE在物理GPU 1按31,314参数匹配，HealthFlux-KL在物理GPU 3保留原生256＋128维。见[方法与运行说明](latent2x_experiment/README.zh-CN.md)和[模型横向比较](outputs/gamma1_model_comparison/summary.md)。

代码位于 [synthetic_cf_level3](synthetic_cf_level3/README.md)。新增内容全部限制在 CausalMed 内，原模型、shared_matched 数据代码和历史实验未改动。

从项目根目录运行：

```bash
conda activate ode
python -m pip install -r CausalMed/requirements.txt
export PYTHONPATH="$PWD/CausalMed${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
pytest -c CausalMed/pytest.ini CausalMed/synthetic_cf_level3/tests -q
python -m synthetic_cf_level3.generate --smoke --out-dir CausalMed/outputs/smoke_100
python -m synthetic_cf_level3.generate --all-gammas --out-dir CausalMed/outputs/level3_suite
```

`--smoke` 使用总计 100 位患者（60/20/20），正式配置每组 10,000/2,000/2,000；四组正式基准总计 56,000 位患者。输出目录必须是新目录，生成器拒绝覆盖已有数据。

本任务没有模型实验。SciPy 数值积分属于数据准备；后续模型训练、推理和模型诊断必须遵守项目 AGENTS.md，使用 ode 环境和物理 GPU 0/1/2/3，禁止 CPU fallback。

实际交付的正式四组数据位于 `outputs/level3_suite/gamma_{0,0.5,1,2}/`，使用 `synthetic_cf_level3/configs/level3_v1_calibrated.yaml`。基础配置仍保留要求的默认 κ=2；最终配置共同固定 **κ=3**，仅由500位训练患者的候选成本模拟结果选定，校准证据在 `outputs/training_cost_calibration.json`。最终配置的100人验收在 `outputs/smoke_100_calibrated/`。

完整复现（写入一个新的输出目录）：

```bash
bash CausalMed/reproduce.sh /absolute/path/to/new_output
```

仅重建与当前交付一致的四组最终数据：

```bash
PYTHONPATH=CausalMed OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n ode python -m synthetic_cf_level3.generate \
  --config CausalMed/synthetic_cf_level3/configs/level3_v1_calibrated.yaml \
  --all-gammas --out-dir CausalMed/outputs/level3_suite_replica
```

每组的 `diagnostics/validation_report.md`、`statistics.json` 和 `plots/` 是对应验收报告与七类诊断图。`environment_versions.json` 与 `requirements-lock.txt` 记录实际软件版本。公开加载器只接收每组 `public/` 路径；模型工作区应只挂载该目录。

`patient_id` 在单个 gamma 数据集内唯一；四个独立总体使用相同的编号形式。若跨目录合并记录，请用 `(gamma, patient_id)` 作联合标识。

为让模型加载接口时无需安装包含真实方程的生成器包，另提供 `model_client/causalmed_client` 最小客户端。模型环境只复制/挂载 **model_client/ + 对应 public/**，通过 `from causalmed_client import PublicDataset, decide` 读取；其余生成源码、config、manifest、diagnostics、oracle 都留在数据管理员环境。
