> 后续实验必须使用四张 RTX 4090：训练、验证、检查点评估及模型诊断默认 CUDA，禁止回退 CPU。CPU 仅用于数据准备、I/O、统计汇总和小型单元测试。下文历史 CPU 结果保留原标签。

# Latent ODE 与 ODE-RNN 论文复现

训练监测统一接入 SwanLab，默认使用本地看板。启动、云端配置和既有任务接入方式见
[SwanLab 使用说明](reproduction/SWANLAB.zh-CN.md)。

ADNI 多模态纵向预测实验的特征提取、冻结 MedicalNet 编码器、患者切分和 GPU 训练协议见
[ADNI 实验说明](reproduction/ADNI_EXPERIMENT.zh-CN.md)。结果与图表保存于
[ADNI GPU 比较](results/adni_cuda/summary.md)。

本地用药、干预记录能否支持动作预测与反事实研究，见
[ADNI 治疗相关研究方案](reproduction/ADNI_TREATMENT_PLAN.zh-CN.md)。文档包含原始表覆盖与日期审计、任务定义、因果分析条件和分阶段实施路线。

全部正式纵向患者的观测次数直方图，以及年龄、性别、诊断和用药分布，见
[ADNI 人群统计报告](results/adni_population/summary.zh-CN.md)，同时提供原始总名册的补充统计、逐次数 CSV 和可导出的图表。

论文 ODE-AE、确定性版本与 LSTM 的独立复现工作区位于
[paper_odeae/](paper_odeae/README.zh-CN.md)，其中包含数据合同、模型、四 GPU 运行入口和实现对照说明。

依据本仓库 `paper_latentODE.pdf`（Rubanova、Chen、Duvenaud，arXiv:1907.03907v1）。当前 `src/` 实现两种基础模型：

- **Latent ODE**：ODE-RNN 编码器 → 高斯初始状态后验 → 重参数采样 → 独立生成 ODE → 逐时刻解码；用 ELBO 训练。
- **ODE-RNN**：观测间用 ODE 演化隐藏状态，在观测处用 GRU 更新；采用因果自回归高斯似然训练。

原 Transformer / Augmented ODE / evidential 实现已备份到 `legacy/`。历史 `models/`、根目录 `results/` 的旧指标不属于本次复现。

## 运行

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ode
python -m unittest discover -s tests -v

# 两种模型 × 插值/外推，训练、验证选模、独立测试、重新载入检查点评估
EPOCHS=20 bash scripts/run_paper_experiments.sh

# 可选：利用现有 adverse_event 标签联合训练，普通 BCE 系数 100
EPOCHS=20 bash scripts/run_paper_classification.sh
```

也可使用 `PYTHON=/path/to/python OUTPUT=results/my_run` 指定解释器、设备和输出目录。脚本默认训练 30 轮，参数可追加，如 `--seed 43 --batch-size 10`。所有命令从仓库根目录执行；bash 脚本会自动切换目录。

单个实验与检查点评估：

```bash
python -m src.train --model latent_ode --task extrapolation \
  --epochs 20 --output results/my_experiment
python -m src.evaluate results/my_experiment/best.pt --split test
python -m src.train --help
```

每个实验目录保存 `config.json`、`data.json`（训练集归一化参数和患者划分）、`history.json`、`best.pt`、`metrics.json` 和训练日志。`best.pt` 只按验证集 MSE 选择；测试集在训练结束后使用。

详细论文对照、实验协议、测试范围及实测结果见 [复现说明](reproduction/PAPER_REPRODUCTION.zh-CN.md)。

## 三模型多随机种子对比

旧 Transformer + Augmented ODE、Latent ODE、ODE-RNN 使用相同患者划分、验证 AUROC 选模和稳定概率评估，报告多次独立训练的均值与样本标准差：

```bash
EPOCHS=20 JOBS=4 OUTPUT=results/comparison_native_cuda \
  bash scripts/run_comparison.sh --seeds 0 1 2 3 4
```

统一 BCE 对照、旧模型适配修改、断点跳过与统计边界见 [对比实验协议](reproduction/COMPARISON.zh-CN.md)。

四张 RTX 4090 按独立任务分配的命令（使用新输出目录）：

```bash
DEVICE=cuda JOBS=4 EPOCHS=20 OUTPUT=results/comparison_cuda \
  bash scripts/run_comparison.sh --gpus 0 1 2 3
```

此前完成的正式比较使用 CPU，属于历史记录；GPU 验证独立保存，后续实验仅使用 GPU。5 个训练种子 × 20 轮的正式结果见 [原生损失](results/comparison_native/summary.md)、[统一 BCE](results/comparison_bce/summary.md) 和 [对比图](results/comparison_overview.png)。

## 参数量匹配的 GPU 对比

扩容前 Git 检查点：`e3752ce`（标签 `baseline-before-capacity-scaling-20261020`）。
Latent ODE 扩大到 126,766 参数，ODE-RNN 扩大到 126,997 参数，旧模型为 126,767 参数。
使用 `conda activate ode` 后运行：

```bash
JOBS=8 EPOCHS=20 bash scripts/run_capacity_comparison.sh
```

脚本在四张 4090 上重新训练大小模型和旧模型，比较原生损失与统一 BCE，默认五个种子。
[实验说明](reproduction/CAPACITY_EXPERIMENT.zh-CN.md)；[结果](results/capacity_cuda/capacity_summary.md)。
