> 新执行规则：所有后续模型实验使用四张 RTX 4090；CPU 不作为训练、评估或诊断设备。本文既有 CPU 表格属于规则生效之前的历史记录，新 GPU 实验使用独立输出目录。

# 三模型公平分类对比

对比任务为完整患者记录上的 `adverse_event` 二分类。旧模型没有同任务的完整轨迹监督，因此此脚本不将旧单点重建 MSE 与论文模型的轨迹 MSE 混在一起比较。

## 运行

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ode

# 默认：5 个训练种子 0/1/2/3/4，50 轮，四张 RTX 4090、4 个任务进程
bash scripts/run_comparison.sh

# 固定同一协议，4 个独立模型/种子任务进程分别使用四张 GPU，每个模型 20 轮
EPOCHS=20 JOBS=4 OUTPUT=results/comparison_native_cuda \
  bash scripts/run_comparison.sh --seeds 0 1 2 3 4

# 统一纯分类 BCE 的独立消融实验；必须使用不同输出目录
EPOCHS=20 JOBS=4 OUTPUT=results/comparison_bce_cuda \
  bash scripts/run_comparison.sh --seeds 0 1 2 3 4 --objective bce
```

`PYTHON=/path/to/ode/bin/python` 可替代环境激活；默认 `DEVICE=cuda JOBS=4` 使用四张 GPU，未指定 GPU 时选择物理卡 0/1/2/3。CPU 实验入口已禁用。并行参数按模型/训练种子组合划分进程，不共享模型、优化器或随机状态。CPU 内存不足时减少 `JOBS`。

为保证不同模型的同条件对比，一个输出目录的协议不允许改变。修改模型代码后应使用新目录；重新运行前指定新目录，或使用相同参数加 `--resume` 跳过已完成的训练。中断且没有 `metrics.json` 的训练不续接优化器状态，应在新目录重新训练，避免把不完整训练混入统计。

```bash
# 所有参数必须与原运行相同；jobs 可改变
EPOCHS=20 JOBS=4 OUTPUT=results/comparison_native_cuda \
  bash scripts/run_comparison.sh --seeds 0 1 2 3 4 --resume

# 只载入保存的最佳权重，重新输出验证/测试预测，检查验证重放一致性
python -m src.comparison --output results/comparison_native_cuda \
  --epochs 20 --seeds 0 1 2 3 4 --reevaluate --jobs 4
```

其他配置见 `python -m src.comparison --help`。改动 `--split-seed` 后必须换输出目录。

## 四张 RTX 4090 的分配与核验

```bash
# unset CUDA_VISIBLE_DEVICES 后使用物理卡号；4 个进程分别持有一张卡
DEVICE=cuda JOBS=4 EPOCHS=20 OUTPUT=results/comparison_cuda \
  bash scripts/run_comparison.sh --gpus 0 1 2 3 --seeds 0 1 2 3 4

nvidia-smi
```

每个 worker 在首次 CUDA 初始化之前只暴露分配给它的一张物理 GPU，因此该进程中的逻辑设备都是 `cuda:0`。每个运行的 `metrics.json.runtime` 记录实际张量设备、`CUDA_VISIBLE_DEVICES` 和 GPU 型号；不能只看逻辑 `cuda:0` 就误认为所有任务在同一张物理卡。

多卡采用独立模型/种子任务并行，不采用 DDP。CUDA 启用确定性算法，并关闭非确定性 Flash/Memory-efficient attention，使用 math attention。四卡验证结果单独保存在 `results/comparison_gpu_smoke`：三模型 × 两种子 × 一轮训练，共 6 个任务；实际使用物理卡 0/1/2/3，已核验 checkpoint 与统计结果；6 个检查点跨分配的 4090 重放后，逐患者预测文件逐字节一致。显存/进程快照见该目录的 `gpu_activity.csv`、`gpu_processes.csv`。

GPU-only 要求提出之前完成的历史正式 20 轮对比统一使用 CPU 单线程/任务、多进程并行；GPU 短跑结果不混入正式统计。小型神经 ODE 含大量自适应求解与调度，GPU 可用不代表这种小数据实验一定更快。CPU/GPU 结果分别保存，避免把硬件数值差异混入同一组训练种子统计。

## 统一条件

| 条件 | 三模型共同设置 |
|---|---|
| 数据 | 同一原始合成 CSV，并记录 SHA-256 |
| 划分 | 默认 split seed=42；患者级标签分层 70/15/15，所有训练种子与模型共用 |
| 归一化 | 仅训练患者拟合 min-max；时间统一除以 72 小时 |
| 输入信息 | 同一患者完整有效记录，无随机观测子采样 |
| 训练采样 | 同一训练种子下相同每轮患者顺序；不重复采样阳性，不额外类别加权 |
| 优化器与预算 | Adam、lr=0.001、weight decay=0、batch size=10、梯度裁剪 10、固定相同轮数 |
| 训练潜变量采样 | 两种随机隐状态模型均 3 次；确定性 ODE-RNN 无随机后验 |
| 选模 | 最大验证 AUROC；并列时最大验证 AP；再并列保留最早轮次 |
| 评估概率 | 默认每名患者 32 个后验样本的**概率平均**，不是平均 logits，也不是挑最佳随机抽样 |
| 评估随机性 | 固定独立 eval seed 加 patient ID；不依赖患者顺序，不改动训练随机状态 |
| 时间参照与评估 | 论文模型统一补入无观测的 0/72 小时边界；逐患者独立推断，消除批内时间轴边界变化和旧模型终点的影响 |
| 校准 | 三者均不进行概率校准 |
| 阈值 | 使用最佳权重的同一组验证概率选择最高 F1 阈值；测试集不选阈值 |
| 测试 | 选定检查点和阈值后才计算测试预测；另外保留统一 0.5 阈值的结果 |
| 报告 | AUROC、AP、Brier、Accuracy、F1、Precision、Recall；均值和样本标准差 ddof=1 |

这 5 次运行的差别是训练初始化、Dropout、采样和训练顺序；测试患者始终相同。因此标准差反映**训练随机性**，不是不同患者总体上的置信区间，也不是五折交叉验证。若评估数据划分敏感性，应在新的输出目录改变 `--split-seed`，并在同一划分内继续保持三模型配对。

论文模型在训练和评估中都使用预定义采集窗口 `[0,72]` 的共同时间边界（归一化 `[0,1]`），边界没有真实观测时 mask=0。Latent ODE 识别从共同右边界向左边界运行；ODE-RNN 在共同左边界初始化、在共同右边界取分类状态。这样更换同批患者不会改变初始状态或分类状态的时间参照。新增跨批时间轴回归测试检查单人/多人输入的分类概率一致性。此项修正前的排查运行未进入最终统计。

32 次后验采样只用于稳定每次训练后的概率估计，不计为 32 次独立训练。`--eval-mode mean` 可对两个随机模型均使用后验均值，但应放在另一个输出目录，不能在看到测试结果后从两种评估方式中挑最好的一种。

每个模型的网络尺寸保留其已有配置，参数量单独报告，不宣称参数量相等。固定优化器与训练轮数是共同训练预算，不等于三个模型各自调参后的最优性能。

## 两种损失协议的含义

默认 `--objective native` 比较“原生辅助目标 + 共同分类评估条件”的完整方法：

- 旧结构：`0.3 * (末次有效观测重建 + 0.01*KL + 0.1*证据项) + 2*BCE`。
- Latent ODE：完整轨迹高斯负 ELBO + `100*BCE`，保持论文基础训练目标。
- ODE-RNN：因果自回归高斯 NLL + `100*BCE`，训练时 teacher forcing=0.5。

原生辅助损失和权重是方法的一部分，本协议的结果不能单独归因为编码器或 ODE 架构变化。

`--objective bce` 则三者只使用无类别权重的平均二分类 BCE，不加重建、KL 或证据项。它用于检查性能差异是否来自辅助损失。此时 Latent ODE 的生成 ODE/解码器不会得到分类损失的梯度，因为该模型的分类头作用于初始隐状态；因此它是**统一分类损失消融**，不是完整 ELBO 训练的论文模型。两种协议应分别报告。

## 旧模型适配边界

旧模块直接从 `legacy/model.py` 复用：64 维 Transformer 隐层、32 维随机隐状态、5 个增广维度、原 Decoder 与 evidential 分类器。原文件不修改，不载入旧实验权重；三个模型均从头训练。

对已确认的实现问题明确进行两处修正：

1. 积分终点为**各患者自己的最后观测时间 + 12 小时**，不使用批次最大时间。由于输入时间统一除以 72，12 小时对应 `12/72`。通过自治 ODE 的时间变量代换并行计算独立终点，单元测试与逐人直接积分对照。
2. 重建目标使用**每名患者最后一次真实观测**，不取补齐张量最后一列，保证每名训练患者都参与辅助任务。

同时统一移除旧训练器的阳性重采样、类别额外加权、独有的优化器/调度器和 Platt 校准。这是“旧网络结构在共同训练评估条件下”的比较，**不是旧代码原封不动运行**，不能将结果等同于旧文档的历史正式分数。

原 evidential 分类器目前不支持特征级缺失，因此适配器遇到缺失特征会明确报错，避免悄悄替旧模型增加未说明的插补方法。当前 CSV 的六项特征完整，无此限制。

## 输出

每个目录 `MODEL/seed_N/` 保存：

- `best.pt`：按验证分类指标选择的最佳权重、配置、患者划分和数据哈希。
- `history.json`：每轮训练损失和验证指标。
- `predictions.json`：验证和测试的 patient ID、标签、平均概率；阈值选择复用验证预测。
- `metrics.json`：验证阈值、最佳轮次、参数量、测试指标和固定 0.5 阈值指标。
- `config.json`：完整运行参数。

输出根目录保存 `protocol.json`、`data.json`、`environment.json`、`run.log`、`runs.csv`、`summary.csv`、`summary.md`、`summary.json`。JSON 还包含按训练种子配对的三组模型指标差值及其均值/样本标准差；AUROC/AP 差值正数表示前者更高，Brier 差值负数表示前者更好。

只有一个训练种子时标准差写为 null，而不是误报为零。统计单元测试使用手工已知均值/样本标准差及配对差值检查结果。

## 本次验证与正式结果

2026-10-20，正式完成两个协议，每个协议三模型 × 五个训练种子 × 20 轮，共 **30 次正式训练**。正式结果统一使用 CPU，GPU 短跑单独核验。33 项单元测试全部通过，包括批次时间边界回归；所有正式运行均通过独立审计。

下表为测试集均值 ± 样本标准差（5 个独立训练种子，固定同一患者划分）。

| 模型 | 原生损失 AUROC | 原生损失 AP | 统一 BCE AUROC | 统一 BCE AP |
|---|---:|---:|---:|---:|
| 旧结构（修正适配器） | 0.9120 ± 0.0867 | 0.8821 ± 0.1072 | 0.9600 ± 0.0894 | 0.9126 ± 0.1955 |
| Latent ODE | 0.3480 ± 0.1474 | 0.3770 ± 0.1569 | 0.3200 ± 0.1449 | 0.3020 ± 0.0553 |
| ODE-RNN | 0.8120 ± 0.2480 | 0.8078 ± 0.2124 | 0.8280 ± 0.0832 | 0.7909 ± 0.1078 |

在这次共同 20 轮预算和固定合成数据划分下，旧结构分类表现最好，Latent ODE 的分类表现较差；统一 BCE 也未消除这个差距，不能只把问题归因为 ELBO 辅助损失。此处不据此推断论文方法在原始基准上的优劣，也不将单一小测试集的结果解释为临床泛化能力。

原生损失 ODE-RNN 的波动较大，说明先前一次 AUROC=1 的结果不代表多次训练的稳定水平。训练预算未针对各模型做最优超参数搜索；后续若比较充分收敛后的性能，应预先定义更长的共同预算或相同规模的验证集调参程序，不根据测试分数选择。

完整结果：`results/comparison_native/summary.md`、`summary.json`、`runs.csv`、`summary.csv`，以及 `results/comparison_bce/` 中的对应文件。配对差值及其标准差见各 `summary.json`。可视化见 `results/comparison_overview.png`。每个目录的 `audit.log` 核验全部 15 个模型/种子组合、共同患者划分、验证集选模、阈值、缓存预测和标准差。依赖版本与源码 SHA-256 见 `provenance.json`。

GPU 短跑是三模型 × 两种子 × 一轮，共 6 个任务，已确认使用全部四张 RTX 4090；它的训练轮数不足，不解释为性能对比结果。


独立核验与绘图：

```bash
python -m reproduction.check_comparison results/comparison_native
python -m reproduction.check_comparison results/comparison_bce
python -m reproduction.plot_comparison results/comparison_native results/comparison_bce \
  --output results/comparison_overview.png
```
