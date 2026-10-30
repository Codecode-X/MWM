# 从这里开始复现 ADNI ODE-AE 论文

本目录用于复现 Zanin 等人的 [Predicting Alzheimer's Disease Progression from Sparse Multimodal Data by NeuralODE Models](https://doi.org/10.1101/2025.08.26.672412)。工作按三个步骤展开：先把数据整理成论文使用的字段和时间序列，再实现论文模型与对照，最后检查训练结果是否接近论文报告。

当前共享与扩容消融已按用户要求停止，已完成结果和中断检查点均保留。最新的 [shared_matched 与论文 ODE-AE 对比](results/shared_matched_comparison/comparison.zh-CN.md) 包含结构差异、本地同数据结果、等参数对照和论文报告值。

后续以 shared_matched 为基线的四组扩容实验见 [实验设计](docs/SHARED_MATCHED_SCALING_4X.zh-CN.md) 与 [实时结果](results/shared_matched_scaling_4x/summary.md)。四组分别扩大 INIT、UPDATE、共享 ODE 和潜在维度，总参数量约为基线四倍。

潜在状态容量比较现已覆盖1x、2x、4x、8x、16x。latent8x和latent16x已完成，新增latent2x（28维、44,875参数）在GPU 0训练。参见 [扩容设计](docs/LATENT_SCALE_UP.zh-CN.md) 与 [五档容量实时比较](results/shared_matched_latent_scaling/summary.md)。

新增 latent2x 的三组损失消融：weak_prior＋Jacobian、仅 weak_prior、仅 Jacobian，新增项权重均为 0.01，分别使用 GPU 1、2、3。参见 [实验设计](docs/LATENT2X_LOSS_ABLATION.zh-CN.md) 与 [四组实时对比](results/latent2x_loss_ablation/summary.md)。

新增 HealthFlux 基线：使用独立的 `healthflux` Conda 环境，按论文 v3 实现双状态连续时间模型，接入相同 ADNI 数据与字段评分。参见 [模型与任务适配](../healthflux/README.zh-CN.md) 和 [实时比较](../healthflux/results/adni_comparison/summary.md)。

shared_matched 已完成 [二维状态空间、共享速度场与患者轨迹可视化](results/shared_matched_state_space/report.zh-CN.md)，并提供 [交互查看页面](results/shared_matched_state_space/interactive.html)。

[临床速度实验 A](results/shared_matched_clinical_velocity/report.zh-CN.md) 进一步使用 DECODER 的链式导数，将潜在速度转为 MMSE、CDR-SB 和 ADAS 各题的分数/年，包含相邻随访变化率评价、临床速度场及较快下降患者的排序对照。

[局部动力学实验 B](results/shared_matched_local_dynamics/report.zh-CN.md) 在六维空间计算不同患者近邻的方向一致性、速度敏感性与精确Jacobian范数，并关联12/24个月MMSE预测误差，包含区域、二维投影、邻域大小及患者重抽样分析。

weak_prior最佳检查点已按同样流程重做实验A、B，见 [与原模型的配对比较](results/weak_prior_research_AB/report.zh-CN.md)，以及 [weak_prior实验A](results/weak_prior_clinical_velocity/report.zh-CN.md)、[weak_prior实验B](results/weak_prior_local_dynamics/report.zh-CN.md)。

针对Dementia下降预测偏慢，新增 [MMSE速率机制分解](results/mmse_decline_mechanism/report.zh-CN.md)：逐状态计算速度×临床头敏感性×方向投影，并沿随访区间积分，比较原始模型和weak_prior的方向折损、速率量级及快速下降者预测。

新增 [连续疾病阶段与进展分析](results/shared_matched_continuous_stage/report.zh-CN.md)：直接复用六维 shared_matched 的诊断头与共享 ODE，计算连续阶段评分、自动微分进展速度和五年自由演化轨迹，并按患者验证12、24、36个月的MCI→Dementia转化。实验冻结原模型参数。

shared_matched、概率 ODE-AE 与 deterministic_22545 的 [统一比较文档](results/odeae_baseline_state_space/report.zh-CN.md) 包含完整指标表、三模型状态与速度场平行图，以及相同患者的潜在轨迹和 MMSE 预测案例。

新增 [弱校正前预测损失消融](docs/WEAK_PRIOR_LOSS.zh-CN.md)：λ=0.01，使用 GPU 0 单卡训练，直接复用已有单卡 λ=0 基线。init_4x 已停止，其余三组扩容继续运行；结果见 [实时比较](results/shared_matched_prior_weak_single/summary.md)。

继续加入 [Jacobian 正则消融](docs/JACOBIAN_REGULARIZATION.zh-CN.md)：λ=0.01，在 GPU 1 单卡训练，独立于 weak_prior。原实时比较页统一展示基线、校正前监督和 Jacobian 正则三组结果。

## 1. 先明确要复现哪些模型和指标

论文的主要预测模型是 ODE-AE。它从历史末端反向处理检查记录，在第一次检查处估计初始潜在状态分布，再用生成 ODE 重构完整轨迹。

本目录同时实现两个对照。确定性版本保留反向编码和 ODE 生成，直接使用一个潜在状态。LSTM 版本用递归单元代替 ODE 编码和生成过程。

先开展论文 Table 3 的轨迹预测实验，分别报告历史范围内的插值误差、未来预测误差和完整序列误差。评分将各数值字段的 MSE 与各类别字段的交叉熵相加。论文中的参照结果为：

| 模型 | 插值误差 | 预测误差 | 整体误差 |
|---|---:|---:|---:|
| ODE-AE | 10.20 | 17.97 | 12.46 |
| 确定性版本 | 10.13 | 17.92 | 12.26 |
| LSTM | 13.01 | 21.90 | 15.50 |

数值来源和论文 PDF 的校验值保存在 [reference/](reference/)。论文的早期诊断分类与轨迹置信带属于后续阶段，当前目录还没有实现这两项实验。

## 2. 再确认本地数据能支持哪些实验

论文描述了 46 项特征。字段合同位于 [features46.json](reference/features46.json)，包括数值字段和类别字段。输入中的类别字段采用独热编码，评分时仍按原始字段计数。

本地 ADNI 文件能够提供其中 40 项。当前缺少：

| 类别 | 缺少的字段 |
|---|---|
| 日常认知问卷 | ECog 自评、ECog 知情者评估 |
| 神经精神症状 | NPI 总分 |
| FDG PET | 顶叶区域 SUVR、颞叶区域 SUVR |
| MRI | 皮质体积相对首次测量的变化 |

本地模式按自然年季度整理检查，使用至少两个有效季度的患者。得到训练 2,093 人、验证 393 人、测试 414 人，沿用源数据的患者划分。这与论文的 2,451/613/766 人划分不同。

据此设置两个运行入口：

- `local_available`：使用当前有训练观测的 40 项字段。它用于验证实现和开展本地模型比较。
- `paper46`：要求全部 46 项字段都有训练观测。准备数据时如果缺字段，会写出审计报告并停止，不生成可训练缓存。

`paper46` 检查的是字段完整性。原论文的患者清单、归一化细节和部分网络设置仍需核对，相关内容在[实现对照](docs/IMPLEMENTATION.zh-CN.md)中逐项说明。

已完成任务与论文存在的差距，以及四 GPU 检查点诊断的证据，见[复现差距分析](docs/DISCREPANCY_AUDIT.zh-CN.md)。其中区分了数据与单位差异、LSTM 后验塌缩和 ODE-AE 信息瓶颈，并列出了后续核查顺序。

将原有 Latent ODE / ODE-RNN 放入相同数据和评测任务的完整比较，见[共同任务比较说明](docs/BASELINE_COMPARISON.zh-CN.md)。该入口使用四张 GPU 运行原容量与小容量对照，统一采用 seed 0。

确定性 ODE-AE 与 ODE-RNN 的结构和误差分解，以及精确扩大到 22,545 参数的四卡 DDP 实验，见[确定性模型与参数量对照](docs/DETERMINISTIC_CAPACITY.zh-CN.md)。

新增的正向历史编码消融同时保留年龄初始化和零初始化，两组都在历史截止点估计状态，并由此重建过去、预测未来，见[正向编码消融说明](docs/FORWARD_ABLATION.zh-CN.md)。

各模型在完整测试集上的积分调用、接受/拒绝步和向量场调用次数，见 [ODE 步数比较](results/solver_steps_full_test/summary.zh-CN.md)。统计区分批积分和患者区间，训练中检查点单独标注。

以 forward_age（22,545参数）为主模型的24组共享场、模块扩容与潜在维度消融，见[实验设计](docs/FORWARD_AGE_SCALING.zh-CN.md)和[实时汇总](results/forward_age_scaling/summary.md)。运行入口为 `bash scripts/run_forward_age_scaling.sh`，会等待既有队列释放GPU后自动执行。

## 3. 目录里的文件怎样配合

```text
paper_odeae/
├── README.zh-CN.md                 从这里开始
├── configs/
│   ├── local_available.json        本地可用字段的训练设置
│   └── paper46.json                完整字段合同的训练设置
├── data.py                        原始表读取、季度合并、标准化、全部历史截断
├── models.py                      ODE-AE、确定性版本、LSTM
├── solver.py                      CUDA 上的自适应 Tsitouras 5/4 求解
├── train.py                       训练、验证选模、测试、检查点重载核对
├── run.py                         四张 GPU 上的独立模型/种子队列
├── report.py                      汇总插值、预测和整体误差
├── tests/                         数据、模型、梯度和求解器检查
├── docs/IMPLEMENTATION.zh-CN.md    原文设置与实现选择的对照
├── reference/                     论文指标、字段合同、来源和本次检查记录
├── cache/                         本地数据缓存，Git 忽略
└── results/                       本目录实验输出，Git 忽略
```

这里的代码和结果独立保存。原有 `src/` 与 `results/adni_cuda/` 继续用于此前的 Latent ODE / ODE-RNN 实验。

## 4. 第一步：准备环境和数据

所有命令从仓库根目录运行，使用已有的 `ode` 环境。模型训练和评估要求 CUDA。

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ode
python -m paper_odeae.data --scope local_available
```

默认读取 `/data/SAMSUNG_backup/ADNI_data/`。数据准备只涉及文件处理和统计，可以使用 CPU。程序会读取原始表，合并访视，再按自然年季度聚合。

完成后，查看 `paper_odeae/cache/local_available_audit.json`。该文件列出实际字段、缺失字段、患者数、来源文件摘要和归一化参数。训练患者用于拟合归一化，验证和测试沿用这些参数。

若准备运行完整字段模式，可执行：

```bash
python -m paper_odeae.data --scope paper46
```

当前资料会在这一步列出六项缺失字段，并停止准备。核对记录已经保存在 [validation_status.json](reference/validation_status.json)。

## 5. 第二步：先检查运行链路

CUDA 检查用于确认模型能够正确处理时间、缺失和梯度：

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest paper_odeae.tests.test_models -v
```

随后可以运行四个小规模任务，分别分配到物理 GPU 0、1、2、3。每个任务只训练一轮，并只使用少量患者和时间点。

```bash
python -m paper_odeae.run --smoke
```

这一步会检查从数据加载、反向传播到保存及重新加载的完整流程。输出位于 `paper_odeae/results/smoke/`，入口是其中的 `summary.md`。

本次已经完成上述检查：6 项检查通过，4 个 GPU 小规模任务完成。它们确认了代码链路可以运行，正式训练的效果还需要下一步实验来测量。

## 6. 第三步：运行正式的本地比较

训练过程默认记录到仓库根目录的 SwanLab 本地看板。可以先运行 `swanlab watch swanlog`，再启动下方命令。配置与使用方法见 [SwanLab 说明](../reproduction/SWANLAB.zh-CN.md)。

使用本地可用字段配置运行三种模型，每种模型五个种子：

```bash
python -m paper_odeae.run \
  --config paper_odeae/configs/local_available.json \
  --gpus 0 1 2 3 \
  --seeds 0 1 2 3 4
```

队列保持每张 GPU 一个训练任务，空闲后领取下一个任务。每个任务单独保存配置、数据摘要、实际设备、训练曲线和检查点。

配置中的初始学习率为 0.003，每 2,000 次更新按 0.97 的比例衰减。最多训练 300 轮，至少训练 50 轮，然后根据验证集整体重构误差使用早停。轮数、批量大小和早停是本次实现选择，说明见实现对照文档。

如果字段已经补齐并成功生成 `paper46.pkl`，可切换配置：

```bash
python -m paper_odeae.run \
  --config paper_odeae/configs/paper46.json \
  --gpus 0 1 2 3
```

运行完成后，队列自动生成报告。也可以单独重新汇总已有结果：

```bash
python -m paper_odeae.report --root paper_odeae/results/local_available
```

## 7. 怎样补充整理好的 46 项数据

程序提供统一 CSV 接口，用来接入补齐后的字段，而无需修改模型。

检查表需要包含 `subject_id`、`date` 和字段合同中列出的 46 个字段。`date` 使用 `YYYY-MM-DD`；数值字段保留原始单位；缺失值留空。类别字段使用从 0 开始的整数，类别数量见字段合同。`age` 使用带小数的年龄，整理后的时间序列需要严格递增。

患者划分另存为 CSV，包含 `subject_id` 和 `split`。`split` 的值为 `train`、`validation` 或 `test`。检查表的空模板见 [canonical_template.csv](reference/canonical_template.csv)。

```bash
python -m paper_odeae.data \
  --canonical /path/to/visits.csv \
  --split-file /path/to/patient_splits.csv \
  --scope paper46
```

程序仍然检查字段是否在训练集实际出现，并只用训练集拟合归一化。

## 8. 从运行结果走向论文效果复现

正式训练完成后，先比较三个模型在相同本地数据上的结果，再检查论文的数据和实现条件是否已经对齐。当前报告始终把本地运行数值与论文参照数值分开列出。

完整复现还需要补齐六项数据，并取得或确认原论文患者划分、归一化及未公开的实现细节。随着这些条件逐步满足，可以在同一个目录下替换数据配置、保留已有运行记录，再开展同口径的效果比较。
