# HealthFlux：在 ADNI 共同任务上复现与比较

## 1. 本次要比较什么

我们已有的 shared_matched 从不规则随访中估计患者状态，再沿 ODE 重构历史、预测未来。HealthFlux 也将患者表示为随时间演化的状态，并在新观测到来时更新。因此，本次让两种方法读取相同的 ADNI 资料，回答相同的字段预测问题。

论文来源为 [A world model simulates the latent dynamics of human health](https://www.medrxiv.org/content/10.64898/2026.09.19.26363460v3.full)，DOI：[10.64898/2026.09.19.26363460](https://doi.org/10.64898/2026.09.19.26363460)。本次固定使用 **2026 年 10 月 7 日发布的 v3**。已下载 [PDF](reference/paper_v3.pdf)、[全文文本](reference/paper_v3.txt) 和 [版本元数据](reference/medrxiv_metadata.json)。

在论文、作者实验室网站和公开仓库检索中，未找到可用的作者 HealthFlux 模型实现。发现的一个同名 GitHub 仓库是 React/PostgreSQL 医疗管理应用，已核对并排除。本目录依据论文 Methods 的公式与结构实现模型，检索结果及 PDF 哈希见 [来源记录](reference/source_audit.json)。

实时训练状态和同任务比较见 [结果汇总](results/adni_comparison/summary.md)。

HealthFlux-kl 与八组对照的状态空间、速度场及病例可视化见 [统一横向分析](../paper_odeae/results/state_space_model_comparison/report.zh-CN.md)。其中 HealthFlux 同时展示 256 维 ODE 状态和 384 维完整健康表示。

## 2. 先理解 HealthFlux 的两种状态

HealthFlux 同时维护两部分：

- **确定性状态 h：256 维。** 它在两次检查之间沿 ODE 连续变化。
- **随机状态 z：128 维。** 它描述当前状态的不确定性，由先验或观测后的后验给出。

模型将二者拼接并做 LayerNorm，得到 384 维健康表示：

\[
r(t)=\operatorname{LN}([h(t),z(t)]).
\]

另外，模型把静态背景编码为 128 维，把当前年龄编码为 64 维。于是，同样的潜在状态，在不同年龄和背景下，可以有不同的演化速度。

### 两次检查之间

ODE 只推进确定性状态。以年龄 a 的“十年”为时间单位：

\[
\frac{dh}{d(a/10)}=\tanh f_\omega([h,a_{\mathrm{enc}}(a),c]).
\]

向量场包含两层宽度为 384 的 SiLU 隐藏层。积分沿用论文的固定步长 RK4：名义单步最长 10 年，每个间隔最多 4 个子步。这里保留原文求解器和时间单位。

### 检查到来时

模型先根据演化到检查时刻的状态得到 Gaussian 先验；再结合本次观测，得到 Gaussian 后验。训练时从后验采样，将观测、随机状态、背景和年龄交给 GRU 更新确定性状态，随后做 LayerNorm。

测试时使用后验均值。没有新观测时，保持 ODE 推进后的确定性状态，并由先验提供随机均值，不执行 GRU 更新。这是未来自由推演的路径。

这些步骤对应论文 Methods 的公式 2–6。当前实现保留 Gaussian 先验/后验、重参数采样、GRU 更新、两处 LayerNorm，以及同年龄观测的模态注意力汇聚。

### 代码中固定的层定义

论文没有逐层展开每个辅助编码器。为使本次实现可以直接检查，代码统一使用两层 SiLU 隐藏层，宽度 384；随机尺度通过 `softplus(raw)+0.0001` 保持为正。向量场内部不放 dropout，其他观测和分布网络使用 0.1 dropout。

年龄编码的 15 个原始输入为：年龄除以100、其平方、`log(1+年龄)/log(101)`，以及六组正弦/余弦；第k组相位为 `2^k×π×年龄/100`，k从0到5。静态编码器接收性别独热编码、性别缺失指示、APOE值及其掩码，共6个输入。这些是本次明确固定的实现细节。

## 3. 怎样接入当前 ADNI 数据

### 保持既有数据合同

使用原缓存 `paper_odeae/cache/local_available.pkl`，没有重新划分患者：

| 划分 | 患者数 |
|---|---:|
| 训练 | 2,093 |
| 验证 | 393 |
| 测试 | 414 |

全部模型使用同样的 40 个原始字段、49 个展开后的输出维度、季度末时间戳和缺失掩码。每位患者枚举相同的历史截止点。

### 按 HealthFlux 的方式组织输入

同一次检查内，先把真实诊断作为临床事件处理，再把其余可用模态做注意力汇聚。这里根据已有字段形成六组：人口学、认知量表、PET、MRI 体积变化、遗传和脑脊液。

静态背景使用第一条历史记录已经提供的性别、APOE 及其缺失标记；没有从之后的记录补填。年龄来自当前检查时间。模型内部把连续输入变为训练集均值和标准差尺度并截断至 [-10,10]，符合原文数值编码方式。监督目标仍保留原比较任务的训练集 min-max 尺度，因此评分完全一致。

原文的三维影像编码器接收原始 MRI 体积；本轮共同任务的 MRI 输入是已经纳入 40 字段的海马体积变化值，按数值模态编码。所有模型读取的字段范围相同。

### 使用共同的输出任务

原论文主要训练目标是“下一个临床事件是什么、何时发生”，随后冻结世界模型，训练疾病风险解码器。本轮比较沿用既有 ADNI 任务，将 HealthFlux 健康表示接到 40 字段解码器，并联合训练。

解码器保留原文的 384→1024→512 隐藏结构、GELU、LayerNorm 和 0.08 dropout，最后一层改为本任务所需的 49 个输出。全部参数从 seed 0 初始化；诊断事件表按 ADNI 的 CN、MCI、Dementia 三类重新学习。

| 内容 | 论文 v3 | 本轮 ADNI 任务 |
|---|---|---|
| 状态与动力学 | 256 维 h＋128 维 z，年龄/背景条件 ODE | 保留 |
| 观测更新 | Gaussian 先验/后验，GRU＋LayerNorm | 保留 |
| 同时观测处理 | 临床事件先处理，其他模态注意力汇聚 | 保留；按本地字段组织模态 |
| 连续时间积分 | 十年单位，RK4 | 保留 |
| 原始影像 | 三维 MRI＋预训练 Triad 编码器 | 使用共同任务中的 MRI 数值字段 |
| 世界模型监督 | 下一临床事件＋时间＋0.01 KL | 改为共同的字段重构/预测监督 |
| 训练顺序 | 世界模型训练后冻结，再拟合风险头 | 本轮联合训练状态模型与字段头 |
| 输出目标 | 多疾病风险 | 当前 ADNI 的全部可用字段 |

因此，结果表比较的是 **HealthFlux 方法在共同 ADNI 任务上的适配结果**；论文 UK Biobank 的风险指标保存在来源资料中，不与 ADNI 字段误差混算。

## 4. 历史重构与未来预测怎样计算

假设患者有 4 次检查，历史截止点设为第 2 次。模型先按顺序读取第 1、2 次检查，得到截止点状态。

此后分两条路径：

1. 从截止点向后积分到第 1 次检查，重构已经获得的历史。截止点本身使用观测后状态。
2. 重新从截止点出发，向前积分到第 3、4 次检查，预测未来。途中不读取第 3、4 次的真实值，也不做观测更新。

每个未观测查询时点，都由确定性 ODE 状态和随机先验均值共同组成健康表示。历史向后重构是为适配共同任务增加的查询方式。

验证和测试直接调用现有 `paper_odeae.train.evaluate`。数值字段计算 MSE，类别字段计算交叉熵，再按字段求和；缺失字段不评分。历史、未来和整体三个区域使用原来的计数方法。所有模型都按原验证集整体误差选择最佳检查点。

## 5. 为什么运行两组损失系数

共同训练目标写为：

\[
L=L_{\mathrm{MSE}}+L_{\mathrm{CCE}}+\beta D_{\mathrm{KL}}(q\Vert p)
+0.02\|\theta_{\mathrm{ODE}}\|_2^2.
\]

KL 使用 HealthFlux 的观测后验对转移先验计算；每个有效观测节点对 128 个随机维度求和，再对节点取平均。ODE L2 作用于向量场 MLP。

| 实验名 | β | 用途 |
|---|---:|---|
| common_task | 1 | 与此前概率基线的共同损失权重一致 |
| paper_kl | 0.01 | 在同一 ADNI 任务中保留原文 KL 系数 |

两组只改变 β。数据、随机初始化、网络、优化器、学习率、batch 和选点规则相同。`paper_kl` 这个名字专指 KL 系数，不表示恢复了原文的下一事件训练任务。

两组实际均为 **5,156,722 个可训练参数**。这里采用原文状态与隐藏宽度，比较表同时列出 shared_matched 六维、latent2x、latent4x 等已有容量的真实参数量。

## 6. 训练预算和 GPU 安排

训练沿用共同协议：seed 0，batch size 32，Adam 初始学习率 0.003，每 2,000 步乘以 0.97 的衰减，输入 dropout 0.1，梯度范数裁剪至 10。最多训练 300 轮，至少 50 轮，连续 40 轮验证分数未改善则早停。

原文使用 AdamW、0.0006 学习率、2,000 步预算及大批量多卡训练。这部分在本轮统一为已有 ADNI 训练协议，具体配置写入每组 `config.json`。

队列检查物理 GPU 0、1、2、3 的计算进程，每张空闲卡最多领取一个新任务。启动时 GPU 0 空闲，另外三张卡继续运行此前的 latent2x 消融。每组先做小规模端到端 CUDA 检查，再正式训练；另一组等待下一张空闲卡。

训练、验证、测试、求解器、Gaussian 运算和损失均在 CUDA 上完成。数据整理、配置管理和报告汇总在 CPU 上运行。每组记录实际张量设备、物理 GPU、数据哈希、参数分解、模型源码哈希和最佳检查点重载差异。全部训练接入 SwanLab。

## 7. 独立环境与复现命令

已从 `ode` 克隆出独立的 `healthflux` Conda 环境。当前实现使用 PyTorch 自带模块与显式 RK4，已有依赖即可运行。环境包括 Python 3.11、PyTorch 2.5.1＋CUDA 12.1、NumPy 1.26.4、SwanLab 0.10.3。实际版本及 GPU 核对见 [环境检查](reference/environment_check.json)。

环境记录：[Conda 导出](reference/environment.yml)、[pip 锁定列表](reference/requirements-lock.txt)。

从头配置环境可运行：

```bash
/home/xjh/miniforge3/bin/conda create -y -n healthflux --clone ode
```

启动实验：

```bash
bash scripts/run_healthflux_adni.sh
```

CUDA 单元检查：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xjh/miniforge3/envs/healthflux/bin/python \
  -m unittest healthflux.tests.test_model -v
```

队列会阻止重复启动，不覆盖未完成的历史记录。已有 shared_matched 和其他参考模型直接读取其完成结果。

## 8. 结果和实现文件在哪里

- [实时对比表](results/adni_comparison/summary.md)：新模型状态与同标准历史/未来/整体误差。
- [实验清单](results/adni_comparison/manifest.json)：固定的数据和参考结果哈希。
- [模型代码](model.py)：年龄条件 ODE、模态汇聚、Gaussian 状态与 GRU 更新。
- [训练入口](train.py)：共同损失、选点、SwanLab 与 CUDA 记录。
- [队列和汇总](study.py)：空闲 GPU 调度及完整比较表。
- [单元检查日志](reference/unit_tests.log)：时间隔离、缺失观测、RK4 和梯度检查。

新实验的检查点、训练曲线、SwanLab 标识与最终指标分别保存在 `results/adni_comparison/common_task/seed_0` 和 `results/adni_comparison/paper_kl/seed_0`。`source_snapshot` 固定了本轮实际执行的模型、训练器和共同评分代码。
