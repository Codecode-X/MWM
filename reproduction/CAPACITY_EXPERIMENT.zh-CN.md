# 参数量匹配实验

扩容前的版本已保存为 Git 提交 `e3752ce`，标签为 `baseline-before-capacity-scaling-20261020`。该提交保留原代码、测试和历史实验结果。

旧 Transformer 模型共有 126,767 个参数。本轮保留论文模型的状态更新、连续时间积分、后验、解码与分类方式，仅扩大隐藏状态维度：

| 模型 | 原参数量 | 扩大后参数量 | 修改 |
|---|---:|---:|---|
| Latent ODE | 32,727 | 126,766 | 识别 ODE-RNN 隐藏维度 40 → 143；生成潜变量仍为 20 |
| ODE-RNN | 15,877 | 126,997 | 递归隐藏状态维度 20 → 140 |

这些是单独改变对应隐藏维度时最接近旧模型参数量的整数选择。ODE 网络宽度 50、隐藏层数 3 和分类头隐藏维度 300 保持不变。Latent ODE 采用识别状态扩容，可以独立于生成潜变量维度增加推断能力。ODE-RNN 的连续状态就是递归隐藏状态，扩容会同时扩大 ODE 输入输出、GRU 和分类头输入。

实验包含小模型与扩大模型，并重新训练旧模型作为共同参照，均在同样的 CUDA 环境下运行。历史 CPU 分数不作为本轮配对对照。固定数据 SHA256、患者划分 seed=42、训练种子 0–4、20 epoch、batch size=10、Adam lr=0.001、相同梯度裁剪与检查点规则。原生损失和统一 BCE 各运行一次完整比较，共 50 次训练。每次训练后只用验证 AUROC/AP 选择检查点，用验证 F1 选择阈值，然后测试一次。概率采样数维持之前的 32，不根据测试结果改采样规则。训练种子保持配对；参数形状发生改变，因此同种子不代表两种容量具有相同的初始权重。

四张 RTX 4090 分配独立任务；`JOBS=4` 默认每张卡一个进程。本次执行 `JOBS=8`，每张卡两个进程，利用小模型显存占用低、单进程 GPU 利用率低的特点。并行数量不改变数据、优化和评分协议。每个 metrics.json 记录实际 tensor_device、CUDA_VISIBLE_DEVICES 和 GPU 型号。

复现命令：

```bash
conda activate ode
JOBS=8 EPOCHS=20 bash scripts/run_capacity_comparison.sh
# 同协议跳过已完整完成的任务；不恢复中断任务的优化器状态：
JOBS=8 EPOCHS=20 bash scripts/run_capacity_comparison.sh --resume
# 独立目录运行：
OUTPUT=results/capacity_cuda_repeat JOBS=4 bash scripts/run_capacity_comparison.sh
```

单独使用参数匹配配置：

```bash
OUTPUT=results/matched_only EPOCHS=20 bash scripts/run_comparison.sh \
  --capacity matched --models latent_ode ode_rnn --seeds 0 1 2 3 4
```

结果见 `results/capacity_cuda/capacity_summary.md`，包含测试均值、样本标准差和同种子差值。完整测试、各组审计与实验日志由脚本保存。参数量相等只能排除“总参数数目差异”这一解释，不能排除数据归纳偏置、分类信息位置、训练目标、潜变量采样噪声和优化预算的差异。

## 本次实际结果

50 次训练、1000 个 epoch 已全部完成；35 项单元测试通过；四组实验共 50 次运行的结果审计均通过。所有模型训练、验证、测试均使用 RTX 4090，实际分配覆盖物理 GPU 0、1、2、3。

| 模型 | 参数量 | 原生损失 Test AUROC | BCE Test AUROC |
|---|---:|---:|---:|
| 旧模型 | 126,767 | 0.968 ± 0.061 | 0.936 ± 0.132 |
| Latent ODE 小 | 32,727 | 0.492 ± 0.194 | 0.416 ± 0.144 |
| Latent ODE 扩大 | 126,766 | 0.524 ± 0.071 | 0.420 ± 0.164 |
| ODE-RNN 小 | 15,877 | 0.932 ± 0.088 | 0.816 ± 0.092 |
| ODE-RNN 扩大 | 126,997 | 0.972 ± 0.039 | 0.772 ± 0.093 |

Latent ODE 扩容的同种子 AUROC 差值为：原生损失 `+0.032 ± 0.215`，BCE `+0.004 ± 0.282`。这轮实验未显示扩容可以解决其分类性能问题。

ODE-RNN 扩容的同种子 AUROC 差值为：原生损失 `+0.040 ± 0.113`，BCE `−0.044 ± 0.185`。原生损失下扩大模型的均值接近旧模型，但种子之间仍有明显波动，BCE 下也没有保持同样趋势，不能将它解释为扩容的稳定收益。此外，原生损失下扩大 ODE-RNN 的 Brier 为 `0.214 ± 0.009`，旧模型为 `0.134 ± 0.051`，排名指标接近并不表示概率质量已经接近。

总参数预算已非常接近，但扩大 Latent ODE 仍远弱于旧模型，所以“总参数太少”不足以解释差距。此前发现的反向识别编码器对不同时间位置的信息保留、生成目标和分类目标的配合、固定预算下的优化，以及小验证集选模波动，仍是需要逐项检验的解释；本轮没有直接区分这些原因。相同 20 epoch 也不代表各模型已经收敛。

BCE 下旧模型的解码器（6,662 参数）不参与分类损失，Latent ODE 的生成 ODE 和解码器（合计 7,296 参数）也不参与。扩大 Latent ODE 的分类路径仍有 119,470 个参数，与旧模型的 120,105 个非常接近。因此，其差距也不能简单归因于“总参数包含无效生成分支”。ODE-RNN 训练时使用 0.5 teacher forcing，预测值会反馈到循环状态，解码器仍可通过此路径接收分类梯度。对应 GPU 梯度检查保存在 `classification_parameter_paths.json`。

完整指标与配对统计见 [汇总表](../results/capacity_cuda/capacity_summary.md)，逐种子 AUROC 与均值/标准差见 [图](../results/capacity_cuda/capacity_auroc.png)。这些结论限于当前合成数据、固定患者划分和本轮训练预算。
