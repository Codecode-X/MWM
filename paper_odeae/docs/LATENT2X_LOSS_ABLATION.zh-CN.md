# latent2x 的 weak_prior 与 Jacobian 损失消融

## 1. 为什么在 latent2x 上做这三组

latent2x 将 shared_matched 的潜在状态从 6 维增加到 28 维，保留 init、update、decoder 的隐藏宽度 159，以及共享 ODE 的隐藏宽度 188。模型共有 44,875 个可训练参数。

本次固定这套网络，分别加入校正前预测约束、Jacobian 约束，再将两项同时加入。这样可以比较两项约束各自的作用，以及它们组合后的效果。

## 2. 四组如何比较

| 组别 | weak_prior 权重 | Jacobian 权重 | 物理 GPU | 训练来源 |
|---|---:|---:|---:|---|
| latent2x 基线 | 0 | 0 | 0 | 使用正在运行的原实验 |
| both | 0.01 | 0.01 | 1 | 从头训练 |
| weak_prior | 0.01 | 0 | 2 | 从头训练 |
| jacobian | 0 | 0.01 | 3 | 从头训练 |

四组均使用 seed 0。三组新实验继承已有 latent2x 的数据划分、归一化、初始化、优化器、batch size、输入 dropout、ODE 求解器与早停规则。原基线继续运行，完成后直接进入同一张对比表。

## 3. 三种损失分别约束什么

### 原训练目标

原损失由数值字段的 MSE、类别字段的交叉熵，以及权重为 0.02 的 ODE 参数 L2 正则组成。训练会从历史截止状态向前预测未来，也向后重构历史。四组保留这些项。模型为确定性版本，KL 项为零。

### weak_prior：观测到来前先预测

患者状态先沿 ODE 演化到下一次历史检查。此时先解码预测，再读取本次真实观测并更新状态。weak_prior 将更新前的预测与本次观测比较，使用数值字段 MSE 加类别字段交叉熵。

第一条检查之前没有历史观测，因此不计入这项损失。只计算历史中第 2 次及以后的有效检查；缺失字段和 padding 不参与。未来检查仍属于原预测任务，不进入 weak_prior。

这项约束希望 ODE 在两次检查之间就能给出合理的预测，从而减少对观测校正的依赖。

### Jacobian：限制邻近状态的速度变化

对共享向量场 \(f_\theta\)，计算：

\[
L_{\mathrm{jac}}=\operatorname{mean}_{z}\left[\sum_{a=1}^{28}\sum_{b=1}^{28}
\left(\frac{\partial f_a(z)}{\partial z_b}\right)^2\right].
\]

每个状态都计算完整 28×28 Jacobian，将 784 项平方求和，再对 batch 中有效的生成状态取平均。求导使用现有 Linear–Tanh–Linear 向量场的解析表达式。这里沿用此前 Jacobian 实验的定义，不按潜在维度除以 28 或 784。

采样位置来自截止状态生成的历史和未来轨迹，padding 不参与。计算正则时将采样位置 detach，因此这项损失直接优化向量场参数。重构与 weak_prior 仍会训练状态估计网络。

### 三组的完整目标

\[
\begin{aligned}
L_{\mathrm{both}}&=L_{\mathrm{base}}+0.01L_{\mathrm{prior}}+0.01L_{\mathrm{jac}},\\
L_{\mathrm{prior\ only}}&=L_{\mathrm{base}}+0.01L_{\mathrm{prior}},\\
L_{\mathrm{jac\ only}}&=L_{\mathrm{base}}+0.01L_{\mathrm{jac}}.
\end{aligned}
\]

## 4. 训练与结果如何记录

所有实验都用原验证集“整体重构误差”选择最佳检查点。新增损失不进入选点分数。训练最多 300 轮，最少 50 轮，连续 40 轮验证分数未改善则早停。batch size 为 32，Adam 初始学习率为 0.003。

每个新实验先在自己的 GPU 上跑小规模端到端检查，通过后进入正式训练。模型、数据张量、损失、验证和测试均在 CUDA 上计算。运行记录包含物理 GPU 编号、实际张量设备、参数量、数据哈希、冻结源码及检查点重载误差。

三组均接入 SwanLab，记录训练损失分项、验证误差、校正前误差、Jacobian 平方范数、学习率、耗时及显存。权重为零的分项仍可作为诊断记录；是否加入训练损失由对应系数决定。

统一对比表同时列出：验证整体误差、验证未来预测误差、测试历史重构误差、测试未来预测误差、测试整体误差，以及校正前预测和 Jacobian 诊断。基线结束后会在 GPU 上补充同口径诊断，无需重训。

## 5. 运行与查看

启动命令：

```bash
bash scripts/run_latent2x_loss_ablation.sh
```

实时结果见 [汇总表](../results/latent2x_loss_ablation/summary.md)。每组的配置、日志、history.json、hardware.json、SwanLab 记录及最终 metrics.json 保存在该汇总目录内。源码快照固定本次实际执行的版本。
