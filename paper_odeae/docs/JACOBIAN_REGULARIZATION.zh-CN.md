# shared_matched 的 Jacobian 正则消融

## 1. 为什么增加这一组

shared_matched 的共享向量场 `f(z)` 决定患者状态怎样连续演化。如果相邻位置上的速度变化很大，较小的状态估计差异也可能导致不同的后续轨迹。

本实验加入 Jacobian 正则，直接约束速度对状态的敏感度。它与已有 weak_prior 分别构成独立消融：weak_prior 监督观测校正前的预测，Jacobian 约束向量场的局部变化。

## 2. 损失怎样定义

新增项为：

\[
L_{\mathrm{jac}}=\mathbb{E}_{z}\left[\left\|\frac{\partial f_\theta(z)}{\partial z}\right\|_F^2\right].
\]

状态维度为 6，因此每个 Jacobian 是 6×6 矩阵。每个元素表示一个速度分量对一个状态分量的变化率。将 36 个元素平方求和，得到该状态处的正则值；再对批内有效状态取平均，不按维度或矩阵元素数额外归一化。

完整训练损失为：

```text
数值重构 MSE + 类别重构交叉熵 + 0.02 × 共享 ODE 参数 L2
    + 0.01 × 平均 Jacobian 平方范数
```

该组 prior_weight=0，不叠加 weak_prior。保留校正前预测的采集，用于记录验证和测试指标。

## 3. 在哪些状态上施加约束

模型根据一段历史生成完整轨迹。本实验收集轨迹在实际检查时间上的潜在状态，覆盖历史重构和未来预测区域；补齐的时间位置排除。首次检查时间也是一个有效轨迹位置。

这些位置作为固定采样点参与本次梯度计算，即状态取 `detach()`，正则梯度直接更新共享向量场参数。原有重构损失继续训练 INIT、UPDATE、向量场和 DECODER。共享网络只计算一份 Jacobian 正则。

## 4. 六维 Jacobian 怎样精确计算

当前向量场由两层全连接和一个 Tanh 组成：

\[
f(z)=W_2\tanh(W_1z+b_1)+b_2.
\]

因此可以直接计算完整导数：

\[
J_f(z)=W_2\operatorname{diag}\left(1-\tanh^2(W_1z+b_1)\right)W_1.
\]

实现使用这个解析式，一次得到全部 36 个元素，保持对网络参数的梯度。CUDA 检查将其数值和参数梯度与完整自动求导 Jacobian 逐项比较，结果通过。

其余检查覆盖生成状态与原预测一致、参数量保持 22,545、补齐位置排除，以及正则梯度只更新向量场。连同原模型、方向和 weak_prior 检查，共 15 项通过。

## 5. 与哪些结果比较

| 组别 | 新增监督或正则 | 权重 | 训练设备 |
|---|---|---:|---|
| 原始 shared_matched | 无 | 0 | 复用已有单卡结果 |
| weak_prior | 校正前预测损失 | 0.01 | GPU 0，原任务继续 |
| jacobian | 完整 Jacobian 平方范数 | 0.01 | GPU 1，新增单卡任务 |

三组都保持 22,545 参数、seed 0、原数据划分、全局 batch 32、Adam 学习率 0.003、原衰减和早停设置。检查点仍按验证整体重构误差选择。其他扩容任务继续运行。

比较时同时查看预测误差和平均 Jacobian 平方范数：前者评价预测表现，后者评价访问到的状态附近的局部敏感度。记录训练分量、每轮时间和显存，以观察新增正则的优化过程与开销。两种新增项系数同为 0.01，各自实际贡献由 weighted_prior、weighted_jacobian 曲线呈现。

已有 shared_matched 和完成后的 weak_prior 检查点会在 CUDA 上补算相同的 Jacobian 指标。补算只执行推理，原训练记录与检查点保持不变。

## 6. 如何运行和查看

```bash
bash scripts/run_shared_matched_jacobian.sh
```

脚本先做单卡小数据训练、验证、测试和检查点重载检查，然后补算基线指标并开始完整训练。训练接入 SwanLab，记录 `jacobian_frobenius_squared` 和 `weighted_jacobian`。

模型、训练器、Jacobian 实现和诊断脚本在启动前保存代码快照及哈希。实验文件记录物理 GPU、模型与张量设备、数据哈希、实际参数量和各损失分量。

原 [weak_prior 汇总页](../results/shared_matched_prior_weak_single/summary.md) 扩展为三组共同的比较表。新增任务的 [独立汇总页](../results/shared_matched_jacobian_single/summary.md) 同步显示结果。

实现见 [jacobian.py](../jacobian.py)、[train.py](../train.py) 和 [jacobian_study.py](../jacobian_study.py)。
