# shared_matched：单卡损失消融比较

三组均为22,545参数、seed 0、相同数据与训练设置。基线复用已有 shared_matched；weak_prior 在 GPU 0 继续训练；新增 Jacobian 在 GPU 1 单卡训练。

| 模型 | 新增损失 | λ | 状态 | 轮数/最佳轮次 | 验证整体 | 测试预测 | 测试整体 | 验证校正前 | 测试校正前 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| shared_matched | 无 | 0 | 已有结果 | 96/56 | 0.88722 | 1.13926 | 0.85063 | 0.81976 | 0.77280 |
| weak_prior | 校正前预测 | 0.01 | complete | 110/70 | 0.86941 | 1.12544 | 0.84361 | 0.81584 | 0.76411 |
| jacobian | 速度场 Jacobian | 0.01 | complete | 187/147 | 0.87711 | 1.14728 | 0.84297 | 0.83463 | 0.77398 |

两项新增损失分别做独立消融：weak_prior 优化读入当前观测之前的预测；jacobian 优化速度场在邻近状态上的变化平缓程度。Jacobian 组的 prior_weight=0，没有叠加 weak_prior。

Jacobian 采用完整6×6矩阵的精确计算，36项平方求和，再对批内有效生成状态取平均。状态覆盖历史与未来的实际检查时间，补齐位置排除；采样位置停止梯度，正则直接训练共享向量场参数。

| 模型 | 验证平均 Jacobian 平方范数 | 测试平均 Jacobian 平方范数 |
|---|---:|---:|
| shared_matched | 0.005799 | 0.005830 |
| weak_prior | 0.005798 | 0.005838 |
| jacobian | 0.005216 | 0.005273 |

Jacobian 指标反映模型在生成状态附近对扰动的局部敏感度；测试预测与整体误差反映预测效果。基线和 weak_prior 的此项指标由已有检查点补算，训练记录保持原样。所有组仍按原验证整体误差选择检查点。
训练通过 SwanLab 记录 jacobian_frobenius_squared 与 weighted_jacobian。设计和公式见 [Jacobian 消融说明](../../docs/JACOBIAN_REGULARIZATION.zh-CN.md)。
