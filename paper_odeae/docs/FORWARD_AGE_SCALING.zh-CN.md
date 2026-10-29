# 以 forward_age 为基准的共享向量场与容量消融

## 1. 这轮实验要回答什么

主模型是正向读取历史、年龄初始化、在历史截止点估计状态的确定性 ODE-AE：潜在维度6，INIT/UPDATE/DECODER隐藏层154，编码/生成ODE隐藏层118，共22,545个可训练参数。

本轮分别检查三个问题：编码与生成共享动力学是否有帮助；额外参数投给哪一个模块更划算；潜在状态只有6维是否限制了效果。

为便于比较，先按近似相同的新增参数预算比较模块，再增加更高预算观察是否继续受益。所有结果都是seed 0，不把单次排序说成统计显著结论。

## 2. 保持不变的条件

使用相同40字段缓存和患者划分（2093/393/414），枚举全部历史截断；输入遮挡10%，全局batch 32，每轮392次优化更新。使用相同Tsit5、年份时间单位、rtol=1e-4、atol=1e-6、Adam初始学习率0.003、每2000步衰减0.97。

数值MSE与类别CCE按字段求和；确定性模型无采样和KL。ODE L2系数固定0.02。最多300轮、至少50轮、耐心值40。初始化、观测更新、解码及向量场仍是一隐藏层Tanh网络，不在这轮同时改变网络深度、数据或影像输入。

旧forward_age使用四卡DDP。新矩阵采用四卡各运行一个独立实验，以提高多组实验吞吐量；每组batch仍为32。因此会重新训练同样结构的单卡baseline，作为矩阵内部的直接参照，旧DDP结果保留为外部参照。这样避免把DDP的数据增强随机流差别混进结构比较。

## 3. 共享向量场的三个对照

- shared_tied：同宽度下让编码ODE和生成ODE指向同一个模块，共21,005参数。向过去生成时沿同一个场负向积分，不另建网络。
- shared_tied_l2x2：共享结构不变，显式将共享场L2乘2，检查“原有两份正则项变为一份”对结果的影响。它是有意设置的正则对照，不是无意重复计数。
- shared_matched：共享场后将MLP宽度设为159、ODE宽度188，恢复到恰好22,545参数。它是等预算结构对照；因为宽度也变了，要与直接共享组共同解释。

主共享组只对唯一的ODE参数集合计算一次L2。优化器也只看到一份参数，保存重载后仍共享。确定性编码状态和生成状态维度相同，因此可以直接绑定；这不代表二者的状态语义已经被独立证明相同。

## 4. 增加参数投向哪些模块

分别只扩大INIT、UPDATE、DECODER、编码ODE、生成ODE，再比较两个ODE一起扩大、三个MLP一起扩大以及所有网络一起扩大。每种投向均设约45,000和90,000参数两个档位，避免仅挑少数模块测更高预算。

由于网络参数量是整数函数，各组总量会略有差别，报告实际值和每新增万参数的收益，不用“约2倍”替代真实计数。INIT只接收一个年龄标量，为了分配同样新增预算，它的隐藏宽度会很大；这组检验的是把参数投入年龄映射是否划算，并不假定它是合理的默认架构。

## 5. 潜在维度消融

保持原隐藏宽度154/118，分别将潜在状态从6维增加到12、24、48维。维度增大也会增加相邻投影和两个向量场的参数，所以这三组不能单独解释成纯粹的维度效应。

再加入latent_24_matched：24维状态、MLP宽度77、ODE宽度73，恰好22,545参数。它比较同一预算下“更宽的网络、6维状态”和“较窄网络、24维状态”的分配方式；仍不把它称为宽度完全不变的单因素实验。

## 6. 完整24组矩阵

各组除表中改动外，均继承主模型设置。

| 配置 | 组别 | 具体改动 | 可训练参数 |
|---|---|---|---:|
| baseline | reference | {} | 22,545 |
| shared_tied | sharing | {"shared_vector_field": true} | 21,005 |
| shared_tied_l2x2 | sharing_regularization_control | {"shared_vector_field": true, "shared_l2_multiplier": 2.0} | 21,005 |
| shared_matched | sharing_parameter_control | {"shared_vector_field": true, "mlp_width": 159, "ode_width": 188} | 22,545 |
| init_2x | allocation_2x | {"init_width": 2967} | 45,049 |
| update_2x | allocation_2x | {"update_width": 517} | 45,051 |
| decoder_2x | allocation_2x | {"decoder_width": 556} | 45,057 |
| encoder_ode_2x | allocation_2x | {"encoder_ode_width": 1849} | 45,048 |
| generator_ode_2x | allocation_2x | {"generator_ode_width": 1849} | 45,048 |
| both_odes_2x | allocation_2x | {"ode_width": 983} | 45,035 |
| all_mlps_2x | allocation_2x | {"mlp_width": 333} | 45,099 |
| all_networks_2x | allocation_2x | {"mlp_width": 308, "ode_width": 236} | 45,017 |
| init_4x | dose_response_4x | {"init_width": 8608} | 90,177 |
| update_4x | dose_response_4x | {"update_width": 1245} | 90,187 |
| decoder_4x | dose_response_4x | {"decoder_width": 1362} | 90,193 |
| encoder_ode_4x | dose_response_4x | {"encoder_ode_width": 5321} | 90,184 |
| generator_ode_4x | dose_response_4x | {"generator_ode_width": 5321} | 90,184 |
| both_odes_4x | dose_response_4x | {"ode_width": 2719} | 90,171 |
| all_mlps_4x | dose_response_4x | {"mlp_width": 691} | 90,207 |
| all_networks_4x | dose_response_4x | {"mlp_width": 616, "ode_width": 472} | 89,961 |
| latent_12 | latent_fixed_width | {"latent": 12} | 29,097 |
| latent_24 | latent_fixed_width | {"latent": 24} | 42,201 |
| latent_48 | latent_fixed_width | {"latent": 48} | 68,409 |
| latent_24_matched | latent_parameter_control | {"latent": 24, "mlp_width": 77, "ode_width": 73} | 22,545 |

## 7. 用什么标准判断“效益最大”

每组仍按验证整体误差选检查点。随后用这个检查点的验证未来预测误差进行配置排序，主指标越低越好；同时看验证整体误差和历史重建误差。测试集只在训练结束后评估，不根据测试排名继续调整实验。

比较模块投向时，先在相近45k或90k预算内比较绝对误差，再计算：

```text
每新增万参数收益 = (baseline验证预测误差 − 本组验证预测误差)
                   ÷ (本组参数量 − baseline参数量) × 10000
```

正数表示有效改善，负数表示变差。减少参数或等参数的组不套用这个新增参数公式，而比较同预算误差和节省量。最终自动报告验证表现最好的结构、45k档最优投向、单位新增参数收益最高的组，并导出参数量—验证误差图。

## 8. 同时检查正则、收敛和计算代价

所有组固定L2系数。更宽的网络有更多受罚参数，所以容量改变时，总正则强度也会变化。每轮记录训练MSE、CCE、原始和加权L2、梯度范数、耗时、峰值已分配显存，并在SwanLab展示。

若大模型训练重构误差也不下降，应先检查优化和正则；若训练好而验证差，考虑过拟合；若最佳轮次接近300轮上限，则当前预算可能不足。不能仅凭某一行最终分数就判断该模块没有价值。耗时是本次并行环境的实际成本，不当作脱离环境的通用速度基准。

## 9. 执行安排与自动化

先等待当前年龄/零初始化消融队列成功结束并释放四张卡，不打断既有任务。随后以GPU 0、1、2、3各一个worker运行24组；每组先通过独立的小规模CUDA端到端检查，再开始完整训练。所有模型实验禁止CPU回退。

代码在source_snapshot中冻结，并记录源码和数据SHA256。各任务从冻结副本加载，防止长队列期间修改工作区改变后面的实验。配置和输出均独立，不覆盖旧结果。完成任务复跑时核对配置和数据后跳过；遇到已有未完成历史则保留并明确报错，没有自动断点续训。

单个任务失败会记录failure.json，worker继续处理其他任务；最后汇总失败状态，不把未完成矩阵标记成功。每分钟更新summary.md/summary.json，全部完成后自动生成排名、图表和设备/参数量/检查点重载核对。

实际运行入口：

```bash
bash scripts/run_forward_age_scaling.sh
```

只准备配置：`bash scripts/run_forward_age_scaling.sh --prepare-only`。

## 10. 输出与已做验证

- [实验汇总](../results/forward_age_scaling/summary.md)：随队列更新的进度、验证和最终测试结果。
- `paper_odeae/results/forward_age_scaling/manifest.json`：24组配置与真实参数量。
- `configs/`：各组完整参数；`source_snapshot/`：冻结源码。
- `checks/`：每组的小规模训练检查；各正式目录保存配置、数据摘要、设备、分模块参数量、历史、检查点和最终指标。

CUDA检查已覆盖全部24组真实参数量、有限梯度、截止点锚定、共享参数身份、L2只计一次与显式两倍对照、共享模型保存重载。原正向初始化与模型检查也通过。完整训练结果仍需等待队列执行，不用小规模检查的分数代替正式效果。
