# Ablation：α=0.1主模型的模块消融

## 1. 要验证什么

当前主模型使用28维潜在状态，按时间从最早到最近吸收患者历史。INIT根据最后一条已知历史记录处的年龄给出初始状态，初始状态放在首次历史记录处；ODE负责两次检查之间的连续演化，UPDATE在检查到来时修正状态。

每次校正固定采用10%的UPDATE建议，保留90%的ODE演化后状态：

\[z_k^+=0.9z_k^-+0.1U_\phi(z_k^-,y_k).\]

读完历史后，模型从历史截止点状态分别向过去回顾、向未来预测。编码和生成共用一个ODE速度场。训练同时包含重建、weak prior、Jacobian和ODE参数L2。

本轮以这套完整模型为参照，每次只去掉一项监督或改变一项建模设置，回答四个问题：读取历史的方向是否重要？校正前预测监督是否有用？Jacobian正则是否有用？相同ODE参数预算下，一个统一速度场和两个任务速度场哪种效果更好？

## 2. 实验矩阵与参数公平性

| 组别 | 历史读取 | weak prior权重 | Jacobian权重 | ODE | α | 本轮执行 |
|---|---|---:|---:|---|---:|---|
| 完整主模型 | 正序 | 0.01 | 0.01 | 统一场 | 0.1 | 复用现有结果 |
| 反序消融 | 最近读到最早 | 0.01 | 0.01 | 统一场 | 0.1 | 复用CUDA 2/3正在进行的实验 |
| 去掉weak prior | 正序 | 0 | 0.01 | 统一场 | 0.1 | 已停止，保留已完成ADNI |
| 去掉Jacobian | 正序 | 0.01 | 0 | 统一场 | 0.1 | 空闲CUDA自动候补 |
| 双速度场 | 正序 | 0.01 | 0.01 | 编码场＋生成场 | 0.1 | 空闲CUDA自动候补 |

**每一组都严格保持ADNI 44,875参数、MIMIC 43,280参数。** 不填充无用参数；预检检查实际唯一参数量、梯度和两个场是否真正独立。INIT、UPDATE、Decoder、28维状态及门控系数保持一致。

### 2.1 双速度场怎样做到完全等参数

完整模型的ODE是28→188→28，隐藏层使用Tanh。双场组把同一笔ODE预算分成两部分：编码ODE负责吸收历史时的演化，生成ODE负责从截止点向过去和未来生成。两个场之间不共享任何参数。

| 速度场 | 结构 | 输出层偏置 | 参数量 |
|---|---|---|---:|
| 主模型统一场 | 28→188→28 | 有，28维 | 10,744 |
| 双场：历史编码场 | 28→94→28 | 无 | 5,358 |
| 双场：历史/未来生成场 | 28→94→28 | 有，28维 | 5,386 |
| 双场合计 | 两个完全独立的场 | | **10,744** |

两组的ODE预算相同，差别是把容量集中到一个共享函数，还是分配给两个专用函数。双场组不改变其他网络宽度。编码场不设输出偏置，正好省下28个参数，使总量精确匹配。

INIT、UPDATE、Decoder从相同seed取得相同初始张量。两个94宽场使用相同的初始权重，生成场的输出偏置为0；它们之后独立更新。初始化取原188宽场的前94个隐藏单元，输出权重乘√2，补偿隐藏单元减半后的初始速度尺度。预检记录两个新场之间、以及新场与主模型初始预测之间的实际差异。

### 2.2 方向组复用什么

正序主模型从首次检查读取到历史截止点，再分别回顾过去和预测未来。反序组从最近检查读取到首次检查，得到已融合整段历史的首次检查状态，再沿同一ODE向未来生成完整历史和未来。

反序组保留一条使用相同网络和α=0.1的正序辅助路径，只计算原weak prior。辅助路径不提供主生成起点。两种方向的参数量、年龄输入、监督目标、损失权重和数据划分相同。

## 3. 损失怎样计算

\[\mathcal L=\mathcal L_{\mathrm{rec}}+\lambda_p\mathcal L_{\mathrm{prior}}+\lambda_J\mathcal L_J+0.02\mathcal L_{\mathrm{ODE-L2}}.\]

**重建项。** 所有有效历史目标和未来目标都参与数值MSE＋类别交叉熵。每个原始字段按当前batch的有效目标数归一化，再对字段求和。缺失目标不计入损失；训练输入dropout只遮蔽输入，不删除真实监督目标。所有组保留这一项和它的原有梯度路径。

**weak prior。** 历史第2次及以后，模型先用ODE演化到检查时刻，在吸收这次观测之前预测它；随后才UPDATE。该项使用相同的MSE＋交叉熵及有效字段掩码。去掉weak prior组只将λp从0.01改成0，仍保存原始诊断分量。

**Jacobian。** 对主生成轨迹中有效的28维状态，精确计算速度场对状态的28×28 Jacobian，784项平方求和，再对状态求平均；采样状态停止梯度。去掉Jacobian组只将λJ从0.01改为0。双场组在相同生成状态上分别计算两个场，再取场间平均：

\[\mathcal L_J=\frac1{|\mathcal F|}\sum_{f\in\mathcal F}\frac1{|\mathcal Z|}\sum_{z\in\mathcal Z}\|J_f(z)\|_F^2.\]

**ODE L2。** 对全部独立ODE参数的平方求和，系数固定0.02。统一场的别名只计数一次；双场分别计数，但两场总参数仍为10,744。这里不对双场L2再除以2，保证每个独立参数接受相同的惩罚系数。

## 4. 数据、训练与测试

| 项目 | ADNI | MIMIC |
|---|---|---|
| 数据缓存 | paper_odeae/cache/local_available.pkl | /data/MIMIC/derived/latentode_iv31/dataset.pkl |
| 训练 / 验证 / 测试 | 2,093 / 393 / 414人 | 21,517 / 4,649 / 4,536条患者序列 |
| 字段 / 展开输出维度 | 40 / 49 | 31 / 44 |
| 时间单位 | 年 | 小时 |
| 数值归一化 | 训练集min-max | 训练集min-max |
| 训练历史构造 | 枚举每位患者全部非空历史前缀 | 每轮按患者与epoch固定种子，从6/12/18/24小时截止点采样 |
| 验证与主测试 | 原全部前缀评测 | 原24小时历史上下文，预测后续至48小时 |
| batch / 梯度累积 | 32 / 无 | 128 / 无，保留末批13人 |
| 评测batch | 32 | 每次32，串行复用原四个固定患者分片 |
| 随机种子 | 0 | 0 |
| 学习率 / 衰减 | Adam，0.003；每2000步乘0.97 | 相同 |
| 梯度裁剪 | 全模型范数10 | 相同 |
| ODE | Tsit5，rtol=1e−4，atol=1e−6 | 相同 |
| 新训练预算 | 最大150轮，至少50轮，验证整体40轮不改善早停 | 相同 |
| 选模 | 验证整体重建误差最低 | 相同 |

全部新消融使用Conda ode环境，每组单卡训练。调度器扫描物理CUDA 0、1、2、3，只有GPU锁可获得且没有计算进程时才派发任务；每张卡同一时刻只运行一组。weak prior消融已按用户要求停止，已完成的ADNI结果保留。其余待跑组自动候补，已有训练不中断。任务换到新GPU时，先在实际GPU重新预检，再执行原冻结训练代码，接入SwanLab。
 
每组结束后重新载入best.pt，核对验证误差，再计算测试历史回顾、测试未来预测、测试整体以及逐字段指标。MIMIC额外保留数值标准化MSE和6/12小时历史上下文测试。测试目标不进入UPDATE。

基线复用已有主模型。ADNI主模型最大150轮、实际早停70轮；MIMIC按此前指定复用已训练237轮的正序组，选择第233轮最佳验证检查点，不再补跑正序150轮。新的MIMIC消融和反序组上限仍为150轮，下面列出各组实际完成轮数。

## 5. CUDA核验与当前进度

| 新任务 | 状态 | 物理GPU | 设计总参数 | 实测总参数 | 独立ODE参数 | 核验 |
|---|---|---:|---:|---:|---:|---|
| adni_no_prior | complete | 0 | 44875 | 44875 | 10744 | passed |
| mimic_no_prior | stopped_by_user | 2 | 43280 | 43280 | 10744 | passed |
| adni_no_jacobian | complete | 0 | 44875 | 44875 | 10744 | passed |
| mimic_no_jacobian | complete | 0 | 43280 | 43280 | 10744 | passed |
| adni_separate_fields | complete | 1 | 44875 | 44875 | 10744 | passed |
| mimic_separate_fields | running | 2 | 43280 | 43280 | 10744 | passed |

预检不执行optimizer.step()。它验证去损失组的初始输出与完整主模型逐位一致、仅移除了目标损失的梯度；验证双场预算精确相等、两场参数无交集、两场均获得梯度；同时检查未来观测值不会影响模型输出、全部实际任务参数的梯度有限。

### 5.1 训练前的数值核验

| 任务 | 移除指定损失后的梯度相对误差 | 两个初始场的最大速度差 | 初始预测相对主模型的差异 |
|---|---:|---:|---:|
| adni_no_prior | 0 | — | 0（逐位相同） |
| mimic_no_prior | 0 | — | 0（逐位相同） |
| adni_no_jacobian | 1.69e-09 | — | 0（逐位相同） |
| mimic_no_jacobian | 8.55e-10 | — | 0（逐位相同） |
| adni_separate_fields | — | 0 | 0.181705 |
| mimic_separate_fields | — | 0 | 1.031854 |

双场组以两个相同的94宽函数开始训练；表中相对主模型的差异来自将一个188宽函数重新分配为两个94宽函数。上表描述训练前的初始化和梯度，训练后的指标见后两节。

## 6. ADNI结果

| 模型 | 状态 | 参数量 | 已完成 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |
|---|---|---:|---:|---:|---:|---:|
| 完整主模型 | complete | 44875 | 70 / 30 | 0.427375 | 1.032291 | 0.683622 |
| 反序消融 | complete | 44875 | 58 / 18 | 0.477745 | 1.081566 | 0.733542 |
| 去掉 weak prior | complete | 44875 | 70 / 30 | 0.424737 | 1.020335 | 0.677058 |
| 去掉 Jacobian 正则 | complete | 44875 | 70 / 30 | 0.418643 | 1.026275 | 0.675746 |
| 编码/生成双速度场（等参数） | complete | 44875 | 84 / 44 | 0.423455 | 1.057298 | 0.690687 |

| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 |
|---|---:|---:|---:|---:|
| 完整主模型 | 30 | 0.413037 | 0.999456 | 0.665897 |
| 反序消融 | 18 | 0.444435 | 1.053120 | 0.706938 |
| 去掉 weak prior | 30 | 0.405482 | 0.998405 | 0.661391 |
| 去掉 Jacobian 正则 | 30 | 0.406955 | 1.009824 | 0.667594 |
| 编码/生成双速度场（等参数） | 44 | 0.403291 | 0.998429 | 0.660562 |

![ADNI训练、验证与梯度曲线](results/alpha01_components/curves/adni_ablation.png)


曲线和表格读取各组实际日志。正在训练的组展示当前最佳验证检查点，测试列在训练结束后填写；纵向虚线标出150轮。

- 反序消融的测试未来误差减主模型为+0.053664。
- 去掉 weak prior的测试未来误差减主模型为-0.001050。
- 去掉 Jacobian 正则的测试未来误差减主模型为+0.010368。
- 编码/生成双速度场（等参数）的测试未来误差减主模型为-0.001027。

## 7. MIMIC结果

| 模型 | 状态 | 参数量 | 已完成 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |
|---|---|---:|---:|---:|---:|---:|
| 完整主模型 | complete | 43280 | 237 / 233 | 0.514434 | 1.810591 | 1.127823 |
| 反序消融 | complete | 43280 | 150 / 125 | 0.562428 | 1.833658 | 1.164118 |
| 去掉 weak prior | 复用已停止训练 | 43280 | 101 / 96 | 0.628275 | 1.819368 | 1.192842 |
| 去掉 Jacobian 正则 | complete | 43280 | 150 / 138 | 0.585277 | 1.825794 | 1.172475 |
| 编码/生成双速度场（等参数） | running | 43280 | 109 / 103 | 0.692706 | 1.825596 | 1.230480 |

| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |
|---|---:|---:|---:|---:|---:|
| 完整主模型 | 233 | 0.506466 | 1.810933 | 1.126278 | 0.830239 |
| 反序消融 | 125 | 0.552019 | 1.818688 | 1.154081 | 0.875601 |
| 去掉 weak prior | — | — | — | — | — |
| 去掉 Jacobian 正则 | 138 | 0.567768 | 1.806117 | 1.156196 | 0.857741 |
| 编码/生成双速度场（等参数） | — | — | — | — | — |

![MIMIC训练、验证与梯度曲线](results/alpha01_components/curves/mimic_ablation.png)


曲线和表格读取各组实际日志。正在训练的组展示当前最佳验证检查点，测试列在训练结束后填写；纵向虚线标出150轮。

- 反序消融的测试未来误差减主模型为+0.007755。
- 去掉 Jacobian 正则的测试未来误差减主模型为-0.004816。

## 8. 原始记录与运行方式

[实验矩阵与源码指纹](results/alpha01_components/manifest.json) · [GPU候补队列](results/alpha01_components/queue.json) · [已有方向实验](../mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation/direction_alpha01/queue.json)

- adni_no_prior：[配置](results/alpha01_components/adni_no_prior/launch_config.json) · [预检](results/alpha01_components/adni_no_prior/checks.json) · [训练记录](results/alpha01_components/adni_no_prior/)
- mimic_no_prior：[配置](results/alpha01_components/mimic_no_prior/launch_config.json) · [预检](results/alpha01_components/mimic_no_prior/checks.json) · [训练记录](results/alpha01_components/mimic_no_prior/)
- adni_no_jacobian：[配置](results/alpha01_components/adni_no_jacobian/launch_config.json) · [预检](results/alpha01_components/adni_no_jacobian/checks.json) · [训练记录](results/alpha01_components/adni_no_jacobian/)
- mimic_no_jacobian：[配置](results/alpha01_components/mimic_no_jacobian/launch_config.json) · [预检](results/alpha01_components/mimic_no_jacobian/checks.json) · [训练记录](results/alpha01_components/mimic_no_jacobian/)
- adni_separate_fields：[配置](results/alpha01_components/adni_separate_fields/launch_config.json) · [预检](results/alpha01_components/adni_separate_fields/checks.json) · [训练记录](results/alpha01_components/adni_separate_fields/)
- mimic_separate_fields：[配置](results/alpha01_components/mimic_separate_fields/launch_config.json) · [预检](results/alpha01_components/mimic_separate_fields/checks.json) · [训练记录](results/alpha01_components/mimic_separate_fields/)

各组保存独立的best.pt、last.pt、优化器与调度器状态、随机数状态、history.json、hardware.json、SwanLab记录和最终metrics.json。原实验检查点与日志不覆盖。

```bash
bash scripts/run_main_alpha01_ablation.sh
```

重复启动时先检查已有结果和单GPU锁，完成的组直接复用，未完成的组从自身last.pt继续；累计训练轮次不超过150。

## 9. 中断后的续训记录

本次进程中断后，从各组最近完成一轮的last.pt恢复模型、Adam、学习率调度器、最佳验证分数、早停计数和累计优化步数。原训练脚本在每轮开始按固定种子重新设置随机数和采样顺序；中断轮中尚未保存的部分从该轮开头重算。

| 实验 | 物理GPU | 检查点已完成轮次 | 续训起始轮次 | 已保存优化步数 | 总轮次上限 |
|---|---:|---:|---:|---:|---:|
| adni_no_prior | 0 | 39 | 40 | 15288 | 150 |
| mimic_reverse | 3 | 41 | 42 | 6929 | 150 |
| update_alpha_00 | 1 | 108 | 109 | 18252 | 150 |

已完成的ADNI反序组和ADNI主模型直接复用。恢复时CUDA 0继续ADNI去prior，后续任务纳入同一队列；此前取消的MIMIC正序150轮重训保持取消。模型结构、损失、batch size和等参数预算均沿用中断前配置。

恢复时数据盘尚未挂载，首次启动没有执行训练更新。数据盘恢复后，重新核对MIMIC缓存SHA256与原实验一致，再启动续训。

[恢复清单与检查点备份](results/alpha01_components/recoveries/20261104T040807Z/manifest.json)。备份保留了续训前的best.pt、last.pt、训练历史、配置和SwanLab记录。

## 10. 空闲GPU自动候补

weak prior消融已按用户要求停止。ADNI去prior组此前已完成，保留其全部结果；MIMIC去prior组停止于已保存的第101轮，不再自动恢复。释放的GPU由其余候补组自动领取；Jacobian、双速度场和反序实验继续原计划。
 
| 待派发顺序 | 实验 | 当前状态 | 物理GPU |
|---:|---|---|---|
| 1 | mimic_no_prior | stopped_by_user | 2 |
| 2 | adni_no_jacobian | complete | 0 |
| 3 | mimic_no_jacobian | complete | 0 |
| 4 | adni_separate_fields | complete | 1 |
| 5 | mimic_separate_fields | running | 2 |
| 6 | adni_no_prior | complete | 0 |

每组仍使用独立的模型和Adam，ADNI batch=32、MIMIC batch=128，不使用梯度累计。GPU分配只改变物理设备编号；实际张量设备、参数量和设备编号写入hardware.json及SwanLab。所有续训累计最多150轮。

[调度变更与接管记录](results/alpha01_components/dispatch_revision.json) · [实时队列](results/alpha01_components/queue.json)

## 11. ADNI去掉Jacobian：第30轮临时测试

按当前最佳验证结果选定第30轮，复制独立参数快照后在CUDA 0评测。现有训练继续；本次不执行优化更新。正式训练完成后的最终测试单独保存。

| 测试区域 | MSE＋CE | 数值MSE之和 | 类别CE之和 | 数值标准化MSE均值 |
|---|---:|---:|---:|---:|
| 历史回顾 | 0.40695493 | 0.32465523 | 0.08229969 | 0.38420591 |
| 未来预测 | 1.00982358 | 0.53508030 | 0.47474328 | 0.59375061 |
| 整体 | 0.66759381 | 0.41221866 | 0.25537515 | 0.47030903 |

[完整评测报告与逐字段表](results/alpha01_components/adni_no_jacobian/temporary_evaluations/epoch030_20261104T075535Z/report.zh-CN.md) · [全部指标JSON](results/alpha01_components/adni_no_jacobian/temporary_evaluations/epoch030_20261104T075535Z/metrics.json) · [逐字段CSV](results/alpha01_components/adni_no_jacobian/temporary_evaluations/epoch030_20261104T075535Z/per_field_metrics.csv)

完整报告包括40个字段在历史、未来和整体区域的损失，数值字段原量纲MAE、RMSE和标准化误差，类别字段准确率、宏平均F1、混淆矩阵，以及校正前预测和速度场Jacobian诊断值。

## 12. 再次中断后的恢复

数据盘恢复后，未完成的组从最近完整轮次继续，恢复模型、Adam、学习率调度器、累计优化步数和早停状态。单轮中未保存的部分从该轮起点重算。设备由空闲GPU调度器重新分配，模型和训练超参数保持原值。

| 实验 | 已保存轮次 | 续训起始轮次 | 已保存优化步数 | 原物理GPU | 当前物理GPU |
|---|---:|---:|---:|---:|---:|
| adni_separate_fields | 14 | 15 | 5488 | 0 | 1 |
| mimic_no_jacobian | 58 | 59 | 9802 | 1 | 0 |
| mimic_separate_fields | 1 | 2 | 169 | 2 | 2 |
| mimic_reverse | 135 | 136 | 22815 | 3 | 3 |

本轮累计仍最多150轮。weak prior消融与MIMIC正序150轮补跑保持停止；已完成的ADNI结果直接复用。

## 13. Jacobian正则的实际作用测试

使用两组固定的第30轮检查点，比较相同历史测量噪声下的预测偏移、小截止点状态扰动在未来ODE中的放大量，以及按真实预测时长分组的测试误差。模型不重新训练。

当前状态：complete；阶段：全部完成。

[诊断进度](results/jacobian_utility_adni/progress.json) · [实验配置](results/jacobian_utility_adni/manifest.json)

[完整实验结果、曲线和机制解释](results/jacobian_utility_adni/report.zh-CN.md) · [定量指标](results/jacobian_utility_adni/metrics.json)

| 实测效果 | 保留Jacobian正则 | 去掉Jacobian正则 |
|---|---:|---:|
| 5年ODE误差放大倍数，中位数 | 1.259699 | 1.309237 |
| 同样5%测量噪声引起的MMSE预测偏移RMSE，分 | 0.081051 | 0.086272 |
| 超过5年真实预测的MMSE RMSE，分 | 2.668789 | 2.946112 |
