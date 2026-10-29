# forward_age 共享向量场与参数扩容实验

用户已停止本轮消融。已完成结果保留；stopped_by_user 表示训练中断，cancelled_by_user 表示取消尚未开始的任务。中断任务没有完整测试结果。

全部任务使用forward_age、seed 0、相同数据与训练设置；四卡各运行一个独立实验。先用同一执行方式重新训练baseline，旧DDP结果只作外部参照。

每个检查点按验证整体误差选择；优先根据该检查点的验证预测误差比较方案。测试结果只在完整训练后填写，不用测试集决定下一组结构。未完成任务的相对收益仍会变化。

| 组别 | 参数量 | 状态 | 轮数 | 最佳验证整体 | 对应验证预测 | 测试预测 | 测试整体 | 每新增万参数的验证预测收益 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 22,545 | complete | 164 | 0.91986 | 1.26217 | 1.16690 | 0.86583 | — |
| shared_tied | 21,005 | complete | 160 | 0.90556 | 1.21469 | 1.14694 | 0.86156 | — |
| shared_tied_l2x2 | 21,005 | complete | 182 | 0.91037 | 1.26113 | 1.14384 | 0.85692 | — |
| shared_matched | 22,545 | complete | 96 | 0.88722 | 1.16130 | 1.13926 | 0.85063 | — |
| init_2x | 45,049 | complete | 121 | 0.89988 | 1.18872 | 1.13577 | 0.86733 | 0.03264 |
| update_2x | 45,051 | stopped_by_user | 86 | 0.90133 | 1.20761 | — | — | 0.02424 |
| decoder_2x | 45,057 | stopped_by_user | 74 | 0.90516 | 1.21173 | — | — | 0.02241 |
| encoder_ode_2x | 45,048 | stopped_by_user | 70 | 0.91073 | 1.20601 | — | — | 0.02496 |
| generator_ode_2x | 45,048 | stopped_by_user | 13 | 1.00420 | 1.29962 | — | — | -0.01664 |
| both_odes_2x | 45,035 | cancelled_by_user | — | — | — | — | — | — |
| all_mlps_2x | 45,099 | cancelled_by_user | — | — | — | — | — | — |
| all_networks_2x | 45,017 | cancelled_by_user | — | — | — | — | — | — |
| init_4x | 90,177 | cancelled_by_user | — | — | — | — | — | — |
| update_4x | 90,187 | cancelled_by_user | — | — | — | — | — | — |
| decoder_4x | 90,193 | cancelled_by_user | — | — | — | — | — | — |
| encoder_ode_4x | 90,184 | cancelled_by_user | — | — | — | — | — | — |
| generator_ode_4x | 90,184 | cancelled_by_user | — | — | — | — | — | — |
| both_odes_4x | 90,171 | cancelled_by_user | — | — | — | — | — | — |
| all_mlps_4x | 90,207 | cancelled_by_user | — | — | — | — | — | — |
| all_networks_4x | 89,961 | cancelled_by_user | — | — | — | — | — | — |
| latent_12 | 29,097 | cancelled_by_user | — | — | — | — | — | — |
| latent_24 | 42,201 | cancelled_by_user | — | — | — | — | — | — |
| latent_48 | 68,409 | cancelled_by_user | — | — | — | — | — | — |
| latent_24_matched | 22,545 | cancelled_by_user | — | — | — | — | — | — |

已完成组按验证预测误差排序：shared_matched, init_2x, shared_tied, shared_tied_l2x2, baseline。


收益为baseline误差减本组误差，正数代表改善；分母是新增可训练参数。共享和等参组不以该比值比较。所有组固定L2系数，容量变化也会改变正则总量，需结合summary.json中的训练重构与正则分量解释。单种子结果不等于显著差异。
