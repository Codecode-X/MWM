# 当前论文实验的四 GPU 执行目录

本轮主模型采用性别 INIT、28 维共享状态、α=0.1 残差 UPDATE、无 weak prior、0.01 Jacobian 和 0.02 ODE L2。ADNI 为 44,875 参数，MIMIC 为 43,280 参数。CausalMed γ=1 的公开资料没有性别，适配版使用公开静态 c 初始化，参数为 31,314；仅观测更新使用残差门控，治疗条件位移保持原定义。

四张物理 GPU 0–3 按独立作业调度。每个正式训练作业单卡，ADNI batch32、MIMIC/CausalMed batch128，不使用梯度累计。包括恢复在内最多训练 150 轮。模型训练、验证、测试和所有导数、PCA、近邻分析使用 CUDA；文件、统计和绘图由 CPU 处理。

## 进度与结果

截至 2026-11-08，89 项作业全部完成，failed=0。三项冻结诊断已经修复并补跑，主报告包含 60 张带解释的图和 34 张带说明的表。[核验记录](validation.json) 覆盖结果文件、CUDA 诊断、源码哈希、图表和本地链接。

[PDF 阅读版](../reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.pdf) 保留全部图表、中文说明和公式，并提供章节书签。导出信息、文档哈希与各图所在页码见 [PDF 导出记录](../reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.pdf.json)。重新导出使用：

~~~bash
/home/xjh/miniforge3/envs/ode/bin/python -m pip install -r scripts/requirements-pdf-export.txt
/home/xjh/miniforge3/envs/ode/bin/python scripts/export_publication_pdf.py
~~~

- [论文实验结果、表格与逐图解释](../reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.md)
- [失败诊断修复与数值核验](REPAIR.zh-CN.md)
- [89 项作业及依赖](manifest.json)
- [GPU 队列](queue.json)
- [进度摘要](progress_summary.json)
- [旧 GPU 程序的中断记录](preemption_record.json)
- [47 种真实数据模型的 CUDA 预检](preflight_checks/summary.json)
- [CausalMed 公开数据隔离与动作检查](preflight_cf/checks/checks.json)
- [控制器日志](controller.log)
- [实时报告日志](report.log)

两个真实数据集先各训练主模型与无 Jacobian 对照。同预算 baseline 和 CausalMed 排在主要比较阶段；其余核心消融随后运行。状态、临床速度、流形、转化、鲁棒性和成本任务等待各自检查点。INIT、完整 α 扫描和容量扩展在最后执行。

训练结束后重新载入最佳检查点，先验证重载误差，再计算测试指标。实时报告将完成的结果写回论文清单，尚未完成的测试指标留空。每组保存配置、CUDA 设备、完整优化器/调度器/RNG 状态、检查点、日志和 SwanLab。

## 实现与源码

[models.py](models.py) 包含性别 INIT、残差 UPDATE、顺序/反序、双场、无编码 ODE、注意力和目标函数；[train.py](train.py) 完成单卡训练与检查；[queue.py](queue.py) 管理 GPU 锁、任务依赖和恢复；[analysis.py](analysis.py)、[analysis_extra.py](analysis_extra.py)、[baseline_analysis.py](baseline_analysis.py) 执行冻结模型分析；[report.py](report.py) 只做统计、绘图、结果回填与检查点归档。

第一批四组使用冻结源码 v1，后续实验使用 v5。2026-11-08 对三项失败诊断补跑：稳定性使用 v6，ADNI/MIMIC Jacobian 效用使用 v7。新版本将切线流计算独立封装，并修复观测噪声 dtype；全部分析读取原已训练权重。源码及哈希保存在各自 snapshot 中，原冻结版本保留。

若某组在训练中出现数值不稳定，记录原始失败日志和实际停止轮次，再使用验证集已选出的有效最佳检查点进行 CUDA 测试。该组结果明确标注不稳定停止；测试阶段不更新参数。实现错误和预检失败单独记为 failed。

## 断点恢复

在项目根目录、ode 环境中运行：

~~~bash
python -u -m publication_sex_init.queue
~~~

控制器会拒绝重复运行。重启时先核对原 worker 进程并接管仍存活的任务；中断的训练从本轮独立目录恢复，累计预算保持 150 轮。实时表格进程使用：

~~~bash
python -u -m publication_sex_init.report --watch
~~~

训练途中每 100 次更新保存一次完整恢复状态，并响应停止信号保存当前 batch 边界。训练批次顺序和历史截止点由患者、轮次与固定种子确定；恢复时重新生成当前 batch 的输入 dropout。最佳检查点另行归档，供固定患者的跨轮次敏感性诊断使用。
