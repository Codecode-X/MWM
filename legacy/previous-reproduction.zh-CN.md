# 本机复现

项目： https://github.com/SankaVaas/LatentODE-PatientMonitoring
源码版本见 `source_commit.txt`。原始模型和结果备份在 `upstream/`。

环境：Miniforge Conda，环境名 `ode`，Python 3.11，PyTorch 2.5.1 + CUDA 12.1。
固定依赖见 `environment.yml`，完整安装版本见 `requirements-lock.txt`。

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ode
cd ~/LatentODE-PatientMonitoring
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
python reproduction/run_pipeline.py preprocessing
python reproduction/run_pipeline.py train
python reproduction/run_pipeline.py evaluate
```

包装脚本只设置种子 42、无界面绘图和模块搜索路径，调用原始脚本；训练使用原始超参数（最多 100 轮，20 轮无提升提前停止）。
日志位于 `logs/`，新模型位于 `../models/`，结果位于 `../results/`。

数据使用仓库随附的 100 名患者合成演示数据，不是真实 MIMIC 临床数据。`/data` 对当前用户不可写，因此小型随附数据暂保留项目目录。
上游未固定训练种子，且评估时仍会随机采样潜变量；本次指标不保证等于 README 示例。

## 本次实测结果

全部依赖安装成功，`pip check`、GPU 检查、前向验证、预处理、重新训练和评估均通过。
训练共 61 轮，第 41 轮最佳模型验证 AUROC 为 1.0。数据划分：训练 70、验证 14、测试 16 名合成患者。

校准测试结果：AUROC 0.41818，AUPRC 0.35821，Accuracy 0.3125，F1 0.47619，阈值 0.01，混淆矩阵 `[[0, 11], [0, 5]]`。
**本次已复现完整运行流程，但未复现上游 README 的 AUROC 0.8727。** 未为匹配指标调整种子、超参数或筛选测试结果。

上游实现的限制：
- `src/model.py` 在 eval 模式下依然随机采样潜变量，不同评估调用使用不同的随机样本；种子相同也会受到调用顺序影响。
- 训练脚本未固定随机种子，上游依赖仅声明最低版本，没有原始完整环境锁文件。
- 上游注意力可视化函数没有启用 attention weights 返回，因此本次没有生成 `attention_weights.png`；其主程序仍会打印该文件名。其余 7 张图已生成。
- 原始示例数据集很小，验证和测试指标波动大；验证 AUROC 不能代表测试性能。

源代码保持上游原样。完整日志及安装日志见 `logs/`；训练轮数和最佳模型元信息见 `run-summary.json`。
