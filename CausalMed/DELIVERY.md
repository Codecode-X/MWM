# CausalMed Level 3 交付与验收

完整代码、配置、四组最终数据、字段说明、自动测试、因果报告与诊断图均已交付。原 shared_matched 模型和原始实验文件未修改。

- 正式患者共56,000位：每种 gamma 的 train/val/test 为10,000/2,000/2,000。
- 共8,000位测试患者，40,000条配对反事实分支，精确终点120,000行。
- 最终配置100人 smoke test、24项自动测试和四组完整数据验收均通过。
- 全患者事实治疗重放最大无噪声结局误差为0；共核对104个数据/配置文件 SHA256，并验证全部生成源码 SHA256。
- 五种策略，逐月真值覆盖分叉至8年边界，另保存精确1/2/3年终点。
- 四组共同成本 κ=3，仅用500位训练患者模拟选定；独立重跑校准报告完全一致。基础配置保留默认 κ=2。
- 全部运行属于 SciPy 数据准备、数学检查和统计汇总，没有训练/推理任何模型。后续模型实验必须使用四张4090 GPU。

## 主要入口

- [完整使用说明和字段字典](synthetic_cf_level3/README.md)
- [基础配置](synthetic_cf_level3/configs/level3_v1.yaml) / [实际最终配置](synthetic_cf_level3/configs/level3_v1_calibrated.yaml)
- [最小模型客户端](model_client/README.md)：仅将 model_client/ 与对应 public/ 挂载给模型，避免暴露真实方程与真值。
- [自动测试结果](outputs/pytest.log) / [独立验证CLI结果](outputs/validation_cli_smoke.log)
- [训练成本校准证据](outputs/training_cost_calibration.json) / [精确复现证据](outputs/calibration_replay.json)
- [机器可读验收汇总](outputs/delivery_summary.json) / [四组清单](outputs/level3_suite/suite_manifest.json)
- [运行环境版本](environment_versions.json) / [锁定依赖](requirements-lock.txt)

## 完整数据与因果验收

| gamma | 患者数 | 事实重放最大误差 | 风险–治疗相关 | 风险–propensity Spearman | 报告 |
|---|---:|---:|---:|---:|---|
| 0 | 14,000 | 0 | 0.0010 | 0.0000 | [验收报告](outputs/level3_suite/gamma_0/diagnostics/validation_report.md) |
| 0.5 | 14,000 | 0 | 0.1789 | 0.9644 | [验收报告](outputs/level3_suite/gamma_0.5/diagnostics/validation_report.md) |
| 1 | 14,000 | 0 | 0.2574 | 0.9647 | [验收报告](outputs/level3_suite/gamma_1/diagnostics/validation_report.md) |
| 2 | 14,000 | 0 | 0.2997 | 0.9643 | [验收报告](outputs/level3_suite/gamma_2/diagnostics/validation_report.md) |

每组 public/ 为模型唯一可读数据；oracle_private/（0700）和 diagnostics/ 只供数据管理员与离线 evaluator 使用。
每组 diagnostics/plots/ 含七幅 PNG 和七幅 SVG，共28种诊断图。

## 3年真实最优策略数量（固定 κ=3）

| gamma | Never | Always | Delayed | Aggressive | Conservative |
|---|---:|---:|---:|---:|---:|
| 0 | 441 | 1145 | 284 | 124 | 6 |
| 0.5 | 447 | 1051 | 302 | 184 | 16 |
| 1 | 451 | 1035 | 307 | 190 | 17 |
| 2 | 487 | 902 | 383 | 206 | 22 |

## 复现

```bash
bash CausalMed/reproduce.sh /absolute/path/to/new_output
```

生成、独立验证与 evaluator 的单步命令见使用说明。评估器支持全轨迹 CF-RMSE、1/2/3年 CF-RMSE、各策略相对 Never 的 sqrt(PEHE)/ATE Error 和含成本 Policy Regret。
Policy Regret 必须使用模型自身预测的治疗时长或区间；缺失时明确不可用。已提供四种 gamma 的评价汇总与误差曲线程序，待后续模型提交预测后调用。
diagnostics/evaluator_selfcheck.json 是用 oracle 数值验证评估器的零误差自检，不能解释为模型性能。

100人最终 smoke test 位于 outputs/smoke_100_calibrated/。早期 outputs/smoke_100/ 为基础成本开发检查，不是最终参数的数据。
patient_id 在每个 gamma 目录内唯一，合并不同独立基准时须加 gamma 命名空间。

最小模型客户端已在项目目录之外独立加载主基准的2,000位测试患者与10,000条查询，并确认生成器包不可导入：[隔离客户端检查](outputs/client_isolation_check.json)。此检查属于数据接口检查，没有执行任何模型。
