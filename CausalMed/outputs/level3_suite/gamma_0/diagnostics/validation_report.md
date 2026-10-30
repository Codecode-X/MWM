# Level 3 合成数据验收报告

状态：**PASS**；gamma=0.0；患者数={'train': 10000, 'val': 2000, 'test': 2000}。

这些检查是 SciPy 合成数据准备、数学验证和统计汇总，未训练或评估任何模型。

全部 14000 位患者事实治疗重放的最大无噪声 Y 误差：0；
更严格容差的最大隐藏状态差异：8.19e-09。没有裁剪任何隐藏状态。

## 成对因果干预与消融

```json
{
  "paired_n": 100,
  "ate_3y": 10.08349037747039,
  "ite_sd": 1.7958546367611976,
  "ite_quantiles": {
    "p05": 7.260892767844053,
    "p50": 9.985446951460194,
    "p95": 12.865770118875783
  },
  "A_to_next_L_mean": -0.06618580522409444,
  "L_to_next_D_mean": 0.0013798455320607871,
  "controlled_L_to_propensity_mean": 0.0,
  "ablated_L_to_D_max": 1.1231682250922859e-11,
  "ablated_L_to_propensity_max": 0.0
}
```

## 治疗选择偏差（固定下一检查终点，按每年结局变化率比较）

| gamma | 风险–治疗相关 | 朴素效应 | 成对真实效应 | 偏差 |
|---|---|---|---|---|
| 0.0 | -0.0069 | 1.6402 | 1.2720 | 0.3683 |
| 0.5 | 0.0602 | 1.4337 | 1.2369 | 0.1968 |
| 1.0 | 0.1072 | 1.3611 | 1.2159 | 0.1452 |
| 2.0 | 0.1426 | 1.2338 | 1.1441 | 0.0897 |

同一诊断总体在四种 gamma 下重用外生随机因素，用于控制比较；四套正式基准患者总体使用独立种子。
朴素估计采用当次实际治疗的下一次检查变化率，两种潜在结局用同一检查间隔配对。
gamma=0 的有限样本偏差不要求精确为零，偏差不要求随 gamma 单调变化。

## 真实最优策略数量（含治疗成本）

```json
{
  "1.0": {
    "0": 1903,
    "1": 97,
    "2": 0,
    "3": 0,
    "4": 0
  },
  "2.0": {
    "0": 1195,
    "1": 795,
    "2": 0,
    "3": 10,
    "4": 0
  },
  "3.0": {
    "0": 441,
    "1": 1145,
    "2": 284,
    "3": 124,
    "4": 6
  }
}
```

## 自动验收检查

- PASS: finite states/observations
- PASS: D/E invariants
- PASS: solver tolerance
- PASS: causal treatment effect
- PASS: paired A->L and L->D, L->propensity with ablations
- PASS: selection and overlap
- PASS: all-patient factual replay
- PASS: fork exact consistency
- PASS: branch-local dynamic decisions
- PASS: train-only thresholds
- PASS: strictly irregular schedule
- PASS: deterministic replay
- PASS: public schema / cutoff isolation

## 数据隔离与使用范围

模型只挂载 public/；oracle_private/ 权限 0700，适配器不读 manifest/config/private。
生成代码和私有审计文件对数据管理员可见，目录分离本身不隔离任意用户代码。
8 年是积分边界而非额外检查，临床访问间隔均在 [0.25,0.75] 年。
反事实积分到 8 年；标准评估保存分叉后至8年边界的逐月真值及精确 1/2/3 年终点。
全部临床观测在本次治疗前生成；分叉历史不包含 fork 时刻的事实治疗。

图表见 plots/；完整统计、风险分层重叠、策略切换和最优策略计数见 statistics.json。
