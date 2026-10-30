# Level 3 合成数据验收报告

状态：**PASS**；gamma=1.0；患者数={'train': 60, 'val': 20, 'test': 20}。

这些检查是 SciPy 合成数据准备、数学验证和统计汇总，未训练或评估任何模型。

全部 100 位患者事实治疗重放的最大无噪声 Y 误差：0；
更严格容差的最大隐藏状态差异：5.45e-09。没有裁剪任何隐藏状态。

## 成对因果干预与消融

```json
{
  "paired_n": 20,
  "ate_3y": 9.423355806551298,
  "ite_sd": 1.8446621850618696,
  "ite_quantiles": {
    "p05": 6.745839781090374,
    "p50": 9.277681738305539,
    "p95": 12.433675105596711
  },
  "A_to_next_L_mean": -0.06541004352364657,
  "L_to_next_D_mean": 0.0013784409195880087,
  "controlled_L_to_propensity_mean": 0.027005181620970965,
  "ablated_L_to_D_max": 6.674660824046441e-12,
  "ablated_L_to_propensity_max": 0.0
}
```

## 治疗选择偏差（固定下一检查终点，按每年结局变化率比较）

| gamma | 风险–治疗相关 | 朴素效应 | 成对真实效应 | 偏差 |
|---|---|---|---|---|
| 0.0 | -0.0263 | 1.6438 | 1.2434 | 0.4003 |
| 0.5 | 0.0597 | 1.3901 | 1.2015 | 0.1886 |
| 1.0 | 0.1545 | 1.2214 | 1.1875 | 0.0339 |
| 2.0 | 0.1464 | 1.2545 | 1.1308 | 0.1237 |

同一诊断总体在四种 gamma 下重用外生随机因素，用于控制比较；四套正式基准患者总体使用独立种子。
朴素估计采用当次实际治疗的下一次检查变化率，两种潜在结局用同一检查间隔配对。
gamma=0 的有限样本偏差不要求精确为零，偏差不要求随 gamma 单调变化。

## 真实最优策略数量（含治疗成本）

```json
{
  "1.0": {
    "0": 18,
    "1": 2,
    "2": 0,
    "3": 0,
    "4": 0
  },
  "2.0": {
    "0": 13,
    "1": 7,
    "2": 0,
    "3": 0,
    "4": 0
  },
  "3.0": {
    "0": 6,
    "1": 8,
    "2": 5,
    "3": 1,
    "4": 0
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
