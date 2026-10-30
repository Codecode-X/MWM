# Level 3: Irregular Continuous-Time Treatment–Confounder Feedback Benchmark

这是固定非线性数学机制的合成医疗纵向反事实基准，不代表真实疾病。生成过程没有 Neural ODE 或拟合模型。生成代码属于数据管理员工具；模型运行时仅挂载 `public/`，不得向模型提供此生成源码、config、manifest、diagnostics 或 oracle_private。

## 运行与复现

在项目根目录：

```bash
conda activate ode
python -m pip install -r CausalMed/requirements.txt
export PYTHONPATH="$PWD/CausalMed${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
pytest -c CausalMed/pytest.ini CausalMed/synthetic_cf_level3/tests -q
# 先做总计100位患者验证，含诊断、全部患者重放和评估器自检
python -m synthetic_cf_level3.generate --smoke --out-dir CausalMed/outputs/smoke_100
# 主基准 gamma=1，默认 10000/2000/2000
python -m synthetic_cf_level3.generate \
  --config CausalMed/synthetic_cf_level3/configs/level3_v1.yaml \
  --out-dir CausalMed/outputs/synthetic_cf_level3
# 四个独立总体，每组14000人
python -m synthetic_cf_level3.generate --all-gammas --out-dir CausalMed/outputs/level3_suite
python -m synthetic_cf_level3.validation --data-dir CausalMed/outputs/level3_suite/gamma_1
```

也可 `cd CausalMed` 后运行 `python -m synthetic_cf_level3...`。默认配置的输出路径相对当前工作目录，建议始终显式指定 `--out-dir`。`--workers` 控制数据准备进程数，`--gamma` 覆盖治疗混杂强度。已存在的数据目录不会被覆盖。四组正式基准的 seed 依次为 `seed + 0/1000000/2000000/3000000`，按患者严格划分；相同 seed、软件版本、配置可重现数据内容。manifest 运行耗时不要求字节相同。

`SeedSequence([seed, split_stream, patient_index, purpose])` 中用途 1/2/3/4 分别为初始化、检查时间、观测噪声、治疗抽样，互相独立。多进程顺序、反事实分支顺序不影响数据。各患者的测量误差与治疗随机数独立。

配置 [configs/level3_v1.yaml](configs/level3_v1.yaml) 包含人数、各 split 独立流、年限、检查间隔、分叉时间、预测窗口、采样频率、所有动力学系数、初始分布、噪声、行为策略系数、positivity、策略阈值分位数、延迟年限、成本、随机种子、输出路径与诊断规模。V1 的策略定义及五种策略 ID 固定；调整机制参数会得到新版本基准，不应与固定 V1 数值混用。

DOP853 每个治疗区间独立求解，rtol=1e-8，atol=1e-10，使用 dense output 精确查询切换点与任意区间内时刻。D=0 时导数为正；E=0/1 时导数指向 [0,1] 内部。L 可以为负；D/E 不做裁剪。非有限数值、积分失败、不连续的治疗区间及越界状态明确报错。内部提供 `PiecewiseTrajectory.state(times)` 和 `.treatment(times)`。

## 治疗和时间语义

所有时间以**年**为单位。临床检查从 t=0 开始，间隔独立 U(0.25,0.75)，最后一个实际检查小于8年；8年仅为模拟边界，不强行增加小于0.25年的临床间隔。最后一次检查决定的治疗延续至8年。

每次检查先积分到当前时刻，再生成观测，再决定治疗。治疗区间为 `[t_start,t_end)`；切换点查询治疗时返回新治疗。8年边界上的治疗值仅用作连续轨迹末端约定，没有新决策。

事实医生只使用当前 Y_obs、X2_obs、过去 Y_obs 的下降率、既往治疗。静态 c 公开但本版医生分数未额外使用 c。行为代码只接收 `ObservedHistory`，没有隐藏 D/E/alpha/beta 输入。X2 是准确测量风险 L；D 仅通过带噪 X1/Y 观测。propensity 只在私有审计。

测试分叉为首次 `time >= 3` 的实际检查，包含该次观测。历史治疗只包含 `t_start < fork_time` 的已完成区间；**不包含分叉后的事实治疗决定**。所有分支复制完全相同的 D/L/E 状态、静态参数、历史和未来检查日程。

## 公开数据字段

| 文件 | 字段 / 含义 |
|---|---|
| train_patients / val_patients / test_patients.parquet | patient_id, c |
| train_visits / val_visits.parquet | patient_id, visit_index, time, Y_obs, X1_obs, X2_obs |
| train_treatments / val_treatments.parquet | patient_id, t_start, t_end, treatment；绝对时间 |
| test_histories.parquet | 与 visits 相同，最多到 fork，含 fork 检查 |
| test_history_treatments.parquet | 与 treatments 相同，仅 fork 之前治疗 |
| test_future_visits.parquet | patient_id, time；外生未来检查时间，**无未来结果**；包含 fork |
| test_queries.parquet | patient_id, fork_time, policy_id, prediction_times=[1,2,3], trajectory_times（fork至8年边界逐月网格，含精确1/2/3年及末端）, future_visit_times（绝对年） |
| policy_specs.json | 五种公开干预规则、训练集阈值、延迟年限、固定成本 |
| train/val/test_sequences.jsonl | 可直接读取的患者级变长序列，包含 c、times、observations、observation_mask、treatment_intervals |

公开特征顺序始终是 `[Y_obs, X1_obs, X2_obs]`，本版没有特征缺失，因此 mask 为全1。没有公开 Y_clean、真实 propensity、隐藏状态或个体 alpha/beta。训练/验证均只有一条完整事实轨迹。

```python
from synthetic_cf_level3.dataset_adapter import PublicDataset
train = PublicDataset('path/to/mounted_public', 'train')
patient = train[0]
# {patient_id,c,times,observations,observation_mask,treatment_intervals}
test = PublicDataset('path/to/mounted_public', 'test')
queries = test.queries()
sequences = list(train.shared_matched_sequences())
```

适配器从公开训练访问计算 zscore，所有 split 共用训练 offset/scale，未使用测试统计。`shared_matched_sequences()` 保留 `subject_id/split/ages/values/mask` 字段，额外带 c/treatment_intervals；ages 表示**经过年数**，并非真实年龄。可与 `paper_odeae.data.Prefixes/collate` 的序列结构衔接，但原固定 ADNI 输出头、年龄初始化和未接受治疗输入的动力学不能直接当作本基准的干预模型。这里不改原模型或训练代码。

后续连续时间模型须把治疗作为控制输入，准确在已知治疗切换/检查时刻推进，并在每次未来检查预测 X2 后执行策略，保存由自身推演产生的治疗区间。归一化模型的 X2 必须先还原到原单位再比较风险阈值。不得将真实 oracle 风险或事实未来风险传给动态策略。

适配器只打开固定白名单 public 文件，不打开 config/manifest/private，并拒绝多余字段与逃逸目录的 Parquet symlink。**此限制不是任意 Python 代码的安全沙箱**；实际盲测需单独进程/容器，仅挂载公开目录。诊断图同样含私有信息，不得作为模型输入。

## 五种策略

| ID | 规则 |
|---|---|
| 0 | 从分叉检查后永不治疗 |
| 1 | 从分叉检查后始终治疗 |
| 2 | fork+1 年及以后第一次实际检查开始持续治疗 |
| 3 | 各自世界当次 X2 > 公开 q50 时治疗 |
| 4 | 各自世界当次 X2 > 公开 q80 时治疗 |

q50/q80 使用**全部训练集事实检查 X2** 的分位数，一经计算固定。策略3/4未来决策逐次更新，严格使用自身分支的 X2，阈值相等时不治疗。通过 `counterfactual_policies.decide()` 可在模型侧执行相同公开策略；该模块不依赖生成器。反事实积分延续到8年，私有 cf_visits/cf_treatments 保存全部未来检查与治疗，标准连续真值保存 fork 后至8年边界每月采样和精确1/2/3年，末端不足一个月时保留精确边界。

## 私有审计与真值（仅数据管理员 / evaluator）

| 文件 | 内容 |
|---|---|
| cf_trajectories.parquet | patient_id, policy_id, relative_time, Y_cf_true, X1_cf_true, X2_cf_true, treatment；无噪声、逐月 |
| cf_outcomes.parquet | patient_id, policy_id, relative_time, Y_cf_true, treatment_duration, utility；精确1/2/3年 |
| cf_treatments.parquet | patient_id, policy_id, t_start, t_end, treatment；**相对 fork 年数** |
| cf_visits.parquet | patient_id, policy_id, relative_time, X2_cf_true, treatment；各分支实际决策时刻 |
| hidden_states.parquet | patient_id, split, visit_index, time, D_true, L_true, E_true, Y_clean, propensity；全部事实临床检查 |
| cf_hidden_states.parquet | patient_id, policy_id, relative_time, D_true, L_true, E_true |
| patient_parameters.parquet | patient_id, split, c, alpha_true, beta_true, D0, L0, E0 |
| factual_visits / factual_treatments.parquet | 全 split 的完整事实数据，含测试未来，仅供审计 |
| evaluation_spec.json | horizons, gamma, treatment_cost, tie_break, policies |

X1_cf_true 是无测量误差的 X1 真值（等于隐藏 D），只在 oracle 中；X2_cf_true 为分支风险 L，Y_cf_true 为无噪声结局。cf_trajectories.treatment 是该采样时刻右连续的治疗值，不应代替精确区间积分计算成本。

## 评价

默认逐月预测文件必须覆盖所有 test 患者、五种策略及公开 trajectory_times，含 relative_time=0。CF-RMSE 全轨迹不计已知 fork 时刻，单独报告精确1/2/3年误差。时间按1e-9年匹配，重复、缺失、额外行、未知患者和非有限数值报错，不通过 inner join 静默忽略缺失预测。

最低预测列为 `patient_id, policy_id, relative_time, Y_pred`。完整策略评价再提供每个1/2/3年终点的 `treatment_duration_pred`（由模型自身分支预测，0到T年之间），或提供自身预测的完整区间文件。

```bash
python -m synthetic_cf_level3.evaluation \
  --predictions predictions.parquet \
  --predicted-treatments predicted_intervals.parquet \
  --oracle-dir CausalMed/outputs/level3_suite/gamma_1/oracle_private \
  --out-dir CausalMed/outputs/my_evaluation/gamma_1
# 若只提交精确终点预测，显式使用 --mode horizons；此时无全轨迹 RMSE
# 四组模型预测均评价后汇总，输出比较CSV与误差-vs-gamma图
python -m synthetic_cf_level3.evaluation \
  --suite eval/gamma_0 eval/gamma_0.5 eval/gamma_1 eval/gamma_2 \
  --out-dir CausalMed/outputs/my_evaluation/comparison
```

预测区间文件字段 `patient_id,policy_id,t_start,t_end,treatment`，**相对 fork 年数**，从0开始、连续无缝、二元治疗，必须覆盖最长窗口。`treatment_duration_pred` 与 `--predicted-treatments` 二选一。不提供时依然报告 CF-RMSE、sqrt(PEHE)、ATE Error，但 Policy Regret 明确为 unavailable，不会用 oracle 治疗时长替代预测时长。

各终点报告五种策略 CF-RMSE、每个策略相对 Never 的 sqrt(PEHE) 和 ATE Error，以及含成本策略后悔与策略选择准确率。U=Y-kappa*治疗年数，默认 kappa=2.0；效用相同时选最小 policy_id。默认成本固定，不根据测试结果调参。若将 `policies.tune_cost=true`，仅在训练患者中模拟所有策略，根据最优策略分布熵选择候选成本，然后冻结成本才生成测试分支；最终成本与校准记录均写入 config/manifest。策略重合会造成真实效用相同，最优策略计数遵循固定 tie-break。

`diagnostics/evaluator_selfcheck.json` 使用真值作为预测验证指标应为零，属于**评估器算术自检，不是已训练模型性能**。没有拟合或运行任何具体模型，因此不伪造四组模型误差结果；`--suite` 会在真实预测提交后生成混杂强度–模型误差图。

## 自动验收和诊断

测试覆盖边界不变量、积分容差、精确分段、因果治疗效应、风险对未来病情的消融、医生只用可观测历史、positivity、延迟检查规则、全部分支分叉一致、分支局部动态决策、篡改事实未来不改变反事实、随机顺序/进程数不变性、患者级隔离、无 private 挂载加载、字段注入拒绝、评估键完整性、已知误差的指标解析值、预测治疗时长要求与区间覆盖。

每次生成后默认运行数据级验收：**全部患者**按实际治疗重放，Y_clean 最大误差<1e-6；成对 do(A=0/1) 与风险干预；切断 L->D 和 L->propensity 的消融；全量有限性、非负性、propensity范围、分叉状态逐位一致、训练阈值重算、动态决策逐项重算、不规则间隔、跨划分患者互斥和观测随机复现。

诊断用独立患者总体，在 gamma=0/.5/1/2 下重用外生随机数作控制比较。朴素效应是当前接受治疗和未治疗患者的下一检查 Y_clean 年变化率之差；真实效应在每位患者当前状态和同一个下一检查时间做 do(1)-do(0) 配对，然后平均年变化率。报告风险、D、当前Y和未来Y的治疗组差异。gamma=0 的有限样本偏差不要求精确0，也不强制偏差单调。

每组 `diagnostics/` 包含 `validation_report.md`、`statistics.json`、因果配对效应表、混杂诊断表与七幅 PNG/SVG：

1. 因果结构；2. 随机5位患者事实观测/风险/治疗；3. 同一患者五条反事实；4. 风险层真实治疗概率与实际比例；5. 治疗效应分布及与c关系；6. 分支风险/检查/治疗/结局动态时序；7. 混杂强度与选择关联及朴素偏差。

manifest 记录最终参数、阈值/成本、种子/流、源码 SHA256、数据文件 SHA256、软件版本与设备。生成过程是 CPU SciPy 数据准备，tensor_devices 与 GPU assignments 为空，明确没有模型实验；后续模型实验必须按项目要求在四张4090 GPU进行，不得把本数据验收视作模型性能验证。

## 当前交付的最终参数

基础 V1 配置保持 κ=2。最终四组数据使用 `configs/level3_v1_calibrated.yaml` 的共同 κ=3，成本仅由主基准训练总体的前500位患者模拟选定，未使用任何验证或测试患者。校准命令为：

```bash
python -m synthetic_cf_level3.calibration --patients 500 \
  --out-config CausalMed/synthetic_cf_level3/configs/level3_v1_calibrated.yaml \
  --report CausalMed/outputs/training_cost_calibration.json
python -m synthetic_cf_level3.generate \
  --config CausalMed/synthetic_cf_level3/configs/level3_v1_calibrated.yaml \
  --all-gammas --out-dir CausalMed/outputs/level3_suite_replica
```

此共同成本在四种 gamma 中固定，避免成本差异干扰跨混杂强度比较。运行 `bash CausalMed/reproduce.sh NEW_OUTPUT_DIR` 可一次执行测试、训练成本校准、100人 smoke test、四组完整生成与逐组验收。
