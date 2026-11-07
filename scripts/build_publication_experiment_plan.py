"""Assemble the publication checklist from local artifacts, without model execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.md"
PLAN = ROOT / "reproduction/no_weak_prior_rerun_plan.json"
if PLAN.exists() and json.loads(PLAN.read_text()).get("execution_started"):
    raise SystemExit("实验已经启动；请运行 python -m publication_sex_init.report 更新实际结果。")
parts, evidence, tasks = [], [], []


def read(path):
    return json.loads((ROOT / path).read_text())


def put(text):
    parts.append(text.strip() + "\n")


def table(headers, rows):
    assert all(len(row) == len(headers) for row in rows)
    put("\n".join(["| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |"]
                   + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]))


def fmt(v):
    return "" if v is None else f"{v:.6f}"


def eligible(m):
    return m.get("status") in {"complete", "adni_common_protocol_complete",
                              "capacity_matched_ddp_completed"} and m.get("epochs") is not None and m["epochs"] <= 150


def row(m, label, source):
    hw = m["hardware"]
    p = hw["parameters"]
    base = [label, f"{p:,}", f'{hw.get("trainable_parameters", p):,}']
    if not eligible(m):
        return base + ["", "", "", "", "待预算内评测"]
    test = m["result"]["test"]
    scores = [test[k]["reconstruction_error"] for k in ["interpolation", "forecast", "overall"]]
    evidence.append({"model": label, "source": source, "run": m.get("path"),
                     "source_sha256": hashlib.sha256((ROOT / source).read_bytes()).hexdigest(),
                     "epochs_run": m["epochs"], "best_epoch": m["best_epoch"],
                     "history": scores[0], "future": scores[1], "overall": scores[2]})
    return base + [f'{m["epochs"]} / {m["best_epoch"]}', *map(fmt, scores), "已有合格结果"]


def core(title, body, variant, source, after):
    put(title)
    put(body)
    caption, detail = after.split("。", 1)
    put(caption + "：" + variant + "。")
    rows = []
    for dataset, params in [("ADNI", 44875), ("MIMIC", 43280)]:
        for name in ["M0", variant]:
            rows.append([dataset, name, f"{params:,}", "", "", "", ""])
    table(["数据集", "模型", "参数量", "最佳轮次", "历史 H ↓", "未来 F ↓", "整体 O ↓"], rows)
    put(detail)
    put("方案与实现来源：" + source)


def task(ident, datasets, priority, operation, deps=(), **kw):
    tasks.append({"id": ident, "datasets": datasets, "priority": priority,
                  "operation": operation, "depends_on": list(deps),
                  "required": True, "status": "planned", **kw})


adni_source = "paper_odeae/results/baselines_matched_latent2x/comparison.json"
mimic_source = "mimic_experiment/results/baselines_matched_latent2x/comparison.json"
msource = "mimic_experiment/results/baselines_single_cuda/comparison.json"
adni = {m["key"]: m for m in read(adni_source)["models"]}
mimic = {m["key"]: m for m in read(mimic_source)["models"]}
mnative = {m["key"]: m for m in read(msource)["models"]}
cf_source = "CausalMed/outputs/gamma1_model_comparison/comparison.json"
cf = {m["name"]: m for m in read(cf_source)["models"]}

put(r"""# 论文实验清单：性别 INIT、共享 ODE 与残差观测更新

更新日期：2026-11-05。本文按论文实验主题组织，每节给出实验介绍、比较设置、结果表和图表交付。空白单元格表示对应结果尚未获得。只填写符合本轮模型定义与评价标准的结果。

## 一、模型定义与共同实验标准

### 1.1 主模型 M0

患者检查按实际时间顺序输入。模型先把性别映射为首次历史检查时刻的潜在状态，再用共享 ODE 描述检查之间的连续变化。新检查到达时，UPDATE 产生候选状态，残差门控吸收其中一部分修正。吸收全部已知历史后，从历史截止点向未来预测或向过去回顾，两类生成路径继续使用同一个速度场。

\[
z_0^-=\operatorname{INIT}_\eta(\mathrm{sex}),\qquad
\dot z=f_\theta(z),\qquad
z_k^+=0.9z_k^-+0.1U_\phi(z_k^-,y_k).
\]

训练保留原来的历史回顾与未来预测联合重建目标。数值字段使用 MSE，类别字段使用 CE，按有效观测掩码逐字段归一化后求和。历史与未来的全部有效时间点合并计算主重建项；辅助项为 0.01 Jacobian 正则与 0.02 ODE 参数 L2。

\[
\mathcal L_{\mathrm{M0}}=
\mathcal L_{\mathrm{recon,MSE}}+\mathcal L_{\mathrm{recon,CE}}
+0.01\,\mathbb E_z\|J_f(z)\|_F^2
+0.02\,\mathcal R_{\mathrm{ODE}}.
\]

weak prior 系数为 0。历史回顾主损失仍然完整端到端反向传播。Jacobian 沿用现有实现：有效生成状态先 detach，每个 28×28 Jacobian 的平方元素求和，再在状态间求平均，只对向量场参数施加该正则梯度。
""")
table(["项目", "固定设置"], [
    ["潜在状态", "28 维"],
    ["INIT", "1→159→28；F=0、M=1；初始状态放在首次历史检查时刻"],
    ["共享 ODE", "28→188→28，Tanh；编码、未来预测、历史回顾共享全部参数"],
    ["UPDATE", "保留原网络，固定 α=0.1 残差更新"],
    ["Decoder", "保留原数值字段和类别输出头"],
    ["weak prior / Jacobian / ODE L2", "0 / 0.01 / 0.02"],
    ["ADNI / MIMIC 参数量", "44,875 / 43,280"],
    ["本轮主模型状态", "两个数据集均需从固定种子重新训练"],
])
put("""ADNI 缓存使用 PTGENDER−1，即 M=0、F=1；INIT 入口需映射为 F=0、M=1。MIMIC 缓存已为 F=0、M=1。类别预测目标继续沿用各自缓存编码，只转换 INIT 输入。实际进入 INIT 的标量为 0/1；若复用内部除以 100 的年龄接口，应补偿缩放并审计真实输入张量。性别只取预测时已经可用的静态资料或首次历史记录，缺失策略根据训练集确定。

实现来源：[模块设计](../ablation/Ablation.zh-CN.md)、[INIT 输入与 CUDA 审计](../mimic_experiment/init_covariate_ablation.py)、[当前模型方法](CURRENT_MODEL_METHOD.zh-CN.md)。""")
put("""### 1.2 数据与训练标准

ADNI 检验多年疾病演化，MIMIC 检验小时尺度的不规则临床序列，CausalMed 检验动作条件下的反事实推演。三个数据集分别固定患者划分，归一化只使用训练集。验证整体误差选择最佳检查点，重新载入后进行独立测试。""")
table(["项目", "ADNI", "MIMIC", "CausalMed"], [
    ["训练 / 验证 / 测试患者", "2,093 / 393 / 414", "21,517 / 4,649 / 4,536", "10,000 / 2,000 / 2,000"],
    ["版本", "local_available，40 字段 / 展开 49 维", "chronology_v2，31 字段 / 展开 44 维", "γ=1，公开事实数据"],
    ["归一化 / 时间单位", "训练 min-max / 年", "训练 min-max / 小时", "训练 z-score / 年"],
    ["训练历史", "枚举所有非空历史前缀", "每轮采样 6/12/18/24 小时截止点", "原患者级历史截止点采样"],
    ["主测试", "各历史前缀之后的真实随访", "前 24 小时历史，预测后续至 48 小时", "首次达到 3 年的检查后分叉，五策略至 8 年边界"],
    ["新模型 batch / 梯度累计", "32 / 无", "128 / 无", "128 / 无"],
    ["训练上限", "150 轮，含续训累计", "150 轮，含续训累计", "150 轮，含续训累计"],
    ["停止规则", "至少 50 轮；验证整体 40 轮无改善早停", "相同", "相同"],
])
put("""新主模型与消融使用 Adam，学习率 0.003，每 2,000 步乘 0.97，输入 dropout=0.1，梯度范数裁剪上限 10；Tsit5 的 rtol=1e−4、atol=1e−6。baseline 保留各自方法的概率目标与求解器。首轮固定 seed=0，并按患者配对 bootstrap 给出结果差值区间；患者重抽样不替代训练种子的变异评价。

独立作业排队使用物理 GPU 0、1、2、3，每张卡默认一个作业。模型训练、验证、测试、ODE、自动微分、PCA、近邻与几何诊断均在 CUDA 上执行。CPU 负责数据准备、文件、统计和绘图。训练接入 SwanLab，记录分项损失、梯度、学习率与验证曲线。

数据来源：[ADNI/MIMIC 合同](ADNI_MIMIC_DATASETS.zh-CN.md)、[CausalMed 公开输入](../CausalMed/latent2x_experiment/data.py)、[150 轮政策](EXPERIMENT_POLICY.json)。""")
put("""### 1.3 结果与参数比较口径

H 是历史区间误差，F 是未来预测误差，O 是全部有效时间点重新汇总的整体误差。各数值字段计算 MSE，各类别字段计算 CE，字段内取有效观测均值，再按字段求和。因此 O 不等于 H、F 的简单平均。ODE-RNN 的 H 为读取该时刻观测前的因果预测，其余模型的 H 为历史重构，主要共同预测比较采用 F。

同预算组匹配实际唯一参数：ADNI 44,875、MIMIC 43,280。HealthFlux 保留原生 256+128 维并列出容量。超过 150 轮的 baseline 先在前 150 轮验证记录中确定应选轮次，再取得相应权重进行 CUDA 复评；缺少该权重时在新目录重训。完成核验前结果留空。

所有填入数字均需关联配置、数据哈希、源码、检查点 SHA256、实际轮次与结果 JSON。同一个 M0 在后续各节复用同一检查点，不重复训练参照组。""")

headers = ["模型", "总参数", "可训练参数", "完成 / 最佳轮次", "历史 H ↓", "未来 F ↓", "整体 O ↓", "状态"]
labels = [("ode_ae", "ODE-AE（概率，同预算）"), ("ode_rnn", "ODE-RNN（同预算）"),
          ("latent_ode", "Latent ODE（同预算）")]
put("""## 二、主要比较实验

### 2.1 ADNI：纵向临床预测

每个模型读取相同患者、相同历史前缀，以相同缺失掩码评价未来随访。主要比较包括 M0、概率与确定性 ODE-AE、ODE-RNN、Latent ODE 和 HealthFlux-KL。优先比较同预算模型，再用原容量表区分方法与容量的影响。每组交付混合字段误差、40 字段的原单位 MAE/RMSE 或类别准确率，重点列出 MMSE、CDR-SB、ADAS-Cog 与诊断。未来目标从不进入观测 UPDATE。

表 1：ADNI 主要比较。""")
rows = [["M0（性别 INIT）", "44,875", "44,875", "", "", "", "", "待新训练"]]
rows.append(row(adni["ode_ae"], labels[0][1], adni_source))
rows.append(["确定性 ODE-AE（同预算）", "44,875", "44,875", "", "", "", "", "待新训练"])
rows.extend(row(adni[k], label, adni_source) for k, label in labels[1:])
for label, folder in [("HealthFlux-KL（KL=1）", "common_task"),
                      ("HealthFlux-KL（KL=0.01）", "paper_kl")]:
    source = f"healthflux/results/adni_comparison/{folder}/seed_0/metrics.json"
    m = read(source)
    h = read(f"healthflux/results/adni_comparison/{folder}/seed_0/hardware.json")
    model = {"status": m["status"], "epochs": m["epochs_run"], "best_epoch": m["best_epoch"],
             "hardware": h, "result": m}
    rows.append(row(model, label, source))
table(headers, rows)
put("表 2：ADNI 原容量补充比较。")
native_adni_rows = [row(adni[k], label, adni_source) for k, label in [
    ("ode_ae_native", "ODE-AE（概率，原容量）"), ("ode_rnn_native", "ODE-RNN（原容量）"),
    ("latent_ode_native", "Latent ODE（原容量）")]]
det_folder = "paper_odeae/results/capacity_matched/deterministic_22545/seed_0"
det_metrics = read(det_folder + "/metrics.json")
native_adni_rows.insert(1, row({
    "status": det_metrics["status"], "epochs": det_metrics["epochs_run"],
    "best_epoch": det_metrics["best_epoch"], "hardware": read(det_folder + "/hardware.json"),
    "result": det_metrics, "path": det_folder,
}, "确定性 ODE-AE（22,545 参数）", det_folder + "/metrics.json"))
table(headers, native_adni_rows)
put("""主要图为各模型共同患者的预测轨迹、逐字段误差和患者配对差值。确定性同预算模型保留原编码与独立生成机制，宽度调整至 44,875 个实际参数。两种 HealthFlux KL 分别列出，最终设置由验证集选择。

结果来源：[同预算 ADNI](../paper_odeae/results/baselines_matched_latent2x/summary.md)、[原容量 ADNI](../paper_odeae/results/baselines_full/summary.md)、[HealthFlux](../healthflux/results/adni_comparison/summary.md)。""")
put("已填写的同预算 baseline 中，ODE-RNN 的未来误差为 1.032329，概率 ODE-AE 为 3.238572；HealthFlux 的两种 KL 设置分别为 1.534999 和 1.603200。后续主比较据此评价新 M0 的预测质量与参数效率。")
put("""### 2.2 MIMIC：联合临床预测与保持最近值

主测试以 ICU 前 24 小时历史预测之后 24 小时。除混合误差外，27 个数值字段的原单位 MSE 除以各自训练标准差平方后等权平均，得到数值标准化 MSE。加入保持最近值基线，历史从未测到的字段用训练均值或众数。完整输出包含全部 31 字段、GCS 分类准确率和临床单位误差，用于判断动力学模型对具体预测任务的收益。

表 3：MIMIC 主要比较。""")
rows = [["M0（性别 INIT）", "43,280", "43,280", "", "", "", "", "待新训练"]]
rows.append(row(mimic["ode_ae"], labels[0][1], mimic_source))
rows.append(["确定性 ODE-AE（同预算）", "43,280", "43,280", "", "", "", "", "待预算内评测"])
rows.extend(row(mimic[k], label, mimic_source) for k, label in labels[1:])
rows.append(row(mnative["healthflux_kl"], "HealthFlux-KL（KL=0.01）", msource))
table(headers, rows)
put("表 4：MIMIC 原容量补充比较。")
table(headers, [row(mimic[k], label, mimic_source) for k, label in [
    ("ode_ae_native", "ODE-AE（概率，原容量）"), ("ode_rnn_native", "ODE-RNN（原容量）"),
    ("latent_ode_native", "Latent ODE（原容量）")]])
put("表 5：MIMIC 临床单位与数值标准化指标，GCS 使用现有三个类别头。")
clinical_rows = []
for label, model, source in [
    ("M0", None, None), ("ODE-AE（概率，同预算）", mimic["ode_ae"], mimic_source),
    ("确定性 ODE-AE（同预算）", None, None), ("ODE-RNN（同预算）", mimic["ode_rnn"], mimic_source),
    ("Latent ODE（同预算）", mimic["latent_ode"], mimic_source),
    ("HealthFlux-KL", mnative["healthflux_kl"], msource),
]:
    if model and eligible(model):
        test = model["result"]["test"]
        clinical = test["clinical_forecast"]
        scores = [test["mean_standardized_numeric_mse"], clinical["heart_rate"]["mae"],
                  clinical["map"]["mae"],
                  *[clinical[f"gcs_{name}"]["accuracy"] for name in ["eye", "motor", "verbal"]]]
        clinical_rows.append([label, *map(fmt, scores)])
        evidence.append({"model": label, "source": source, "clinical_scores": dict(zip(
            ["std_numeric_mse", "heart_rate_mae", "map_mae", "gcs_eye_accuracy",
             "gcs_motor_accuracy", "gcs_verbal_accuracy"], scores))})
    else:
        clinical_rows.append([label, "", "", "", "", "", ""])
carry = mnative["healthflux_kl"]["result"]["test"]
clinical_rows.append(["保持最近值", fmt(carry["carry_forward_mean_standardized_numeric_mse"]),
                      fmt(carry["carry_forward"]["heart_rate"]["mae"]),
                      fmt(carry["carry_forward"]["map"]["mae"]),
                      *[fmt(carry["carry_forward"][f"gcs_{name}"]["accuracy"])
                        for name in ["eye", "motor", "verbal"]]])
evidence.append({"model": "保持最近值", "source": msource,
                 "mean_standardized_numeric_mse": carry["carry_forward_mean_standardized_numeric_mse"],
                 "clinical": carry["carry_forward"]})
table(["模型", "数值标准化 MSE ↓", "心率 MAE（bpm）↓", "平均动脉压 MAE（mmHg）↓",
       "GCS-eye 准确率 ↑", "GCS-motor 准确率 ↑", "GCS-verbal 准确率 ↑"], clinical_rows)
put("""配对误差按患者汇总并给出置信区间。主模型以性别进入 INIT；baseline 保留各方法既有年龄条件或年龄协变量，各模型静态输入及训练方式随配置一并记录。

结果来源：[同预算 MIMIC](../mimic_experiment/results/baselines_matched_latent2x/summary.md)、[原单位与简单基线](../mimic_experiment/results/baselines_single_cuda/summary.md)、[确定性模型](../mimic_experiment/results/baselines_with_deterministic_cuda/summary.md)。""")
put("已填写的 HealthFlux 数值标准化 MSE 为 0.919463，保持最近值为 0.580933。后续 M0 与其他模型需要同时接受这两个参照的检验，结果分析结合生命体征、化验与 GCS 分项展开。")
put("""### 2.3 预测跨度与历史信息量

在固定最佳检查点上改变评价窗口，研究更长推演和更少历史观测的影响。ADNI 按实际间隔统计 0–12、12–24、24–36、36–60 个月。MIMIC 从 24 小时截止点计算未来 6/12/24 小时累计误差，再分别读取前 6/12 小时历史测试。每个窗口报告人数和有效目标数，各模型使用相同患者、时间点与掩码。

表 6：MIMIC 不同未来范围的累计误差。""")
forecast_rows = []
for label, m in [("M0", None), ("ODE-AE（概率，同预算）", mimic["ode_ae"]),
                 ("确定性 ODE-AE（同预算）", None), ("ODE-RNN（同预算）", mimic["ode_rnn"]),
                 ("Latent ODE（同预算）", mimic["latent_ode"]), ("HealthFlux-KL", mnative["healthflux_kl"])]:
    values = [m["result"]["test"][f"forecast_{h}h"]["reconstruction_error"] for h in [6, 12, 24]] if m and eligible(m) else [None]*3
    forecast_rows.append([label, *map(fmt, values)])
    if m and eligible(m):
        evidence.append({"model": label, "source": msource if label == "HealthFlux-KL" else mimic_source,
                         "horizon_errors": dict(zip(["6h", "12h", "24h"], values))})
table(["模型", "未来 6 小时 ↓", "未来 12 小时 ↓", "未来 24 小时 ↓"], forecast_rows)
put("表 7：M0 在不同信息量与跨度下的表现。")
table(["数据集", "窗口 / 上下文", "患者数", "有效目标数", "未来误差 ↓", "数值标准化 MSE ↓"],
      [["ADNI", x, "", "", "", ""] for x in ["0–12 个月", "12–24 个月", "24–36 个月", "36–60 个月"]]
      + [["MIMIC", f"{x} 小时历史 / 至 48 小时", "", "", "", ""] for x in [6, 12, 24]])
put("""主要图为误差随预测跨度和历史观测数的曲线。患者等权汇总，避免随访较多的患者主导比较。

方案来源：[MIMIC 时间窗口评价](../mimic_experiment/results/baselines_single_cuda/summary.md)、[ADNI 共同任务](../paper_odeae/results/baselines_full/summary.md)。""")
put("""### 2.4 CausalMed γ=1：反事实、治疗效应与策略选择

固定分叉患者状态后执行 Never、Always、Delayed、Aggressive、Conservative 五策略，封存预测与模型产生的治疗区间，再由独立评分器读取配对真值。报告全轨迹 CF-RMSE、1/2/3 年终点误差、Always 相对 Never 的 √PEHE、ATE 误差、策略遗憾和最优策略准确率。

公开数据只有静态 c，没有性别，因此合成适配版命名为 M0-c：INIT 接收训练集归一化的公开 c；其余使用共享 ODE、α=0.1、无 weak prior、Jacobian=0.01。门控只用于观测 UPDATE。治疗切换仍采用新旧动作条件输出之差，相同动作下位移为零。新目标实现必须显式删除原代码写死的 prior 系数。

表 8：γ=1 反事实比较，治疗效应与策略指标为 3 年窗口。""")
rows = [["M0-c（公开 c INIT）", "31,314", "", "", "", "", "", "待新训练"]]
for key, label in [("ODE-AE", "ODE-AE"), ("deterministic_22545（同预算）", "确定性 ODE-AE（同预算）"),
                   ("HealthFlux-KL", "HealthFlux-KL（原生容量）")]:
    m = cf[key]
    if eligible(m):
        a = m["with_actions"]; h = a["horizons"]["3.0"]; e = h["effects_vs_never"]["1"]
        scores = [a["cf_rmse_trajectory"], h["cf_rmse"], e["sqrt_pehe"], e["ate_error"], h["policy_regret"]]
        rows.append([label, f'{m["hardware"]["parameters"]:,}', *map(fmt, scores), "已有合格结果"])
        evidence.append({"model": label, "source": cf_source, "epochs_run": m["epochs"],
                         "best_epoch": m["best_epoch"], "scores": dict(zip(
                             ["cf_rmse", "cf_rmse_3y", "sqrt_pehe_3y", "ate_error_3y", "regret_3y"], scores))})
    else:
        rows.append([label, f'{m["hardware"]["parameters"]:,}', "", "", "", "", "", "待预算内评测"])
table(["模型", "参数量", "全轨迹 CF-RMSE ↓", "3 年 CF-RMSE ↓", "3 年 √PEHE ↓", "3 年 ATE 误差 ↓", "3 年策略遗憾 ↓", "状态"], rows)
put("""确定性模型匹配到 31,314 参数，HealthFlux 保留原生容量。四模型对相同固定病例绘制五策略真实/预测平行轨迹，补充每策略误差和 1/2 年指标。动作关闭测试复用相同检查点与分叉状态，只关闭未来动作校正，检验模型是否使用治疗信息。

表 9：M0-c 动作通路测试。""")
table(["设置", "全轨迹 CF-RMSE ↓", "3 年 √PEHE ↓", "3 年 ATE 误差 ↓", "3 年策略遗憾 ↓"],
      [[name, "", "", "", ""] for name in ["正常动作输入", "同检查点关闭未来动作校正"]])
put("""方案与结果来源：[四模型反事实评价](../CausalMed/outputs/gamma1_model_comparison/summary.md)、[动作适配](../CausalMed/latent2x_experiment/README.zh-CN.md)、[目标与动作代码](../CausalMed/latent2x_experiment/model.py)。""")

put("""## 三、核心消融实验

以下六个主题均在 ADNI、MIMIC 实施，固定性别 INIT、无 prior、28 维状态及相同训练预算。除研究变量外保持 M0 设置，实际总参数分别精确为 44,875 和 43,280。所有参照行复用 M0。""")
core("### 3.1 历史读取顺序",
     """M0 从最早历史检查开始，向未来逐条演化和吸收，最终状态放在历史截止点。反序组将性别 INIT 输出放在最近历史检查时刻，向过去逐条吸收，将最终状态放在最早历史检查，再从该时刻向未来生成整条轨迹。两组使用相同 INIT 网络和共享速度场参数化，积分方向由实际时间差正负决定。本实验比较读取顺序及对应时间锚点对预测、回顾与稳定性的影响。""",
     "反序共享 ODE",
     "[ADNI 编码方向](../paper_odeae/results/latent2x_history_encoder_ablation/summary.md)、[α 方向实验](../mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation/summary.md)、[方向实现](../mimic_experiment/alpha_direction_model.py)。",
     "表 10。交付分项损失、验证曲线、最佳轮次和长期患者配对误差；反序移除仅服务于 prior 的辅助读取。")
core("### 3.2 统一共享场与独立双场",
     """将 M0 的单一共享速度场替换为历史编码场与预测/回顾生成场，其余模块不变。共享场 28→188→28 共 10,744 参数；双场都用宽 94，编码场去掉末层 bias 后为 5,358，生成场为 5,386，合计仍为 10,744。实验检验相同总预算集中学习统一动力学，还是分给两个专用场更有效。""",
     "等预算独立双场",
     "[等参数双场设计](../ablation/Ablation.zh-CN.md)、[参数核验](../mimic_experiment/component_ablation_model.py)、[两场损失](../mimic_experiment/component_ablation_loss.py)。",
     "表 11。L2 对两场独立参数求和，Jacobian 在相同有效生成状态上对两场取平均。交付预测结果、状态衔接、分项训练曲线与求解成本。")
core("### 3.3 历史检查间的 ODE 演化",
     """只取消历史观测之间的 ODE 推进，下一次 UPDATE 前直接继承上一次 UPDATE 后状态。INIT、残差 UPDATE、生成 ODE、Decoder 和所有损失不变。共享 ODE 仍通过未来预测和历史回顾训练，实际任务参数量保持一致。按训练集预先固定的间隔不均匀度分组，检验连续时间编码对不规则随访的作用。""",
     "历史间隔无 ODE 推进",
     "[历史编码器](../paper_odeae/history_encoders.py)、[不规则性定义](SAMPLING_IRREGULARITY_COMPARISON.zh-CN.md)。",
     "表 12。交付误差随历史最大间隔、间隔变异系数、观测条数变化的曲线，以及主要测试指标。")
core("### 3.4 注意力与实际时间编码",
     """以因果注意力和实际时间位置编码替换整个 ODE+UPDATE 历史编码器，生成端保留原 ODE 与 Decoder。只读取截止点以内的有效历史，时间编码采用实际时间差。先核查现有实现的参数归属，再精确配置替换宽度，保持总预算。该组比较完整编码器家族；3.3 节则单独检验检查间 ODE 推进。""",
     "注意力＋实际时间编码",
     "[ADNI 编码器比较](../paper_odeae/results/latent2x_history_encoder_ablation/summary.md)、[MIMIC 编码器比较](../mimic_experiment/results/history_encoder_ablation_cuda/summary.md)、[实现](../paper_odeae/history_encoders.py)。",
     "表 13。注意力替换 UPDATE 后不再应用 α 门控。交付预测时长、短历史表现、训练曲线与实际成本。")
core("### 3.5 Jacobian 正则",
     """仅将 Jacobian 权重从 0.01 改为 0，保留所有主重建、ODE L2、性别 INIT 与门控设置。主要指标评价预测效果，第 4.6 节进一步对这两个检查点进行扰动实验，将局部规整性与实际输出稳定性联系起来。训练中保存加权及未加权 Jacobian 值、历史/未来分项损失和梯度。""",
     "Jacobian=0",
     "[损失消融设计](../paper_odeae/results/latent2x_loss_ablation/summary.md)、[精确 Jacobian](../paper_odeae/jacobian.py)、[效用方法](../ablation/results/jacobian_utility_adni/report.zh-CN.md)。",
     "表 14。分别报告平均误差、长期误差、患者配对差值区间；结果分析与第 4.6 节联合撰写。")
core("### 3.6 weak prior 增项对照",
     """以无 weak prior 的 M0 为参照，只加回 0.01 weak prior，检验目标简化的影响。该辅助项在每次观测吸收前评价预测，与从最终截止点生成的主重建项分开记录。INIT、α、参数、训练配置和全部测试设置相同，比较预测误差以及额外监督的计算成本。""",
     "加回 0.01 weak prior",
     "[模块损失设计](../ablation/Ablation.zh-CN.md)、[字段与归约实现](../mimic_experiment/component_ablation_loss.py)。",
     "表 15。此处列入后续计划；执行使用新的独立目录，保留此前停止任务和停止标记。交付完整结果与曲线。")

put("""## 四、状态空间、流形与动力学理论探究

以下各节均纳入本轮论文实验。使用 M0 和对应消融的最佳检查点，冻结模型，复用现有 Decoder 与诊断头，不增加可学习参数。数值表完成后插入实际图组，并围绕观察结果撰写分析。""")
put("""### 4.1 状态空间与速度场的平行可视化

在各模型训练集 UPDATE 后状态上分别拟合 PCA，固定用于该模型的测试状态、速度箭头和病例轨迹。ADNI 按诊断、MMSE、CDR-SB 着色；MIMIC 按 GCS、心率等预先固定指标着色。原始逐状态箭头与邻域平均速度场分别展示，测试状态与自由生成状态分开标记。计算状态平面保留的方差和速度能量，检查二维图对动力学的表达程度。

表 16：状态与速度的二维信息保留。""")
table(["数据集", "模型", "状态 PCA2 方差占比 ↑", "该平面速度能量占比 ↑", "速度 PCA2 能量占比 ↑", "状态有效秩"],
      [[d, m, "", "", "", ""] for d in ["ADNI", "MIMIC"] for m in
       ["M0", "确定性 ODE-AE（同预算）", "ODE-AE（概率，同预算）",
        "ODE-RNN（同预算）", "Latent ODE（同预算）", "HealthFlux-KL"]])
put("""主要图为状态分布、未平滑速度场、平均速度场、UPDATE 跳变和自由演化轨迹。ADNI 固定病例为 003_S_4900、002_S_4219、002_S_0619，并展示临床真实/预测曲线；MIMIC 从预先固定的患者名单展示 24→48 小时轨迹。不同模型分别使用自己的训练 PCA，各自内部保持坐标一致。编码状态与生成状态不同的模型分别提取两类表示，速度场图匹配实际生成状态及对应生成场；概率模型沿用共同测试的均值预测与条件定义。

方案来源：[共享场方法](../paper_odeae/results/shared_matched_state_space/report.zh-CN.md)、[模型横向可视化](../paper_odeae/results/state_space_model_comparison/report.zh-CN.md)、[ODE-AE 病例](../paper_odeae/results/odeae_baseline_state_space/report.zh-CN.md)。""")
put(r"""### 4.2 临床速度：潜在运动对应怎样的真实指标变化

用已有 Decoder 计算 \(\dot y=\nabla g(z)^\top f(z)\)。区间评价从前一次 UPDATE 后状态出发，自由积分到下一次真实检查，取解码差除以真实间隔，与真实变化率比较；预测区间内不吸收终点观测。ADNI 评价 MMSE、CDR-SB 年变化，MIMIC 评价心率和平均动脉压的小时变化。

表 17：真实随访区间上的临床变化率预测。""")
table(["数据集", "指标 / 单位", "区间数", "患者数", "变化率 MAE ↓", "Pearson r ↑", "Spearman ρ ↑", "方向准确率 ↑"],
      [[d, y, "", "", "", "", "", ""] for d, y in
       [("ADNI", "MMSE，分/年"), ("ADNI", "CDR-SB，分/年"),
        ("MIMIC", "心率，bpm/小时"), ("MIMIC", "平均动脉压，mmHg/小时")]])
put("""主要图并排展示同一 PCA 位置下的潜在速度、预测临床变化率与真实变化率，并绘制真实/预测散点。稳定、上升、下降的判定规则在训练/验证阶段固定，患者等权汇总后 bootstrap。

方案来源：[临床速度实验 A](../paper_odeae/results/shared_matched_clinical_velocity/report.zh-CN.md)。""")
put(r"""### 4.3 MMSE 下降速度的分解

将临床速度写为 \(\dot y=\|\nabla g(z)\|\,\|f(z)\|\cos\theta\)，按 CN、MCI、Dementia 分析潜在速度、Decoder 敏感性和临床方向投影。沿真实区间积分得到临床变化预算，并单独分析真实下降较快的区间。快速下降阈值根据训练分布或预先固定的临床规则确定。

表 18：M0 的 MMSE 动力学机制。""")
table(["组别", "患者数", "平均 ‖f‖", "平均 ‖∇g‖", "平均 cosθ", "真实年变化", "预测年变化", "变化率偏差"],
      [[x, "", "", "", "", "", "", ""] for x in ["CN", "MCI", "Dementia", "快速下降区间"]])
put("""图表包括三个分解因素的分布、速度与临床梯度夹角、快速下降与稳定患者的轨迹。根据实际分解结果判断临床变化偏差来自哪一部分。

方案来源：[MMSE 机制分解](../paper_odeae/results/mmse_decline_mechanism/report.zh-CN.md)。""")
put(r"""### 4.4 高维局部一致性与预测误差

在完整 28 维中为测试状态寻找其他训练患者的 k 近邻，主分析 k=20，补充 10/40。计算速度余弦 \(C_{ij}\)、局部变化比 \(D_{ij}=\|f_i-f_j\|/(\|z_i-z_j\|+\epsilon)\) 和 Jacobian 范数。核心与交界区域根据训练近邻诊断比例定义，并提前冻结。随后比较局部变化比与未来误差的关联，调整时间跨度、观测数和缺失程度。

表 19：ADNI 各区域的局部动力学。""")
table(["区域", "患者数", "方向一致性 C ↑", "局部变化比 D", "Jacobian 范数", "二维/高维邻域重叠率"],
      [[x, "", "", "", "", ""] for x in ["CN 核心", "CN/MCI 交界", "MCI 核心", "MCI/Dementia 交界", "Dementia 核心"]])
put("表 20：局部变化比与未来误差的关联。")
table(["数据集", "窗口", "患者数", "Spearman ρ", "调整后关联", "95% CI"],
      [[d, t, "", "", "", ""] for d, t in
       [("ADNI", "12 个月"), ("ADNI", "24 个月"), ("MIMIC", "12 小时"), ("MIMIC", "24 小时")]])
put("""主要图为高维方向一致性与敏感性在 PCA 位置上的热图、分层误差散点。近零速度与重复状态的处理写入配置，按患者汇总和重抽样。

方案来源：[局部动力学实验 B](../paper_odeae/results/shared_matched_local_dynamics/report.zh-CN.md)。""")
put("""### 4.5 低维流形与动力学维度

提取两个真实数据集的训练/测试状态，计算全局 PCA90/95/99% 维度、TwoNN、kNN 内在维度和局部 PCA。以同协方差高斯点云为比较，并按患者独立抽样复核。把速度分解为训练近邻切空间内外分量，检验状态点云的低方差方向是否仍承载重要临床演化。

表 21：M0 的几何维度与切空间外速度。""")
table(["数据集", "PCA95% 维度", "TwoNN 维度", "kNN 维度", "局部 PCA95% 维度", "12 维局部切平面外速度能量"],
      [[d, "", "", "", "", ""] for d in ["ADNI", "MIMIC"]])
put("""进一步比较原始自由推演、训练点云构建的冻结几何投影、向训练均值收缩。投影维数与收回强度只在验证集选择，再固定用于测试。Decoder 诱导的临床度量用于量化潜在位移对输出的影响。该分析不训练新参数。

表 22：冻结几何投影的预测检验。""")
table(["数据集", "设置", "未来误差 ↓", "相对原始推演差值", "主要临床 RMSE ↓"],
      [[d, setting, "", "", ""] for d in ["ADNI", "MIMIC"] for setting in
       ["原始自由积分", "验证集选定的冻结投影", "向均值收缩对照"]])
put("""主要图为维度随近邻数变化、真实点云与高斯对照、切空间内外速度、投影前后临床轨迹。结果分析区分“状态可以压缩”和“动力学可以限制在低维平面”。

方案来源：[流形设计](../paper_odeae/results/latent2x_both_manifold/report.zh-CN.md)、[几何与投影评价](../paper_odeae/results/latent2x_both_manifold/metrics_report.zh-CN.md)。""")
put(r"""### 4.6 Jacobian 的效用：误差传播、噪声和长期预测

使用 M0 与第 3.5 节无 Jacobian 模型，在相同患者、扰动方向和噪声下比较。先测局部 Jacobian，再测完整 ODE 流增益 \(\|D_z\Phi_T(z)\|_2\) 和临床输出扰动。潜在扰动为状态范数的 1%，固定 8 个方向；观测噪声为训练标准差的 1% 和 5%，只作用于输入历史中的可变数值字段，目标与评价掩码保持干净。

ADNI 取 0/1/2/3/5 年，MIMIC 取 0/6/12/24 小时。导数采用 float64 严容差，与更严容差或独立方法做收敛核验。按患者配对 bootstrap 给出差值区间，把正则对规整性的影响与实际误差改善分别量化。

表 23：规整性与实际鲁棒性。""")
table(["数据集", "模型", "平均 ‖Jf‖² ↓", "最长范围流增益中位数 ↓", "5% 噪声输出偏移 RMSE ↓", "干净目标临床 RMSE ↓", "长范围临床 RMSE ↓"],
      [[d, model, "", "", "", "", ""] for d in ["ADNI", "MIMIC"] for model in ["M0", "无 Jacobian"]])
put("""ADNI 主要临床指标为 MMSE、补充 CDR-SB；MIMIC 主要连续指标为心率、补充平均动脉压。联合预测另外报告数值标准化 MSE。图表包括流增益随时间变化、噪声—误差曲线、长跨度配对差值与区间，据此判断正则的实际作用。

方案来源：[正则效用设计及数值核验](../ablation/results/jacobian_utility_adni/report.zh-CN.md)、[实现脚本](../mimic_experiment/jacobian_utility_adni.py)。""")
put(r"""### 4.7 连续疾病阶段与 MCI 转化

利用已有诊断头定义 \(s=p_{\mathrm{MCI}}+2p_{\mathrm{Dementia}}\)，通过自动微分计算 \(\dot s=\nabla s^\top f\)。从固定已知历史截止点自由推演 0/6/12/24/36/60 个月。每个窗口内每位 MCI 患者只选一个预先确定的锚点：窗口内有 Dementia 记录为转化，稳定组须有足以支持该窗口的真实随访。当前评分、瞬时速度、未来评分增量与未来 Dementia 概率分别预测转化。

表 24：疾病阶段与真实转化区分能力。""")
table(["窗口", "稳定 / 转化患者数", "预测量", "AUROC ↑", "AUPRC ↑", "相对 s₀ 的 AUROC 差值"],
      [[f"{w} 个月", "", score, "", "", ""] for w in [12, 24, 36] for score in
       ["当前 s₀", "瞬时速度 ṡ₀", "阶段增量 Δs", "未来 Dementia 概率"]])
put("""主要图为相同 PCA 坐标上的阶段/速度热图、稳定与转化患者的平均阶段曲线、ROC 与 PR 对比。按患者计算区间，检验自由 ODE 推演是否在当前诊断概率之外增加区分信息。

方案来源：[连续阶段与转化评价](../paper_odeae/results/shared_matched_continuous_stage/report.zh-CN.md)。""")
put("""### 4.8 训练稳定性与递归增益

M0 训练中记录历史/未来损失对共享 ODE 的梯度范数及余弦，并在固定患者和固定跨度上跟踪向过去、未来积分的敏感性。预定轮次冻结参数，测量 UPDATE 状态 Jacobian 和截止点对历史状态的递归增益。与性别 INIT、无 prior、α=1 组比较，检验门控对 UPDATE 递归及历史损失梯度放大的影响。

表 25：MIMIC 稳定性。""")
table(["模型", "最佳轮次", "停止轮次", "历史/未来梯度比 P95", "梯度余弦中位数", "递归状态增益 P95", "非有限值次数"],
      [[m, "", "", "", "", "", ""] for m in ["M0（α=0.1）", "同配置 α=1"]])
put("""真实训练 batch 与固定诊断 batch 分别标记，诊断阶段不执行参数更新。图中共同展示分项损失、梯度比、递归增益和验证误差。α=1 在第 5.2 节完成后复用；先开展 M0 的训练记录与冻结分析。

方案来源：[梯度诊断](../mimic_experiment/results/direction_full_batch_cuda/gradient_sensitivity_analysis/report.zh-CN.md)、[敏感性与积分精度](../mimic_experiment/results/weak_prior_ablation_cuda/mechanism_diagnosis/report.zh-CN.md)、[针对性干预](../mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation/summary.md)。""")
put("""### 4.9 求解成本与预测质量

在相同 GPU 型号、共同测试患者和固定 batch 上测量编码/生成 NFE、接受与拒绝步数、每患者耗时和峰值显存。各模型权重固定，同样预热并同步 CUDA 后计时，记录求解器、容差、时间单位和统计范围。固定步长方法记录实际步数，自适应方法分别记录接受/拒绝步。

表 26：准确性与推演成本。""")
table(["数据集", "模型", "编码 NFE", "生成 NFE", "接受 / 拒绝步", "每患者 ms ↓", "峰值显存 GiB ↓", "未来误差 ↓"],
      [[d, model, "", "", "", "", "", ""] for d in ["ADNI", "MIMIC"] for model in
       ["M0", "确定性 ODE-AE（同预算）", "ODE-AE（概率，同预算）",
        "ODE-RNN（同预算）", "Latent ODE（同预算）", "HealthFlux-KL"]])
put("""主要图为误差—成本关系和真实病例的动态步长，展示观测更新前后时间步变化。

方案来源：[完整测试步数](../paper_odeae/results/solver_steps_full_test/summary.zh-CN.md)、[动态步长病例](../paper_odeae/results/step_examples/examples.zh-CN.md)。""")

put("""## 五、最后完成的补充消融

### 5.1 INIT 初始化方式

只替换 INIT 的标量输入，比较性别、首次检查年龄、历史截止点年龄、固定零输入。四组保留 1→159→28 网络，输出放在首次历史时刻；年龄按年/100 输入。零输入仍通过原 INIT 网络，它检验无个体静态信息的起点，并单列恒零输入使首层 159 个权重不参与任务这一事实。其他损失、α 和全部训练设置一致。

表 27：INIT 信息消融。""")
table(["数据集", "INIT 输入", "总参数", "任务使用参数", "历史 H ↓", "未来 F ↓", "整体 O ↓"],
      [[d, name, f"{p:,}", f"{p-159 if name == '固定零输入' else p:,}", "", "", ""]
       for d, p in [("ADNI", 44875), ("MIMIC", 43280)] for name in
       ["性别（M0）", "首次检查年龄", "截止点年龄", "固定零输入"]])
put("""按相同种子配对训练曲线与测试误差，把静态输入信息与网络容量分别说明。

方案来源：[INIT 实验与审计](../mimic_experiment/results/init_covariate_ablation_cuda/summary.md)、[ADNI 字段来源](../paper_odeae/data.py)。""")
put("""### 5.2 固定 α=0、0.1、0.2、0.3、0.4、0.5、1

统一性别 INIT、无 prior、Jacobian=0.01 和共享场，仅改变残差更新幅度。α=0 不吸收观测，只依赖 INIT 与 ODE；α=1 完全采用 UPDATE 候选状态。比较全部取值的历史回顾、未来预测与训练稳定性，绘制 α—误差、训练曲线和递归增益曲线。α=0.1 复用 M0，α=1 同时供第 4.8 节分析。

表 28：固定残差门控系数。""")
table(["α", "ADNI 参数", "ADNI H ↓", "ADNI F ↓", "ADNI O ↓", "MIMIC 参数", "MIMIC H ↓", "MIMIC F ↓", "MIMIC O ↓"],
      [[str(a) + ("（M0）" if a == 0.1 else ""), "44,875", "", "", "", "43,280", "", "", ""]
       for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]])
put("""α=0 保留名义结构，但 UPDATE 不参与任务；实际使用参数为 ADNI 27,993、MIMIC 27,193。所有组上限 150 轮，分别报告实际完成与最佳轮次。

方案来源：[门控扫描设计](../mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation/summary.md)。""")
put("""### 5.3 容量分配与潜在维度

固定最终模型的输入、损失和门控，比较潜在维度 6/28/73，以及分别增加 INIT、UPDATE、ODE 宽度的效果。先登记每个版本的精确参数和宽度，再训练。容量研究有意改变预算，分析额外参数投向哪一模块更有收益；28 维行复用 M0。

表 29：容量分配。""")
table(["数据集", "设置", "潜在维度", "实际参数", "未来 F ↓", "整体 O ↓", "每增加 1 万参数的未来误差改善"],
      [[d, setting, str(dim), f"{p:,}" if setting == "M0" else "", "", "", ""]
       for d, p in [("ADNI", 44875), ("MIMIC", 43280)] for setting, dim in
       [("较小状态", 6), ("M0", 28), ("较大状态", 73), ("仅 INIT 扩容", 28), ("仅 UPDATE 扩容", 28), ("仅 ODE 扩容", 28)]])
put("""主要图为误差—参数量—耗时关系，每组报告实际训练曲线。

方案来源：[潜在维度扩容](../paper_odeae/results/shared_matched_latent_scaling/summary.md)、[模块扩容](../paper_odeae/results/shared_matched_scaling_4x/summary.md)。""")

put("""## 六、后续 Agent 执行清单

### 6.1 执行顺序

本次仅形成计划与结果表；后续任务使用独立输出根目录 publication_sex_init/results/，已有训练、检查点与停止标记保持原状。P0 先补齐主结果，P1 完成核心消融和理论探究，P2 最后完成 INIT、完整 α 与容量补充。""")
table(["阶段", "任务 ID", "工作", "交付"], [
    ["P0", "SETUP", "性别入口、精确参数、损失和 CUDA 核验", "冻结配置与输入审计"],
    ["P0", "M-ADNI、M-MIMIC", "两个真实数据集主模型从种子初始化", "主结果与曲线"],
    ["P0", "B-ADNI、B-MIMIC", "同预算/原容量基线复用、预算核验与补缺", "横向比较表"],
    ["P0", "M-CF、B-CF", "γ=1 适配版与反事实基线", "封存预测、独立评分"],
    ["P1", "A-REV、A-SPLIT、A-ENCODE", "读取顺序、共享场、编码 ODE", "两数据集各 3 组"],
    ["P1", "A-ATTN、A-J0、A-PRIOR", "编码器家族、Jacobian、prior 增项", "两数据集各 3 组"],
    ["P1", "D-HORIZON、D-STATE、D-CLINICAL、D-MMSE、D-LOCAL", "时间窗口、空间与临床动力学", "图组、表格、定量解释"],
    ["P1", "D-GEOM、D-JAC、D-STAGE、D-STABILITY、D-COST、D-CF", "流形、效用、转化、稳定性、成本、反事实", "图组、表格、定量解释"],
    ["P2", "X-INIT", "三种额外 INIT 对照，性别行复用 M0", "表 27 与曲线"],
    ["P2", "X-ALPHA", "六种额外 α，0.1 行复用 M0", "表 28 与曲线"],
    ["P2", "X-CAPACITY", "潜在维度和模块容量补充", "表 29 与曲线"],
])
put("""D-JAC 依赖 A-J0；D-STABILITY 的 α=1 比较等待 X-ALPHA 对应组，M0 诊断可提前完成。主模型完成后即可开展其余冻结分析，GPU 0–3 分配独立作业。""")
put("### 6.2 代码衔接")
table(["位置", "本轮要求"], [
    ["INIT 入口", "F=0/M=1；ADNI 映射、真实输入缩放、首条历史来源可审计；网络与时间锚点固定"],
    ["component_ablation_study / checks", "参照为性别 INIT、prior=0、Jacobian=.01、α=.1；只更改指定变量"],
    ["component_ablation_loss", "保持原主重建归约与梯度；prior=0 时不依赖无效辅助输出"],
    ["alpha_direction_model", "记录反序初始时间与生成锚点，去掉仅服务于 prior 的辅助读取"],
    ["history_encoders", "注意力只看历史；核对参数；无编码 ODE 组保留残差 UPDATE"],
    ["CausalMed model", "显式去 prior；仅观测门控；动作差值不变；公开 c 初始化"],
    ["训练恢复", "完整保存 Adam、调度器、Python/NumPy/Torch/CUDA RNG、数据顺序与 batch 位置"],
])
put("""实现入口：[组件配置](../mimic_experiment/component_ablation_study.py)、[梯度与参数核验](../mimic_experiment/component_ablation_checks.py)、[损失](../mimic_experiment/component_ablation_loss.py)、[方向模型](../mimic_experiment/alpha_direction_model.py)、[历史编码](../paper_odeae/history_encoders.py)、[CausalMed](../CausalMed/latent2x_experiment/model.py)。

### 6.3 每节完成标准

每节交付实验介绍、填写后的结果表、真实曲线/图组、结果分析。分析按“实际数值→比较含义→支持的模型论点”展开，报告患者配对区间。动力学分析保存患者、截止点、时间跨度与积分精度；固定诊断不更新参数。

训练产物包含 best/last、完整恢复状态、SwanLab、配置、数据与源码哈希、检查点 SHA256、实际唯一参数量、物理 GPU 和张量设备。曲线包含训练/验证总损失、MSE、CE、历史/未来分项、Jacobian、L2、裁剪前后梯度范数与学习率。所有表中缺失指标继续留空，获得结果后再据实填写和分析。

机器可读入口：[任务、依赖和可复用数值来源](no_weak_prior_rerun_plan.json)。""")

task("SETUP", ["ADNI", "MIMIC", "CausalMed"], "P0", "implement_and_cuda_audit")
for d in ["ADNI", "MIMIC"]:
    task(f"M-{d}", [d], "P0", "train_from_fixed_seed", ["SETUP"], init_covariate="sex",
         resume_age_init_as_main=False, max_total_epochs=150)
    task(f"B-{d}", [d], "P0", "reuse_audit_or_train_baselines", ["SETUP"],
         preserve_baseline_objectives=True, prohibit_unaudited_overbudget_results=True)
task("M-CF", ["CausalMed"], "P0", "train_from_fixed_seed", ["SETUP"],
     init_covariate="public_train_normalized_c", gamma=1, preserve_action_shift=True,
     observation_update_alpha=0.1, max_total_epochs=150)
task("B-CF", ["CausalMed"], "P0", "reuse_audit_or_train_baselines", ["SETUP"])
changes = {
    "A-REV": {"history_order": "reverse", "initial_state_time": "last_history_visit",
              "generation_anchor": "first_history_visit"},
    "A-SPLIT": {"shared_vector_field": False, "encoder_ode_width": 94, "generator_ode_width": 94,
                "encoder_output_bias": False},
    "A-ENCODE": {"integrate_between_history_visits": False},
    "A-ATTN": {"history_encoder": "causal_attention_actual_time", "alpha_applicable": False},
    "A-J0": {"jacobian_weight": 0}, "A-PRIOR": {"prior_weight": 0.01},
}
for theme, change in changes.items():
    for d in ["ADNI", "MIMIC"]:
        task(f"{theme}-{d}", [d], "P1", "train_from_fixed_seed", ["SETUP"],
             changes=change, exact_parameter_match=True, init_covariate="sex",
             max_total_epochs=150, preserve_previous_stop_flags=True)
deps = {
    "D-HORIZON": ["M-ADNI", "M-MIMIC", "B-ADNI", "B-MIMIC"],
    "D-STATE": ["M-ADNI", "M-MIMIC", "B-ADNI", "B-MIMIC"],
    "D-CLINICAL": ["M-ADNI", "M-MIMIC"], "D-MMSE": ["M-ADNI", "D-CLINICAL"],
    "D-LOCAL": ["M-ADNI", "M-MIMIC"], "D-GEOM": ["M-ADNI", "M-MIMIC"],
    "D-JAC": ["M-ADNI", "M-MIMIC", "A-J0-ADNI", "A-J0-MIMIC"],
    "D-STAGE": ["M-ADNI"], "D-STABILITY": ["M-MIMIC", "X-ALPHA-MIMIC"],
    "D-COST": ["M-ADNI", "M-MIMIC", "B-ADNI", "B-MIMIC"], "D-CF": ["M-CF", "B-CF"],
}
for ident, ds in deps.items():
    datasets = ["CausalMed"] if ident == "D-CF" else ["ADNI"] if ident in ["D-MMSE", "D-STAGE"] else ["MIMIC"] if ident == "D-STABILITY" else ["ADNI", "MIMIC"]
    task(ident, datasets, "P1", "cuda_frozen_checkpoint_analysis", ds,
         parameter_updates_allowed=False, adds_learnable_parameters=False)
for d in ["ADNI", "MIMIC"]:
    task(f"X-INIT-{d}", [d], "P2", "train_control_groups", [f"M-{d}"],
         inputs=["sex", "first_age", "cutoff_age", "zero_scalar"], reuse_main_sex_row=True, max_total_epochs=150)
    task(f"X-ALPHA-{d}", [d], "P2", "train_control_groups", [f"M-{d}"],
         alpha_values=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1], reuse_main_alpha01_row=True,
         max_total_epochs=150)
    task(f"X-CAPACITY-{d}", [d], "P2", "train_capacity_controls", [f"M-{d}"],
         latent_dimensions=[6, 28, 73], modules=["init", "update", "ode"], max_total_epochs=150)
ids = {t["id"] for t in tasks}
assert len(ids) == len(tasks)
source_docs = {
    "SETUP": ["ablation/Ablation.zh-CN.md", "mimic_experiment/init_covariate_ablation.py"],
    "M": ["ablation/Ablation.zh-CN.md", "mimic_experiment/init_covariate_ablation.py"],
    "B": ["paper_odeae/results/baselines_matched_latent2x/summary.md",
          "mimic_experiment/results/baselines_matched_latent2x/summary.md"],
    "CF": ["CausalMed/outputs/gamma1_model_comparison/summary.md",
           "CausalMed/latent2x_experiment/README.zh-CN.md"],
    "A-REV": ["paper_odeae/results/latent2x_history_encoder_ablation/summary.md",
              "mimic_experiment/alpha_direction_model.py"],
    "A-SPLIT": ["ablation/Ablation.zh-CN.md", "mimic_experiment/component_ablation_model.py"],
    "A-ENCODE": ["paper_odeae/history_encoders.py", "reproduction/SAMPLING_IRREGULARITY_COMPARISON.zh-CN.md"],
    "A-ATTN": ["paper_odeae/results/latent2x_history_encoder_ablation/summary.md",
               "mimic_experiment/results/history_encoder_ablation_cuda/summary.md"],
    "A-J0": ["paper_odeae/results/latent2x_loss_ablation/summary.md", "paper_odeae/jacobian.py"],
    "A-PRIOR": ["ablation/Ablation.zh-CN.md", "mimic_experiment/component_ablation_loss.py"],
    "D-HORIZON": ["mimic_experiment/results/baselines_single_cuda/summary.md"],
    "D-STATE": ["paper_odeae/results/state_space_model_comparison/report.zh-CN.md"],
    "D-CLINICAL": ["paper_odeae/results/shared_matched_clinical_velocity/report.zh-CN.md"],
    "D-MMSE": ["paper_odeae/results/mmse_decline_mechanism/report.zh-CN.md"],
    "D-LOCAL": ["paper_odeae/results/shared_matched_local_dynamics/report.zh-CN.md"],
    "D-GEOM": ["paper_odeae/results/latent2x_both_manifold/report.zh-CN.md"],
    "D-JAC": ["ablation/results/jacobian_utility_adni/report.zh-CN.md"],
    "D-STAGE": ["paper_odeae/results/shared_matched_continuous_stage/report.zh-CN.md"],
    "D-STABILITY": ["mimic_experiment/results/weak_prior_ablation_cuda/mechanism_diagnosis/report.zh-CN.md"],
    "D-COST": ["paper_odeae/results/solver_steps_full_test/summary.zh-CN.md"],
    "D-CF": ["CausalMed/outputs/gamma1_model_comparison/summary.md"],
    "X-INIT": ["mimic_experiment/results/init_covariate_ablation_cuda/summary.md"],
    "X-ALPHA": ["mimic_experiment/results/weak_prior_ablation_cuda/targeted_causal_ablation/summary.md"],
    "X-CAPACITY": ["paper_odeae/results/shared_matched_latent_scaling/summary.md",
                   "paper_odeae/results/shared_matched_scaling_4x/summary.md"],
}
for t in tasks:
    assert set(t["depends_on"]) <= ids, t["id"]
    ident = t["id"]
    key = "CF" if ident in {"M-CF", "B-CF"} else (
        ident if ident in source_docs else ident.rsplit("-", 1)[0])
    t["source_documents"] = source_docs[key]
    t["experiment_document"] = str(DOC.relative_to(ROOT))
    for source in t["source_documents"]:
        assert (ROOT / source).exists(), source
plan = {
    "schema_version": 2, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "planning_only", "execution_started": False,
    "document": str(DOC.relative_to(ROOT)),
    "main_model_definition": {
        "working_name": "M0_sex_init_no_prior_shared_alpha01_jac01",
        "latent": 28, "init_covariate": "sex", "init_encoding": {"female": 0, "male": 1},
        "init_network": [1, 159, 28], "initial_state_time": "first_known_history_visit",
        "history_order": "chronological", "shared_vector_field": True,
        "ode_network": [28, 188, 28], "update_alpha": 0.1,
        "prior_weight": 0, "jacobian_weight": 0.01, "l2_weight": 0.02,
        "jacobian_definition": "mean_valid_generated_states(sum_784_squared_entries); states detached",
        "main_loss": "original masked all-time field-normalized MSE plus CE",
        "parameters": {"ADNI": 44875, "MIMIC": 43280, "CausalMed_c_adaptation": 31314},
        "completed_matching_main_runs": [],
        "causalmed_adaptation": {"name": "M0-c", "init_covariate": "public_train_normalized_c",
                                "public_sex_available": False, "gamma": 1, "action_shift_unchanged": True},
    },
    "execution_policy": {
        "environment": "ode; native HealthFlux retains healthflux", "physical_gpus": [0, 1, 2, 3],
        "model_execution": "CUDA only", "workers_per_gpu": 1, "max_total_epochs": 150,
        "seed": 0, "gradient_accumulation_steps": 1,
        "batch_size": {"ADNI": 32, "MIMIC": 128, "CausalMed": 128},
        "planned_output_root": "publication_sex_init/results/",
        "preserve_existing_checkpoints_and_logs": True, "preserve_stop_flags": True,
        "launch_existing_queues": False,
    },
    "result_policy": {
        "obsolete_results_in_document": False, "missing_numeric_cell": "",
        "age_init_can_be_main_or_resume_as_main": False,
        "overbudget_results_require_separate_checkpoint_audit": True,
        "all_theoretical_topics_required": True, "init_and_full_alpha_priority": "P2",
    },
    "task_count": len(tasks), "tasks": tasks, "compatible_result_evidence": evidence,
}
DOC.write_text("\n".join(parts))
PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"document": str(DOC.relative_to(ROOT)), "lines": len(DOC.read_text().splitlines()),
                  "tasks": len(tasks), "compatible_sources": len(evidence)}, ensure_ascii=False))
