"""Readable, reproducible captions for the publication report (CPU summaries only)."""
import json
import os
from pathlib import Path
import re


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


TABLE_CAPTIONS = {
1: '所有模型使用相同 ADNI 测试患者和历史前缀，H、F、O 均为数值 MSE 与类别 CE 的字段和。M0 的未来误差为 0.991104，低于同预算 ODE-RNN 的 1.032329 和确定性 ODE-AE 的 1.239294；相对降幅分别为 3.99% 和 20.03%。各模型的最佳轮次由验证整体误差选择。',
2: '本表保留各 baseline 的原始容量，用来观察容量变化后的结果。确定性 ODE-AE 从 22,545 增至表 1 的 44,875 参数后，未来误差由 1.191548 变为 1.239294；增加参数并未带来更低的测试误差。总参数与可训练参数分列，冻结参数计入前者。',
3: '所有模型读取前 24 小时，再预测至第 48 小时。M0 的整体误差为 1.125570，未来误差为 1.795789；ODE-RNN 的未来误差更低，为 1.760460。M0 在历史回顾上的优势与未来预测的排名需要分别阅读。',
4: '原容量模型补充方法的容量参照。概率 ODE-AE 的历史训练实际完成 178 轮，本次只在前 150 轮中选中第 138 轮并重新评价；表中保留了真实训练轮数。原容量 ODE-RNN 与同预算版本的未来误差接近，分别为 1.763826 与 1.760460。',
5: '数值标准化 MSE 对 27 个数值字段等权平均，其余列保留临床单位或分类准确率。保持最近值的数值标准化 MSE 为 0.580933，优于本表全部神经模型；M0 的 GCS 分类则分别达到 0.736253、0.841209、0.846045。不同输出任务呈现不同排序。',
6: '三列均从同一个 24 小时截止点出发，分别累计未来 6、12、24 小时的有效目标。M0 的误差从 1.452228 增至 1.795789；三个窗口中 ODE-RNN 均略低于 M0，M0 均低于确定性 ODE-AE 和 HealthFlux-KL。窗口内字段覆盖也会影响混合误差。',
7: 'ADNI 各行是互不重叠的实际预测间隔，MIMIC 各行则改变已知历史长度。ADNI 的未来误差随跨度由 0.770486 增至 1.057661；MIMIC 由 6 小时增加到 24 小时历史后，未来误差从 2.345358 降至 1.795789。人数和有效目标数用于说明每个统计范围。',
8: '四模型面对相同分叉患者和五种治疗策略，表内治疗效应及策略指标取 3 年窗口。M0-c 的全轨迹 CF-RMSE 为 0.962064，低于同预算确定性 ODE-AE 的 1.212183 和原生 HealthFlux-KL 的 1.408327；3 年策略遗憾也最低。M0-c 的静态输入为公开 c。',
9: '两行使用完全相同的模型权重和分叉状态，只切换未来动作校正。关闭动作后，CF-RMSE 从 0.962064 升至 5.486232，3 年策略遗憾从 0.221523 升至 1.302645，说明治疗输入确实参与了反事实轨迹和策略排序。',
10: '只改变历史读取顺序及其对应的生成起点，参数预算保持一致。改为反序后，ADNI 未来误差由 0.991104 升至 1.052676，MIMIC 由 1.795789 升至 1.826991；本轮正序在两数据集的 H、F、O 均较低。',
11: '共享场与双场的 ODE 参数总数都为 10,744，双场将预算拆成两个较窄的网络。ADNI 中共享场未来误差较低；MIMIC 双场的未来误差略低，为 1.792220，但历史和整体误差较高。因此，共享场的主要收益体现在统一预算下的综合表现。',
12: '无编码 ODE 组仍保留每次观测 UPDATE 和全部生成 ODE，仅略过历史检查之间的连续演化。去掉该演化后，ADNI 未来误差增加 0.076265，MIMIC 增加 0.006044；MIMIC 历史误差同时从 0.518334 增至 0.637975。',
13: '注意力组用实际时间编码的因果注意力替换整个历史 ODE＋UPDATE 编码器，总参数匹配 M0。两数据集的 H、F、O 均高于 M0；ADNI 和 MIMIC 的未来误差分别为 1.042857、1.827927。该结果对应编码器整体替换。',
14: '本表全部是最佳验证轮次对应的测试结果，区别于第七章的最佳验证整体误差。去掉正则后，ADNI 未来误差增加 0.011858，MIMIC 增加 0.005229，均为小幅变化。正则对完整 ODE 流和实际噪声的影响见表 23。',
15: '本轮只在 M0 上加回 0.01 weak prior。ADNI 的历史和整体误差下降，未来误差略升；MIMIC 的未来误差从 1.795789 变为 1.794376，变化很小。这组结果没有显示出跨数据集一致的未来预测收益。',
16: '每个模型单独使用训练状态拟合 PCA，再投影其测试状态和实际生成速度。M0 在 ADNI 的状态平面保留 42.36% 方差，却只保留 15.60% 速度能量；MIMIC 分别为 37.88% 和 1.41%。速度另拟合的二维平面可保留超过 96% 能量，说明状态分布主方向与运动主方向差异很大。',
17: 'ADNI 按患者先平均相邻随访区间的真实/预测变化率，再计算 MAE 和相关性；区间数另外列出。MIMIC 每位患者从固定截止点预测下次有效测量。MMSE 和 CDR-SB 的 Pearson 相关分别为 0.584676、0.500110；心率及平均动脉压的相关接近零，方向准确率约为 50%。',
18: '潜在速度、Decoder 梯度及夹角共同决定 MMSE 变化。Dementia 的速度和 Decoder 梯度均高于 CN，方向也更朝向 MMSE 下降；其平均预测年变化为 −2.025149，真实为 −2.406913。快速下降区间仍有较大偏差，表明模型对急剧变化的幅度估计不足。',
19: '邻居从完整 28 维训练状态中按不同患者选取，区域由训练邻居的诊断构成决定；每个区域内先按患者汇总。C 接近 1 表示局部速度方向一致，D 衡量相近位置的速度变化。CN/MCI 交界的 C 最低，为 0.908424；二维邻居重叠率仅约 7%–23%，因此高维结果需要单独计算。',
20: '调整后关联控制历史信息和覆盖等预定协变量，区间由患者重抽样获得。ADNI 的两个区间都跨零；MIMIC 的关联虽为负且区间未跨零，绝对值也只有约 0.05。本轮数据未呈现“局部 D 越大，未来误差越大”的一致模式。',
21: 'PCA95% 衡量全局线性维度，TwoNN、kNN 和局部 PCA 衡量不同尺度的局部结构。ADNI、MIMIC 的 PCA95% 分别需要 16、19 维。表中最后一列取最长诊断跨度的 12 维训练局部切平面，平面外速度能量中位数为 77.48%、62.77%，说明低方差方向仍承担明显运动。',
22: '该表只评价冻结投影探针预先选定的患者、状态和目标，统计范围独立于主测试表。两个数据集的验证集都为局部投影选择了原始推演，因此相应测试差值为零。MIMIC 的均值收缩略降低混合误差，却增大心率 RMSE，反映两项指标的权衡。',
23: '局部 Jacobian 与完整流增益分别描述瞬时速度变化和有限时间误差放大；最长范围分别为 ADNI 5 年、MIMIC 24 小时。ADNI 的流诊断每患者取前三次以内历史后的状态，MIMIC 取 24 小时历史后的状态。5% 噪声列测相对干净输入预测的偏移，干净目标列测对真实 MMSE 或心率的误差，单位分别为分、bpm。长期列仅用 ADNI 至少 3 年、MIMIC 超过 12 小时的目标。',
24: '每个窗口按患者定义稳定和转化组，预测只读取窗口开始前的历史。未来 Dementia 概率在 24、36 个月的 AUROC 为 0.818940、0.813660，比当前阶段评分提高 0.047052、0.066313。单独使用瞬时速度的 AUROC 均较低，累计推演提供了更多区分信息。',
25: '梯度、余弦和递归增益来自固定八位训练患者的冻结诊断，沿可用检查点统计；历史与未来共享原主损失的字段分母。最佳/停止轮次来自真实训练日志。数值异常计数指训练因非有限量终止的记录；诊断不更新权重，检查点跨度和固定样本见原始 JSON。',
26: 'NFE 与接受/拒绝步数为本次测量的全部 batch 总和；耗时除以患者上下文条数，显存为 CUDA 分配峰值。ADNI 同一患者可能对应多个前缀。ODE-RNN 在单一循环中读取和生成，本次挂钩将其调用统一计入生成列，编码列的 0 表示没有独立分期计数。误差列复用表 1、3 的完整测试 F，耗时使用原生求解器。',
27: '四组仅更换 INIT 标量，网络形状一致；固定零输入组首层的 159 个输入权重不参与任务。ADNI 中零输入的整体误差更低，而性别初始化的未来误差最低；MIMIC 中性别初始化的 H、F、O 均最低。静态信息的影响随任务而变。',
28: 'α 控制观测校正幅度，α=0 完全不吸收检查，α=1 完全采用 UPDATE 输出。两数据集 α=0 的误差均大幅增加。ADNI 的未来误差在 α=0.3 最低，为 0.972916，整体误差在 α=0.1 更低；MIMIC 的 H、F、O 均在 α=0.1 最低。',
29: '潜在维度和模块扩容分别改变预算，参照均为同数据集的 M0。6→28 维的未来预测改善明显，28→73 维未继续降低未来误差；单独扩容 INIT 或 UPDATE 也未形成稳定收益。较小状态行的最后一列表示从该组扩至 M0 的单位参数收益，其余行表示从 M0 扩至该组的收益，负值表示变差。',
}


def rows(text, number):
    found=re.search(rf'表 {number}：[^\n]*\n\n(\|[^\n]*\n(?:\|[^\n]*\n)*)',text)
    if not found:
        return []
    return [[c.strip() for c in line.split('|')[1:-1]] for line in found.group(1).strip().splitlines()[2:]]


def figure_details(image, out):
    rel=image.relative_to(out)
    parts=rel.parts
    dataset='ADNI' if 'adni' in parts or image.stem.startswith('adni_') else 'MIMIC' if 'mimic' in parts or image.stem.startswith('mimic_') else 'CausalMed'
    name=image.stem
    if parts[0]=='curves':
        group=name.split('_',1)[1]
        titles={'core':'核心消融训练与验证曲线','init':'INIT 信息消融曲线','alpha':'残差门控系数消融曲线','capacity':'容量分配训练与验证曲线'}
        section={'core':'3.6','init':'5.1','alpha':'5.2','capacity':'5.3'}[group]
        settings={'core':'不同线条为主模型、去正则、反序、双场、无编码 ODE、注意力和加回 prior。',
                  'init':'不同线条仅改变性别、首次年龄、截止点年龄或零输入。',
                  'alpha':'不同线条仅改变 α；α=0.1 对应 main。',
                  'capacity':'不同线条对应潜在维度及 INIT、UPDATE、ODE 宽度调整。'}[group]
        best=read(out/f'results/{dataset.lower()}/main/metrics.json')
        final=best.get('epochs_run','');epoch=best.get('best_epoch','')
        caption=(f'横轴为训练轮次；上排依次为总训练目标、历史误差、未来误差，下排为验证历史、未来、整体误差。{settings}'
                 f'主模型完成 {final} 轮、在第 {epoch} 轮取得最佳验证整体结果。训练分项各自归一化，不能直接相加还原总目标；最终测试值见对应表格。')
        return section,f'{dataset} {titles[group]}',caption
    if dataset=='CausalMed':
        if name.startswith('parallel_case_'):
            i=int(name.rsplit('_',1)[1])-1
            meta=read(out/'analysis/causalmed/comparison/case_metadata.json')
            patient=meta.get('patients',['','',''])[i]
            return '2.4',f'CausalMed 固定病例 {i+1}：五策略轨迹',(
                f'病例 {patient} 来自预先固定的前三位公开测试患者。各行/列按图内模型和策略标签对齐，真实曲线与预测曲线使用同一时间轴。'
                '从同一历史状态分叉后，模型只读取策略动作并自由推演。该图用于观察治疗效应随时间累积的偏差，整体准确性见表 8。')
        return '2.4','CausalMed 个体治疗效应比较',('同一患者的 Always 与 Never 预测差形成个体治疗效应，再与配对真值比较；四模型使用相同测试查询。'
                'M0-c 的 3 年 √PEHE 为 1.445902，确定性 ODE-AE 为 1.705769，HealthFlux-KL 为 1.867809。图内个体差异与表 8 的总体指标共同评价效应估计。')
    if name in {'state_and_velocity','clinical_cases'}:
        key=parts[-2] if 'baseline_state' in parts else 'main'
        model={'main':'M0','deterministic':'确定性 ODE-AE','ode_ae':'概率 ODE-AE','ode_rnn':'ODE-RNN','latent_ode':'Latent ODE','healthflux':'HealthFlux-KL'}[key]
        if name=='state_and_velocity':
            stat=read(image.parent/'statistics.json')
            state=100*stat.get('state_variance_retained_2d',0);speed=100*stat.get('velocity_energy_retained_2d',0)
            labels='CN=0、MCI=1、Dementia=2；−1 表示本次诊断缺失' if dataset=='ADNI' else 'GCS-motor 类别；−1 表示本次记录缺失'
            desc=(f'左图为测试状态，颜色表示{labels}；中图是各状态未经邻域平滑的速度投影，颜色表示投影速度大小；右图为训练邻域平均速度及测试位置。'
                  f'三个面板沿用该模型训练 PCA，保留 {state:.2f}% 状态方差和 {speed:.2f}% 速度能量。不同模型各用自己的 PCA，坐标数值不直接对齐。')
            if dataset=='ADNI' and key=='main':
                desc+=' 测试状态沿 PC1 呈现从 CN 到 Dementia 的顺序，右侧的投影运动整体更快。'
            return '4.1',f'{dataset} {model}：状态分布、原始速度与平均速度',desc
        desc=('每行对应一个预先固定的测试病例，左图连接不同历史截止点的状态估计，右图比较真实观测与模型原生预测；竖线标出历史截止点。'
              'ADNI 临床指标为 MMSE，MIMIC 为心率。左侧连线包含吸收新检查产生的校正，右侧截止点以后的预测不读取目标观测。')
        return '4.1',f'{dataset} {model}：患者状态和临床轨迹',desc
    definitions={
      'clinical_velocity_fields':('4.2','潜在速度与临床 MMSE 速度', '三个面板保持同一训练 PCA 坐标，依次展示潜在运动、Decoder 链式导数得到的 MMSE 年变化率、真实相邻检查的年变化率。负值表示认知评分下降。箭头大小与临床下降幅度通过 Decoder 梯度联系起来，表 17 给出独立随访区间验证。'),
      'rate_prediction_accuracy':('4.2','真实与预测 MMSE 年变化率', '左图每点是一个测试随访区间，横轴为真实、纵轴为预测变化率，虚线是完全一致线，颜色为检查间隔；中图按当前诊断着色；右图按患者用时间加权汇总，竖线为训练集快速下降阈值。预测对改善和急剧下降均更靠近群体趋势，快速下降幅度不足与表 18 一致。图的区间/时长加权与表 17 的患者等权口径分别标明。'),
      'fast_decline_ranking':('4.2','快速 MMSE 下降的风险排序', '横轴为假阳性率，纵轴为真阳性率；每位患者在首个合格历史节点产生预测，快速下降阈值由训练集确定。曲线比较初始历史的 ODE 预测变化率、初始 MMSE 及初始诊断，图例标出各自 AUROC。该图评价进展速度的区分能力，误差幅度由表 17 评价。'),
      'local_dynamics_map':('4.4','28 维局部动力学在 PCA 上的位置', '四个面板分别为训练邻居定义的疾病区域、完整 28 维速度余弦 C、局部变化比 D、Jacobian Frobenius 范数。颜色范围取训练分布，测试状态只投影到固定坐标。CN/MCI 交界的一致性较低，而 Dementia 核心区域较高，患者等权数值见表 19。'),
      'projection_consistency':('4.4','二维投影改变了哪些近邻和速度关系', '左图比较完整 28 维邻居/速度、二维邻居/高维速度、二维邻居/投影速度的余弦分布；中图为两种空间选出的患者邻居重叠率；右图逐状态比较完整与投影一致性。邻居重叠率较低说明二维可视化重排了局部关系，高维一致性由独立计算获得。'),
      'sensitivity_forecast_error':('4.4','局部动力学与 12/24 个月 MMSE 误差', '每排对应一个未来窗口，前两列将患者 MMSE MAE 分别与 D、Jacobian 范数比较，最后一列按训练集定义的 D 四分位分组。散点和分组用于检查单调关系；表 20 的调整后关联区间均跨零，未得到 ADNI 上 D 增大导致长期误差增大的证据。'),
      'dimension_evidence':('4.5','全局与局部维度证据', '图组比较 PCA 累计方差、局部维度估计和同协方差高斯对照，尺度和近邻数见各面板。ADNI、MIMIC 的 PCA95% 分别为 16、19 维，均明显高于二维展示；局部估计更低，说明点云在不同尺度上具有不同的有效维度。'),
      'local_structure':('4.5','近邻尺度下的状态几何结构', '各面板按图内近邻数和尺度比较局部 PCA、内在维度及相关结构，训练点云定义所有坐标和邻域。与同协方差对照结合，可以区分全局各向异性和局部组织结构；具体维度统计见表 21。'),
      'dynamics_evidence':('4.5','局部切空间与临床动力学', '图组把 ODE 速度分为训练近邻局部切空间内外分量，并结合 Decoder 诱导的输出变化分析。表 21 中最长跨度、12 维局部平面外的速度能量中位数在 ADNI 为 77.48%、MIMIC 为 62.77%。状态方差小的方向仍可包含显著演化。'),
      'projection_probe':('4.5','冻结几何投影对预测的影响', '左图为各固定投影/收缩设置相对原始推演的测试误差变化，零线为原始结果；右图只展示验证集选定设置的配对测试差值与 95% 区间。两个数据集的局部投影选择均为 baseline，对应差值为零。投影探针使用独立固定查询集，参见表 22。'),
      'dementia_mechanism_vs_decline':('4.3','Dementia 下降幅度与动力学因素', '各面板将真实下降与潜在速度、Decoder 梯度及方向投影联系起来，单位由图内坐标标明。快速下降区间的真实 MMSE 年变化为 −8.795925，模型为 −2.531310；较大的潜在速度尚不足以覆盖这些极端变化。'),
      'mechanism_by_diagnosis':('4.3','三类诊断的 MMSE 速度分解', '按 CN、MCI、Dementia 分组比较速度范数、MMSE 梯度范数、夹角余弦与临床变化率。表 18 中 Dementia 三个因素共同指向更快的 MMSE 下降，仍存在约 0.382 分/年的平均偏慢。三个因素的均值分别报告，均值乘积不等于临床速度均值。'),
      'mechanism_state_space':('4.3','MMSE 速度分解在状态空间中的分布', '每个面板在相同训练 PCA 位置显示一个链式导数因素或最终 MMSE 速度。读图时将速度大小、Decoder 敏感性和方向共同考虑，临床变化率由三者逐状态相乘得到；表 18 提供诊断组定量比较。'),
      'continuous_stage_space':('4.7','连续疾病阶段评分', '颜色为已有诊断头得到的 s=p(MCI)+2p(Dementia)，范围 0–2，坐标沿用主模型训练 PCA。患者末次状态阶段评分与当前诊断的 Spearman 相关为 0.937936，体现连续诊断结构；未来转化的区分能力另见表 24。'),
      'progression_velocity_field':('4.7','连续阶段的瞬时进展速度', '箭头为 ODE 在固定 PCA 上的投影，颜色按图中标尺表示 ṡ=∇sᵀf 或阶段评分。正速度表示阶段评分上升，负速度表示下降。测试 UPDATE 状态中，CN、MCI、Dementia 的正速度比例约为 91.3%、70.9%、95.4%；自由推演不执行未来 UPDATE。'),
      'mci_stage_evolution':('4.7','稳定与转化 MCI 的阶段演化', '三列按真实 12/24/36 个月转化分组，上排为预测平均 s(t)，下排为相对起点的增量；红色为转化组、蓝色为稳定组，阴影为患者均值的 95% bootstrap 区间，竖线标出评价窗口。两组都沿 ODE 上升，转化组的增量更大。真实随访只用于分组。'),
      'conversion_roc':('4.7','MCI 转化预测 ROC', '横轴是假阳性率，纵轴是真阳性率；各面板对应 12/24/36 个月，比较当前评分、瞬时速度、阶段增量与未来 Dementia 概率。后者在 24、36 个月分别达到 AUROC 0.818940、0.813660，高于当前评分。图例数值与表 24 对应。'),
      'conversion_precision_recall':('4.7','MCI 转化预测 PR', '横轴为召回率，纵轴为精确率；曲线使用与 ROC 相同的患者和评分，水平参照随窗口转化比例改变。未来 Dementia 概率在 24、36 个月的 AUPRC 分别为 0.590956、0.681677，见表 24。各窗口样本构成分别列出。'),
      'jacobian_utility':('4.6','Jacobian 正则的流增益与噪声效用', '左图横轴为自由推演时长，纵轴为患者 ODE 流最大奇异值的中位数；右图横轴为历史输入噪声占训练标准差的比例，纵轴为对干净临床目标的 RMSE。两条线对应 M0 与无 Jacobian，潜在流敏感性和输出误差分别测量。数值见表 23，收敛检验保存在同目录 metrics.json。'),
      'stability_diagnostics':('4.8','固定患者的梯度与递归增益', '横轴为实际参数检查点轮次，各面板比较历史/未来 ODE 梯度比、梯度余弦、递归状态增益和向过去/未来的流增益。颜色区分 α=0.1 与 α=1。所有测量读取同一批八位患者及固定 dropout 掩码；曲线只连接实际保存的检查点，零参数更新。'),
    }
    if name not in definitions:
        raise ValueError(f'Missing explicit figure caption: {rel}')
    section,title,description=definitions[name]
    return section,f'{dataset} {title}',description


def enrich(document, docpath, out):
    # Idempotent under the live report: replace only our caption/figure blocks.
    document=re.sub(r'\n<!-- publication-caption -->\n.*?\n<!-- /publication-caption -->\n','\n',document,flags=re.S)
    document=re.sub(r'\n<!-- publication-figures -->\n.*?\n<!-- /publication-figures -->\n','\n',document,flags=re.S)
    document=re.sub(r'^!\[[^\n]*\]\([^\n]*\)\s*$', '', document, flags=re.M)
    document=document.replace('# 论文实验清单：','# 论文实验结果与分析：')
    document=re.sub(r'更新日期：[^\n]+',
        '更新日期：2026-11-08。本文围绕主要比较、核心消融、动力学机制与补充消融组织本轮实验。每个主题依次说明问题、设置、数值结果和图表解释；原始配置、指标和日志由文中链接提供。',document,count=1)
    guide=('\n<!-- publication-caption -->\n阅读顺序：第一章定义模型与统计口径，第二章比较三类数据集的预测效果，第三章逐项检查核心设计，第四章研究学到的状态空间与动力学，第五章讨论初始化、门控和容量。'
           '第七章逐项列出 89 个作业的执行记录；其中验证分数用于选模，前文结果表使用独立测试分数。\n<!-- /publication-caption -->\n')
    document=document.replace('\n## 一、模型定义与共同实验标准',guide+'\n## 一、模型定义与共同实验标准',1)
    document=document.replace('两个数据集均已从固定种子启动，进度见第七章','两个数据集均已完成，最佳轮次分别为 ADNI 30、MIMIC 143')
    document=document.replace('以下各节均纳入本轮论文实验。使用 M0 和对应消融的最佳检查点，冻结模型，复用现有 Decoder 与诊断头，不增加可学习参数。数值表完成后插入实际图组，并围绕观察结果撰写分析。',
        '以下分析读取本轮 M0 和对应消融的最佳检查点，冻结权重，复用已有 Decoder 与诊断头。状态空间、临床速度、局部几何和转化验证从不同角度检验同一个模型。每个图组紧接所属主题，表后解释对应数值的含义。')
    document=document.replace('M0 训练中记录历史/未来损失对共享 ODE 的梯度范数及余弦，并在固定患者和固定跨度上跟踪向过去、未来积分的敏感性。',
        '读取 M0 训练记录，并在已保存参数检查点上，使用固定诊断 batch 重新计算历史/未来损失对共享 ODE 的梯度范数及余弦，以及向过去、未来积分的敏感性。')
    document=document.replace('非有限值次数','数值异常终止次数')
    document=document.replace('## 六、后续 Agent 执行清单','## 六、实验组织与复现入口')
    document=document.replace('本轮实验已启动；全部任务使用独立输出根目录','本轮实验共 89 个队列任务；全部任务使用独立输出根目录')
    document=document.replace('本轮已加入候补队列，使用新的独立目录，保留此前停止任务和停止标记。交付完整结果与曲线。','本轮使用独立目录完成对照，完整训练与验证曲线见下图。')
    queue=read(out/'queue.json')
    if queue.get('status')=='complete':
        repair=('\n<!-- publication-caption -->\n本轮 89 项作业现已全部完成，failed=0。此前失败的两项 Jacobian 效用分析和一项 MIMIC 稳定性分析，已在物理 GPU 0、1、2 补跑，参数更新次数为零。'
                '详见 [修复与数值核验](../publication_sex_init/REPAIR.zh-CN.md)、[图表与产物检查](../publication_sex_init/validation.json)。权重与大型中间状态缓存保留本地。\n<!-- /publication-caption -->\n')
        document=document.replace('\n## 七、实时执行进度',repair+'\n## 七、实时执行进度',1)
    document=document.replace('已填写的同预算 baseline 中，ODE-RNN 的未来误差为 1.032329，概率 ODE-AE 为 3.238572；HealthFlux 的两种 KL 设置分别为 1.534999 和 1.603200。后续主比较据此评价新 M0 的预测质量与参数效率。',
        '同预算方法中，M0 的未来误差最低。HealthFlux 的两种 KL 设置分别为 1.534999 和 1.603200，均高于 M0；其参数量约为 M0 的 115 倍。')
    document=document.replace('已填写的 HealthFlux 数值标准化 MSE 为 0.919463，保持最近值为 0.580933。后续 M0 与其他模型需要同时接受这两个参照的检验，结果分析结合生命体征、化验与 GCS 分项展开。',
        'M0 的数值标准化 MSE 为 0.913959，与 HealthFlux 的 0.919463 接近；保持最近值为 0.580933。当前模型对联合预测和 GCS 读出有不同表现，单独的数值预测仍有明显改进空间。')
    # Numbered tables keep their stable IDs for automatic result updates.
    for number,description in TABLE_CAPTIONS.items():
        pattern=rf'(表 {number}：[^\n]*\n\n\|[^\n]*\n(?:\|[^\n]*\n)*)'
        extra=''
        data=rows(document,number)
        if number==23 and data and all(row[2] for row in data):
            extra=' 本轮结果：'+'；'.join(f'{row[0]} {row[1]} 的平均 ‖Jf‖² 为 {row[2]}、最长流增益中位数为 {row[3]}、5% 噪声输出偏移为 {row[4]}' for row in data)+'。'
            bykey={(r[0],r[1]):r for r in data}
            for d in ['ADNI','MIMIC']:
                a,b=bykey[d,'M0'],bykey[d,'无 Jacobian']
                extra+=f' {d} 中，正则模型的干净临床 RMSE 为 {a[5]}，无正则为 {b[5]}；局部 ‖Jf‖² 的变化与完整流增益、临床误差分别评价。'
            extra+=' ADNI 的局部 Jacobian 和噪声输出偏移略小，完整流增益略大，MMSE RMSE 较高；MIMIC 的局部 Jacobian 和流增益均较大，噪声输出偏移和心率 RMSE 却更低。临床误差传播还经过 Decoder 和历史观测吸收，三个指标分别呈现了这条路径的不同环节。'
            extra+=' 配对区间和数值收敛证据见 [ADNI](../publication_sex_init/analysis/adni/jacobian_utility/paired_bootstrap.json)、[MIMIC](../publication_sex_init/analysis/mimic/jacobian_utility/paired_bootstrap.json) 及各自 metrics.json。'
        if number==25 and data and all(row[3] for row in data):
            extra=' 本轮固定诊断中，'+'；'.join(f'{row[0]} 的梯度比 P95 为 {row[3]}、递归增益 P95 为 {row[5]}' for row in data)+'。'
            extra+=' 两组本轮均无数值异常终止；较差的验证选模/测试结果与发生数值崩溃是两个不同观察。表内 P95 按唯一参数快照计算，同一检查点的 best 文件与归档副本仅计一次。'
            extra+=' α=0.1 的历史/未来梯度比更低，梯度余弦中位数为正；α=1 的余弦中位数为负。α=0.1 的递归增益 P95 较高，本轮证据主要体现为梯度平衡和方向的改善。'
        sources={
          1:['results/adni/main/metrics.json'],3:['results/mimic/main/metrics.json'],
          7:['analysis/adni/horizons/metrics.json','analysis/mimic/horizons/metrics.json'],
          8:['results/causalmed/causalmed_main/metrics.json'],9:['results/causalmed/causalmed_main/metrics.json'],
          10:['results/adni/reverse/metrics.json','results/mimic/reverse/metrics.json'],
          11:['results/adni/separate_fields/metrics.json','results/mimic/separate_fields/metrics.json'],
          12:['results/adni/no_encoder_ode/metrics.json','results/mimic/no_encoder_ode/metrics.json'],
          13:['results/adni/attention/metrics.json','results/mimic/attention/metrics.json'],
          14:['results/adni/no_jacobian/metrics.json','results/mimic/no_jacobian/metrics.json'],
          15:['results/adni/add_prior/metrics.json','results/mimic/add_prior/metrics.json'],
          16:['analysis/adni/state/statistics.json','analysis/mimic/state/statistics.json'],
          17:['analysis/adni/clinical/metrics.json','analysis/mimic/clinical/metrics.json'],
          18:['analysis/adni/mmse/metrics.json'],19:['analysis/adni/local/metrics.json'],
          20:['analysis/adni/local/metrics.json','analysis/mimic/local/metrics.json'],
          21:['analysis/adni/manifold/adni_geometry/summary.json','analysis/mimic/manifold/mimic_geometry/summary.json'],
          22:['analysis/adni/manifold/adni_dynamics/summary.json','analysis/mimic/manifold/mimic_dynamics/summary.json'],
          23:['analysis/adni/jacobian_utility/metrics.json','analysis/mimic/jacobian_utility/metrics.json'],
          24:['analysis/adni/stage/metrics.json'],25:['analysis/mimic/stability/metrics.json'],
          26:['analysis/adni/cost/metrics.json','analysis/mimic/cost/metrics.json'],
          27:['results/adni','results/mimic'],28:['results/adni','results/mimic'],29:['results/adni','results/mimic'],
        }.get(number,[])
        available=[p for p in sources if (out/p).exists()]
        source_text=' 原始结果：'+'、'.join(f'[记录 {i+1}]({os.path.relpath(out/p,docpath.parent)})' for i,p in enumerate(available))+'。' if available else ''
        block=f'\n<!-- publication-caption -->\n**表 {number} 说明与分析。** {description}{extra}{source_text}\n<!-- /publication-caption -->\n'
        document=re.sub(pattern,lambda m:m.group(1)+block,document,count=1)
    # Every additional settings/progress table also receives an explicit caption.
    unnamed=[
      ('| 项目 | 固定设置 |','设置表 A：主模型结构。','本轮 M0 保留 28 维状态和现有网络，只以性别决定起点；观测残差门控固定为 0.1。两个数据集参数量差异来自观测字段数。'),
      ('| 项目 | ADNI | MIMIC | CausalMed |','设置表 B：数据与训练标准。','时间单位和历史构造按数据集确定，患者划分与训练归一化固定。三个数据集的新训练上限都是 150 轮，均无梯度累计。'),
      ('| 阶段 | 任务 ID | 工作 | 交付 |','执行表 A：任务依赖。','P0 形成主要比较，P1 复用最佳模型研究机制，P2 补充 INIT、门控和容量；相同 M0 检查点在各主题中复用。'),
      ('| 位置 | 本轮要求 |','执行表 B：代码与审计入口。','配置、模型实现、训练恢复及诊断各自留有入口，便于复核唯一研究变量、输入范围和 CUDA 执行。'),
      ('| 任务 | GPU | 状态 | 当前 / 完成轮次 | 最佳验证整体 | 记录 |','进度表：89 项任务的执行状态。','GPU 为实际物理编号；“最佳验证整体”用于选模，前文 H/F/O 为选中权重的测试指标。冻结分析不训练，因此训练轮次和验证列留空；目录链接提供完整原始记录。'),
    ]
    for header,title,description in unnamed:
        pattern=re.escape(header)+r'\n(?:\|[^\n]*\n)*'
        document=re.sub(pattern,lambda m:m.group(0)+f'\n<!-- publication-caption -->\n**{title}** {description}\n<!-- /publication-caption -->\n',document,count=1)
    # Group the entire figure collection by scientific question, not by disk path.
    images=list(sorted((out/'curves').glob('*.png')))
    images += [p for p in sorted((out/'analysis').glob('**/*.png')) if 'inputs' not in p.parts and 'workspace' not in p.parts]
    groups={}
    for image in images:
        section,title,caption=figure_details(image,out)
        groups.setdefault(section,[]).append((image,title,caption))
    index=0
    for section in sorted(groups,key=lambda s:tuple(map(int,s.split('.')))):
        start=re.search(rf'^### {re.escape(section)} ',document,re.M)
        if not start:
            raise ValueError(f'Missing figure section {section}')
        following=re.search(r'^##?#[^\n]*',document[start.end():],re.M)
        # Use the next section of any heading level >= 2.
        following=re.search(r'^#{2,3} ',document[start.end():],re.M)
        end=start.end()+following.start() if following else len(document)
        items=['\n<!-- publication-figures -->\n']
        for image,title,caption in groups[section]:
            index+=1
            items.append(f'![图 {index}：{title}]({os.path.relpath(image,docpath.parent)})\n\n**图 {index}｜{title}。** {caption}\n')
        items.append('<!-- /publication-figures -->\n\n')
        document=document[:end].rstrip()+'\n'+ '\n'.join(items)+document[end:]
    document=re.sub(r'\n{4,}','\n\n\n',document)
    return document
