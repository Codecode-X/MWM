"""Generate the central Ablation document from actual saved GPU artifacts."""
import fcntl
import json
import os
from pathlib import Path


def read(p, default=None):return json.loads(p.read_text()) if p.exists() else default


def link(path, parent):return os.path.relpath(path,parent)


def generate(out):
    out=Path(out)
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _generate(out)


def _generate(out):
    manifest=read(out/'manifest.json');queue=read(out/'queue.json',{});project=out.parents[2];doc=project/'ablation/Ablation.zh-CN.md';parent=doc.parent
    dispatch=read(out/'dispatch_revision.json')
    execution='空闲CUDA自动候补' if dispatch else 'CUDA 0，新训练'
    stopped_prior=read(out/'mimic_no_prior/stop_request.json',{}).get('prevent_automatic_resume',False)
    scheduling=('weak prior消融已按用户要求停止，已完成的ADNI结果保留。其余待跑组自动候补，已有训练不中断。' if stopped_prior else 'MIMIC去weak prior优先，其他待跑组自动候补，已有训练不中断。')
    labels={'no_prior':'去掉 weak prior','no_jacobian':'去掉 Jacobian 正则','separate_fields':'编码/生成双速度场（等参数）'}
    lines=['# Ablation：α=0.1主模型的模块消融','',
           '## 1. 要验证什么','',
           '当前主模型使用28维潜在状态，按时间从最早到最近吸收患者历史。INIT根据最后一条已知历史记录处的年龄给出初始状态，初始状态放在首次历史记录处；ODE负责两次检查之间的连续演化，UPDATE在检查到来时修正状态。','',
           '每次校正固定采用10%的UPDATE建议，保留90%的ODE演化后状态：','',
           r'\[z_k^+=0.9z_k^-+0.1U_\phi(z_k^-,y_k).\]','',
           '读完历史后，模型从历史截止点状态分别向过去回顾、向未来预测。编码和生成共用一个ODE速度场。训练同时包含重建、weak prior、Jacobian和ODE参数L2。','',
           '本轮以这套完整模型为参照，每次只去掉一项监督或改变一项建模设置，回答四个问题：读取历史的方向是否重要？校正前预测监督是否有用？Jacobian正则是否有用？相同ODE参数预算下，一个统一速度场和两个任务速度场哪种效果更好？','',
           '## 2. 实验矩阵与参数公平性','',
           '| 组别 | 历史读取 | weak prior权重 | Jacobian权重 | ODE | α | 本轮执行 |','|---|---|---:|---:|---|---:|---|',
           '| 完整主模型 | 正序 | 0.01 | 0.01 | 统一场 | 0.1 | 复用现有结果 |',
           '| 反序消融 | 最近读到最早 | 0.01 | 0.01 | 统一场 | 0.1 | 复用CUDA 2/3正在进行的实验 |',
           f'| 去掉weak prior | 正序 | 0 | 0.01 | 统一场 | 0.1 | {"已停止，保留已完成ADNI" if stopped_prior else execution} |',
           f'| 去掉Jacobian | 正序 | 0.01 | 0 | 统一场 | 0.1 | {execution} |',
           f'| 双速度场 | 正序 | 0.01 | 0.01 | 编码场＋生成场 | 0.1 | {execution} |','',
           '**每一组都严格保持ADNI 44,875参数、MIMIC 43,280参数。** 不填充无用参数；预检检查实际唯一参数量、梯度和两个场是否真正独立。INIT、UPDATE、Decoder、28维状态及门控系数保持一致。','',
           '### 2.1 双速度场怎样做到完全等参数','',
           '完整模型的ODE是28→188→28，隐藏层使用Tanh。双场组把同一笔ODE预算分成两部分：编码ODE负责吸收历史时的演化，生成ODE负责从截止点向过去和未来生成。两个场之间不共享任何参数。','',
           '| 速度场 | 结构 | 输出层偏置 | 参数量 |','|---|---|---|---:|',
           '| 主模型统一场 | 28→188→28 | 有，28维 | 10,744 |',
           '| 双场：历史编码场 | 28→94→28 | 无 | 5,358 |',
           '| 双场：历史/未来生成场 | 28→94→28 | 有，28维 | 5,386 |',
           '| 双场合计 | 两个完全独立的场 | | **10,744** |','',
           '两组的ODE预算相同，差别是把容量集中到一个共享函数，还是分配给两个专用函数。双场组不改变其他网络宽度。编码场不设输出偏置，正好省下28个参数，使总量精确匹配。','',
           'INIT、UPDATE、Decoder从相同seed取得相同初始张量。两个94宽场使用相同的初始权重，生成场的输出偏置为0；它们之后独立更新。初始化取原188宽场的前94个隐藏单元，输出权重乘√2，补偿隐藏单元减半后的初始速度尺度。预检记录两个新场之间、以及新场与主模型初始预测之间的实际差异。','',
           '### 2.2 方向组复用什么','',
           '正序主模型从首次检查读取到历史截止点，再分别回顾过去和预测未来。反序组从最近检查读取到首次检查，得到已融合整段历史的首次检查状态，再沿同一ODE向未来生成完整历史和未来。','',
           '反序组保留一条使用相同网络和α=0.1的正序辅助路径，只计算原weak prior。辅助路径不提供主生成起点。两种方向的参数量、年龄输入、监督目标、损失权重和数据划分相同。','',
           '## 3. 损失怎样计算','',
           r'\[\mathcal L=\mathcal L_{\mathrm{rec}}+\lambda_p\mathcal L_{\mathrm{prior}}+\lambda_J\mathcal L_J+0.02\mathcal L_{\mathrm{ODE-L2}}.\]','',
           '**重建项。** 所有有效历史目标和未来目标都参与数值MSE＋类别交叉熵。每个原始字段按当前batch的有效目标数归一化，再对字段求和。缺失目标不计入损失；训练输入dropout只遮蔽输入，不删除真实监督目标。所有组保留这一项和它的原有梯度路径。','',
           '**weak prior。** 历史第2次及以后，模型先用ODE演化到检查时刻，在吸收这次观测之前预测它；随后才UPDATE。该项使用相同的MSE＋交叉熵及有效字段掩码。去掉weak prior组只将λp从0.01改成0，仍保存原始诊断分量。','',
           '**Jacobian。** 对主生成轨迹中有效的28维状态，精确计算速度场对状态的28×28 Jacobian，784项平方求和，再对状态求平均；采样状态停止梯度。去掉Jacobian组只将λJ从0.01改为0。双场组在相同生成状态上分别计算两个场，再取场间平均：','',
           r'\[\mathcal L_J=\frac1{|\mathcal F|}\sum_{f\in\mathcal F}\frac1{|\mathcal Z|}\sum_{z\in\mathcal Z}\|J_f(z)\|_F^2.\]','',
           '**ODE L2。** 对全部独立ODE参数的平方求和，系数固定0.02。统一场的别名只计数一次；双场分别计数，但两场总参数仍为10,744。这里不对双场L2再除以2，保证每个独立参数接受相同的惩罚系数。','',
           '## 4. 数据、训练与测试','',
           '| 项目 | ADNI | MIMIC |','|---|---|---|',
           '| 数据缓存 | paper_odeae/cache/local_available.pkl | /data/MIMIC/derived/latentode_iv31/dataset.pkl |',
           '| 训练 / 验证 / 测试 | 2,093 / 393 / 414人 | 21,517 / 4,649 / 4,536条患者序列 |',
           '| 字段 / 展开输出维度 | 40 / 49 | 31 / 44 |',
           '| 时间单位 | 年 | 小时 |',
           '| 数值归一化 | 训练集min-max | 训练集min-max |',
           '| 训练历史构造 | 枚举每位患者全部非空历史前缀 | 每轮按患者与epoch固定种子，从6/12/18/24小时截止点采样 |',
           '| 验证与主测试 | 原全部前缀评测 | 原24小时历史上下文，预测后续至48小时 |',
           '| batch / 梯度累积 | 32 / 无 | 128 / 无，保留末批13人 |',
           '| 评测batch | 32 | 每次32，串行复用原四个固定患者分片 |',
           '| 随机种子 | 0 | 0 |',
           '| 学习率 / 衰减 | Adam，0.003；每2000步乘0.97 | 相同 |',
           '| 梯度裁剪 | 全模型范数10 | 相同 |',
           '| ODE | Tsit5，rtol=1e−4，atol=1e−6 | 相同 |',
           '| 新训练预算 | 最大150轮，至少50轮，验证整体40轮不改善早停 | 相同 |',
           '| 选模 | 验证整体重建误差最低 | 相同 |','',
           ('全部新消融使用Conda ode环境，每组单卡训练。调度器扫描物理CUDA 0、1、2、3，只有GPU锁可获得且没有计算进程时才派发任务；每张卡同一时刻只运行一组。'+scheduling+'任务换到新GPU时，先在实际GPU重新预检，再执行原冻结训练代码，接入SwanLab。' if dispatch else '全部新消融在Conda ode环境、物理CUDA 0上逐组运行。首先对六组全部进行CUDA预检，再按“ADNI去prior → MIMIC去prior → ADNI去Jacobian → MIMIC去Jacobian → ADNI双场 → MIMIC双场”的顺序训练，接入SwanLab。'),' ',
           '每组结束后重新载入best.pt，核对验证误差，再计算测试历史回顾、测试未来预测、测试整体以及逐字段指标。MIMIC额外保留数值标准化MSE和6/12小时历史上下文测试。测试目标不进入UPDATE。','',
           '基线复用已有主模型。ADNI主模型最大150轮、实际早停70轮；MIMIC按此前指定复用已训练237轮的正序组，选择第233轮最佳验证检查点，不再补跑正序150轮。新的MIMIC消融和反序组上限仍为150轮，下面列出各组实际完成轮数。','',
           '## 5. CUDA核验与当前进度','',
           '| 新任务 | 状态 | 物理GPU | 设计总参数 | 实测总参数 | 独立ODE参数 | 核验 |','|---|---|---:|---:|---:|---:|---|']
    for job in manifest['jobs']:
        folder=Path(job['output']);check=read(folder/'checks.json');status=queue.get('jobs',{}).get(job['name'],{}).get('status','queued')
        assigned=queue.get('jobs',{}).get(job['name'],{}).get('physical_gpu')
        if dispatch and status=='queued':assigned='待分配'
        if assigned is None:assigned=0
        vals=[job['name'],status,str(assigned),str(manifest['same_parameter_budget'][job['dataset']]),str(check['parameters']) if check else '—',str(check['unique_ode_parameters']) if check else '—',check['status'] if check else '待核验']
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['','预检不执行optimizer.step()。它验证去损失组的初始输出与完整主模型逐位一致、仅移除了目标损失的梯度；验证双场预算精确相等、两场参数无交集、两场均获得梯度；同时检查未来观测值不会影响模型输出、全部实际任务参数的梯度有限。','']
    checked=[(j,read(Path(j['output'])/'checks.json')) for j in manifest['jobs']]
    if any(c for j,c in checked):
        lines += ['### 5.1 训练前的数值核验','',
                  '| 任务 | 移除指定损失后的梯度相对误差 | 两个初始场的最大速度差 | 初始预测相对主模型的差异 |','|---|---:|---:|---:|']
        for job,c in checked:
            if not c:continue
            vals=[job['name'],f"{c['removed_loss_gradient_relative_error']:.3g}" if 'removed_loss_gradient_relative_error' in c else '—',
                  f"{c['initial_two_field_max_absolute_difference']:.3g}" if 'initial_two_field_max_absolute_difference' in c else '—',
                  f"{c['prediction_relative_difference_from_main_initialization']:.6f}" if 'prediction_relative_difference_from_main_initialization' in c else '0（逐位相同）']
            lines.append('| '+' | '.join(vals)+' |')
        lines += ['','双场组以两个相同的94宽函数开始训练；表中相对主模型的差异来自将一个188宽函数重新分配为两个94宽函数。上表描述训练前的初始化和梯度，训练后的指标见后两节。','']
    directions=Path(manifest['direction_study']);direction_queue=read(directions/'queue.json',{}).get('jobs',{})
    for dataset in ['adni','mimic']:
        reference=Path(manifest['references'][dataset]);reverse=directions/(dataset+'_reverse')
        groups=[('完整主模型',reference,read(reference/'queue.json',{}).get('status','已训练')),
                ('反序消融',reverse,direction_queue.get(dataset+'_reverse',{}).get('status','queued'))]
        groups += [(labels[j['component']],Path(j['output']),queue.get('jobs',{}).get(j['name'],{}).get('status','queued')) for j in manifest['jobs'] if j['dataset']==dataset]
        data=[]
        lines += [f'## 6. {dataset.upper()}结果' if dataset=='adni' else '## 7. MIMIC结果','',
                  '| 模型 | 状态 | 参数量 | 已完成 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |','|---|---|---:|---:|---:|---:|---:|']
        for name,folder,status in groups:
            hs=read(folder/'history.json',[]);result=read(folder/'metrics.json')
            if dataset=='mimic' and folder==reference:result=result or read(directions/'mimic_forward_existing/metrics.json')
            if (folder/'stop_request.json').exists() and not result:status='复用已停止训练'
            best=min(hs,key=lambda h:h['validation']['overall']['reconstruction_error']) if hs else None
            selected=result['validation'] if result else best['validation'] if best else None
            n=result['epochs_run'] if result else len(hs);e=result['best_epoch'] if result else best['epoch'] if best else '—'
            vals=[name,'complete' if result else status,str(manifest['same_parameter_budget'][dataset]),f'{n} / {e}']
            vals += [f"{selected[k]['reconstruction_error']:.6f}" if selected else '—' for k in ['interpolation','forecast','overall']]
            lines.append('| '+' | '.join(vals)+' |');data.append((name,folder,hs,result))
        lines += ['','| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 |'+(' 数值标准化MSE |' if dataset=='mimic' else ''),
                  '|---|---:|---:|---:|---:|'+('---:|' if dataset=='mimic' else '')]
        for name,folder,hs,result in data:
            temporary=read(folder/'temporary_test_epoch030.json')
            interim=read(Path(temporary['metrics_path'])) if temporary else None
            scored=result or interim
            test=scored['test'] if scored else None
            epoch_label=str(result['best_epoch']) if result else f"{interim['checkpoint_epoch']}（临时）" if interim else '—'
            vals=[name,epoch_label]
            vals += [f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation','forecast','overall']]
            if dataset=='mimic':vals += [f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—']
            lines.append('| '+' | '.join(vals)+' |')
        if any(d[2] for d in data):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            curves=out/'curves';curves.mkdir(exist_ok=True);fig,axes=plt.subplots(2,3,figsize=(16,8.5))
            english=['main alpha=0.1','reverse','without prior','without Jacobian','two fields, matched params']
            keys=['train','interpolation','forecast','overall','gradient','maximum']
            for (_,folder,hs,result),label in zip(data,english):
                if not hs:continue
                for ax,key in zip(axes.flat,keys):
                    values=[];epochs=[]
                    for row in hs:
                        value=row.get('training_loss') if key=='train' else row.get('maximum_batch_gradient_norm') if key=='maximum' else row.get('training_components',{}).get('gradient_norm') if key=='gradient' else row['validation'][key]['reconstruction_error']
                        if value is not None:epochs.append(row['epoch']);values.append(value)
                    if values:ax.plot(epochs,values,label=label,marker='o' if len(epochs)==1 else None,markersize=3)
            for ax,title in zip(axes.flat,['Training total loss','Validation history','Validation future','Validation overall','Mean pre-clip batch gradient','Max pre-clip batch gradient']):
                ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.axvline(150,color='grey',ls=':',alpha=.5);ax.legend(fontsize=7)
            axes[1,1].set_yscale('log');axes[1,2].set_yscale('log');fig.tight_layout()
            plot=curves/(dataset+'_ablation.png');fig.savefig(plot,dpi=180);fig.savefig(curves/(dataset+'_ablation.pdf'));plt.close(fig)
            lines += ['',f'![{dataset.upper()}训练、验证与梯度曲线]({link(plot,parent)})','']
        lines += ['','曲线和表格读取各组实际日志。正在训练的组展示当前最佳验证检查点，测试列在训练结束后填写；纵向虚线标出150轮。','']
        if data[0][3]:
            main=data[0][3]
            for name,folder,hs,result in data[1:]:
                if result:
                    delta=result['test']['forecast']['reconstruction_error']-main['test']['forecast']['reconstruction_error']
                    lines += [f'- {name}的测试未来误差减主模型为{delta:+.6f}。']
            lines.append('')
    lines += ['## 8. 原始记录与运行方式','',
              f'[实验矩阵与源码指纹]({link(out/"manifest.json",parent)}) · [GPU候补队列]({link(out/"queue.json",parent)}) · [已有方向实验]({link(directions/"queue.json",parent)})','']
    for job in manifest['jobs']:
        folder=Path(job['output']);lines += [f"- {job['name']}：[配置]({link(folder/'launch_config.json',parent)}) · [预检]({link(folder/'checks.json',parent)}) · [训练记录]({link(folder,parent)}/)"]
    lines += ['','各组保存独立的best.pt、last.pt、优化器与调度器状态、随机数状态、history.json、hardware.json、SwanLab记录和最终metrics.json。原实验检查点与日志不覆盖。','',
              '```bash','bash scripts/run_main_alpha01_ablation.sh','```','',
              '重复启动时先检查已有结果和单GPU锁，完成的组直接复用，未完成的组从自身last.pt继续；累计训练轮次不超过150。','']
    recovery=read(out/'latest_recovery.json')
    if recovery:
        lines += ['## 9. 中断后的续训记录','',
                  '本次进程中断后，从各组最近完成一轮的last.pt恢复模型、Adam、学习率调度器、最佳验证分数、早停计数和累计优化步数。原训练脚本在每轮开始按固定种子重新设置随机数和采样顺序；中断轮中尚未保存的部分从该轮开头重算。','',
                  '| 实验 | 物理GPU | 检查点已完成轮次 | 续训起始轮次 | 已保存优化步数 | 总轮次上限 |',
                  '|---|---:|---:|---:|---:|---:|']
        for job in recovery['jobs']:
            lines.append('| '+' | '.join(str(job[k]) for k in ['job','physical_gpu','completed_epoch','next_epoch','optimizer_steps','max_total_epochs'])+' |')
        lines += ['',
                  '已完成的ADNI反序组和ADNI主模型直接复用。恢复时CUDA 0继续ADNI去prior，后续任务纳入同一队列；此前取消的MIMIC正序150轮重训保持取消。模型结构、损失、batch size和等参数预算均沿用中断前配置。','',
                  '恢复时数据盘尚未挂载，首次启动没有执行训练更新。数据盘恢复后，重新核对MIMIC缓存SHA256与原实验一致，再启动续训。','',
                  f'[恢复清单与检查点备份]({link(Path(recovery["recovery_directory"])/"manifest.json",parent)})。备份保留了续训前的best.pt、last.pt、训练历史、配置和SwanLab记录。','']
    if dispatch:
        lines += ['## 10. 空闲GPU自动候补','',
                  ('weak prior消融已按用户要求停止。ADNI去prior组此前已完成，保留其全部结果；MIMIC去prior组停止于已保存的第101轮，不再自动恢复。释放的GPU由其余候补组自动领取；Jacobian、双速度场和反序实验继续原计划。' if stopped_prior else '现有ADNI去prior训练进程由新调度器直接接管，训练进程持续运行。MIMIC去prior优先使用当前空闲的CUDA 2。其余组按下面顺序领取后续空闲GPU；CUDA 1的无UPDATE组、CUDA 3的反序组及随后测试任务保留原调度。'),' ',
                  '| 待派发顺序 | 实验 | 当前状态 | 物理GPU |','|---:|---|---|---|']
        for index,name in enumerate(dispatch['priority'],1):
            q=queue.get('jobs',{}).get(name,{})
            lines.append(f"| {index} | {name} | {q.get('status','queued')} | {q.get('physical_gpu') if q.get('status')!='queued' else '待空闲后分配'} |")
        lines += ['',
                  '每组仍使用独立的模型和Adam，ADNI batch=32、MIMIC batch=128，不使用梯度累计。GPU分配只改变物理设备编号；实际张量设备、参数量和设备编号写入hardware.json及SwanLab。所有续训累计最多150轮。','',
                  f'[调度变更与接管记录]({link(out/"dispatch_revision.json",parent)}) · [实时队列]({link(out/"queue.json",parent)})','']
    temporary=read(out/'adni_no_jacobian/temporary_test_epoch030.json')
    if temporary:
        interim=read(Path(temporary['metrics_path']))
        lines += ['## 11. ADNI去掉Jacobian：第30轮临时测试','',
                  '按当前最佳验证结果选定第30轮，复制独立参数快照后在CUDA 0评测。现有训练继续；本次不执行优化更新。正式训练完成后的最终测试单独保存。','',
                  '| 测试区域 | MSE＋CE | 数值MSE之和 | 类别CE之和 | 数值标准化MSE均值 |','|---|---:|---:|---:|---:|']
        for region,label in [('interpolation','历史回顾'),('forecast','未来预测'),('overall','整体')]:
            x=interim['test'][region];numeric=interim['clinical_test'][region]['mean_standardized_numeric_mse']
            lines.append(f"| {label} | {x['reconstruction_error']:.8f} | {x['sum_feature_mse']:.8f} | {x['sum_feature_cce']:.8f} | {numeric:.8f} |")
        lines += ['',
                  f"[完整评测报告与逐字段表]({link(Path(temporary['report_path']),parent)}) · [全部指标JSON]({link(Path(temporary['metrics_path']),parent)}) · [逐字段CSV]({link(Path(temporary['metrics_path']).parent/'per_field_metrics.csv',parent)})",'',
                  '完整报告包括40个字段在历史、未来和整体区域的损失，数值字段原量纲MAE、RMSE和标准化误差，类别字段准确率、宏平均F1、混淆矩阵，以及校正前预测和速度场Jacobian诊断值。','']
    recovery2=read(out/'latest_recovery_20261105.json')
    if recovery2:
        lines += ['## 12. 再次中断后的恢复','',
                  '数据盘恢复后，未完成的组从最近完整轮次继续，恢复模型、Adam、学习率调度器、累计优化步数和早停状态。单轮中未保存的部分从该轮起点重算。设备由空闲GPU调度器重新分配，模型和训练超参数保持原值。','',
                  '| 实验 | 已保存轮次 | 续训起始轮次 | 已保存优化步数 | 原物理GPU | 当前物理GPU |','|---|---:|---:|---:|---:|---:|']
        for job in recovery2['jobs']:
            name=job['job']
            actual=(read(directions/'queue.json',{}).get('jobs',{}).get(name,{}) if name=='mimic_reverse' else queue.get('jobs',{}).get(name,{})).get('physical_gpu','—')
            lines.append(f"| {name} | {job['last_completed_epoch']} | {job['next_epoch']} | {job['optimizer_steps']} | {job['physical_gpu_before']} | {actual} |")
        lines += ['', '本轮累计仍最多150轮。weak prior消融与MIMIC正序150轮补跑保持停止；已完成的ADNI结果直接复用。','']
    utility=out.parent/'jacobian_utility_adni'
    if (utility/'manifest.json').exists():
        utility_progress=read(utility/'progress.json',{})
        lines += ['## 13. Jacobian正则的实际作用测试','',
                  '使用两组固定的第30轮检查点，比较相同历史测量噪声下的预测偏移、小截止点状态扰动在未来ODE中的放大量，以及按真实预测时长分组的测试误差。模型不重新训练。','',
                  f"当前状态：{utility_progress.get('status','待运行')}；阶段：{utility_progress.get('stage','已完成') if utility_progress.get('status')!='complete' else '全部完成'}。",'',
                  f'[诊断进度]({link(utility/"progress.json",parent)}) · [实验配置]({link(utility/"manifest.json",parent)})','']
        if (utility/'report.zh-CN.md').exists():
            lines += [f'[完整实验结果、曲线和机制解释]({link(utility/"report.zh-CN.md",parent)}) · [定量指标]({link(utility/"metrics.json",parent)})','']
            um=read(utility/'metrics.json')
            if um:
                a=um['models']['regularized'];b=um['models']['unregularized']
                lines += ['| 实测效果 | 保留Jacobian正则 | 去掉Jacobian正则 |','|---|---:|---:|',
                          f"| 5年ODE误差放大倍数，中位数 | {a['flow']['flow_gain_median'][-1]:.6f} | {b['flow']['flow_gain_median'][-1]:.6f} |",
                          f"| 同样5%测量噪声引起的MMSE预测偏移RMSE，分 | {a['noise_0.05']['mmse_prediction_shift_rmse'][0]:.6f} | {b['noise_0.05']['mmse_prediction_shift_rmse'][0]:.6f} |",
                          f"| 超过5年真实预测的MMSE RMSE，分 | {a['noise_0']['mmse_rmse'][-1]:.6f} | {b['noise_0']['mmse_rmse'][-1]:.6f} |",'']
    text='\n'.join(lines);tmp=doc.with_suffix('.tmp.md');tmp.write_text(text);tmp.replace(doc)
    atomic=out/'report_status.json';tmp=atomic.with_suffix('.tmp');tmp.write_text(json.dumps({'document':str(doc),'phase':queue.get('phase'),'current_job':queue.get('current_job')}));tmp.replace(atomic)
