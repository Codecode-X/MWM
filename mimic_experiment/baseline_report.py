"""Live five-model MIMIC comparison from saved scalar/field metrics; CPU reporting only."""
from pathlib import Path
import json
from .report import read
from .prepare import atomic

NAMES={'healthflux_kl':'HealthFlux-KL','ode_ae':'ODE-AE','ode_rnn':'ODE-RNN','latent_ode':'Latent ODE'}


def generate(output):
    out=Path(output);manifest=read(out/'manifest.json',{});queue=read(out/'queue.json',{});rows=[]
    reference=Path(manifest.get('reference_output',out.parent/'latent2x_both_cuda'))
    names={**NAMES,**manifest.get('additional_model_names',{})};outputs=manifest.get('model_outputs',{})
    sources=[('latent2x_both','latent2x both',reference/'seed_0')]+[(key,title,Path(outputs.get(key,out/key/'seed_0'))) for key,title in names.items()]
    for key,title,path in sources:
        history=read(path/'history.json',[]);final=read(path/'metrics.json');hardware=read(path/'hardware.json',{})
        best=min(history,key=lambda row:row['validation']['overall']['reconstruction_error']) if history else None
        source_queue=read(manifest.get('model_queues',{}).get(key,out/'queue.json'),{})
        state=source_queue.get('jobs',{}).get(key,{})
        rows.append({'key':key,'name':title,'path':str(path),'hardware':hardware,'history':history,'result':final,
                     'validation':final['validation'] if final else best['validation'] if best else None,
                     'status':'complete' if final else state.get('status','waiting'),'phase':state.get('phase','training'),
                     'epochs':final['epochs_run'] if final else len(history),'best_epoch':final['best_epoch'] if final else best['epoch'] if best else None})
    lines=['# MIMIC-IV：模型横向比较','',
           '## 1. 所有模型回答同一个预测问题','',
           '每位患者取首次ICU住院的前48小时。训练21,517人，验证4,649人，测试4,536人，使用同一份chronology_v2缓存、31个原始字段、44维展开输出及训练集min-max归一化。','',
           '训练每人每轮选一个6、12、18或24小时历史截止点。主要验证与测试固定读取前24小时，评价之后24小时的观测。数值字段计算MSE，类别字段计算交叉熵；每个字段先在有效观测上取均值，再按字段求和。验证整体误差用于选模。','',
           '各baseline独立单卡训练。每次优化使用128位患者，按四组在单卡上累积梯度，每组最多32人。每个字段使用整批有效观测数作为分母，KL按患者数或有效观测节点数归约，ODE参数L2每个优化步骤计一次。','',
           'Adam学习率0.003，按0.97^(step/2000)衰减，输入dropout为0.1，梯度范数上限10，最多300轮、至少50轮、验证整体误差连续40轮未改善时早停。全部训练接入SwanLab，重新载入最佳检查点后进行测试。','',
           '## 2. 各模型保留什么结构','',
           '| 模型 | 物理GPU | 总参数 | 可训练参数 | 状态与未来生成方式 |','|---|---|---:|---:|---|']
    descriptions={'latent2x_both':'28维；共享ODE自由推演；prior=0.01，Jacobian=0.01',
                  'healthflux_kl':'256维确定性＋128维随机；条件ODE＋先验均值；KL=0.01',
                  'ode_ae':'6维随机潜在状态；反向编码、独立生成ODE；KL=1',
                  'ode_rnn':'32维；ODE＋GRU；未来以模型预测作递归输入',
                  'latent_ode':'32维随机潜在状态；64维ODE-RNN编码器；KL=1',
                  'deterministic_matched':'6维确定性状态；反向编码与独立生成ODE；43,280参数匹配；KL=0'}
    for row in rows:
        hw=row['hardware'];lines.append(f"| {row['name']} | {hw.get('physical_gpu','—')} | {hw.get('parameters','—'):,} | {hw.get('trainable_parameters','—'):,} | {descriptions[row['key']]} |" if hw else f"| {row['name']} | {queue.get('jobs',{}).get(row['key'],{}).get('physical_gpu','—')} | — | — | {descriptions[row['key']]} |")
    lines += ['',
              'ODE-AE、Latent ODE和我们方法使用原有自适应Tsit5。HealthFlux使用RK4，按小时推进，年龄条件以年提供；输入分为生命体征、化验、GCS三组做注意力汇聚，性别进入静态背景。它在有效观测节点计算后验与转移先验之间的KL。','',
              '各模型均获得相同的年龄信息：ODE-AE和我们方法通过年龄初始化，HealthFlux通过年龄条件编码，ODE-RNN和Latent ODE增加年龄协变量输入通道。该通道以年龄/100输入GRU，44维目标保持相同。','',
              '历史区间一栏中，ODE-RNN统计读取当前观测之前的因果预测误差；其余模型统计由已知历史形成的潜在估计所生成的历史重构误差。未来一栏统一评价截止点之后的真实有效观测。','',
              '## 3. 训练进度与主要测试结果','',
              '| 模型 | 状态 | 已训练 / 最佳轮次 | 验证整体 | 测试历史 | 测试未来 | 测试整体 | 未来数值标准化MSE |','|---|---|---:|---:|---:|---:|---:|---:|']
    def number(value):return f'{value:.5f}' if value is not None else '—'
    for row in rows:
        val=row['validation'];test=row['result']['test'] if row['result'] else None
        values=[row['name'],row['status']+(f" / {row['phase']}" if row['status']!='complete' else ''),f"{row['epochs']} / {row['best_epoch'] or '—'}",
                number(val['overall']['reconstruction_error'] if val else None)]
        values += [number(test[key]['reconstruction_error'] if test else None) for key in ['interpolation','forecast','overall']]
        values += [number(test['mean_standardized_numeric_mse'] if test else None)]
        lines.append('| '+' | '.join(values)+' |')
    ref_result=rows[0]['result'];ref_test=ref_result['test'] if ref_result else None
    if ref_test:lines+=['',f"保持最近观测值的数值标准化MSE为 **{ref_test['carry_forward_mean_standardized_numeric_mse']:.5f}**。该分数先将27个数值字段的原始单位MSE除以各自训练标准差平方，再等权平均。"]
    for row in rows[1:]:
        if row['history']:
            minutes=sum(h['epoch_seconds'] for h in row['history'])/len(row['history'])/60
            lines+=['',f"{row['name']}：已完成轮次平均约{minutes:.2f}分钟/轮。"]
    lines+=['','## 4. 未来不同范围的累计误差','',
            '| 模型 | 未来6小时 | 未来12小时 | 未来24小时 |','|---|---:|---:|---:|']
    for row in rows:
        test=row['result']['test'] if row['result'] else None
        lines.append('| '+' | '.join([row['name']]+[number(test[f'forecast_{h}h']['reconstruction_error'] if test else None) for h in [6,12,24]])+' |')
    lines+=['','## 5. 原始单位下的未来误差','',
            '保持最近值使用同一段历史；历史中从未测到的字段使用训练均值或训练众数。以下各栏的目标时刻及字段掩码相同。','',
            '### MAE','',
            '| 字段 | 单位 | '+' | '.join(row['name'] for row in rows)+' | 保持最近值 |','|'+'|'.join(['---','---']+['---:']*(len(rows)+1))+'|']
    mae_rows=[]
    if ref_test:
        for field,value in ref_test['clinical_forecast'].items():
            if 'mae' not in value:continue
            record={'field':field,'unit':value['unit']};values=[]
            for row in rows:
                test=row['result']['test'] if row['result'] else None
                v=test['clinical_forecast'].get(field,{}).get('mae') if test else None
                values.append(number(v));record[row['key']]=v
            last=ref_test['carry_forward'][field]['mae'];record['carry_forward']=last;mae_rows.append(record)
            lines.append('| '+' | '.join([field,value['unit']]+values+[number(last)])+' |')
    lines+=['','### RMSE','',
            '| 字段 | 单位 | '+' | '.join(row['name'] for row in rows)+' | 保持最近值 |','|'+'|'.join(['---','---']+['---:']*(len(rows)+1))+'|']
    rmse_rows=[]
    if ref_test:
        for field,value in ref_test['clinical_forecast'].items():
            if 'rmse' not in value:continue
            record={'field':field,'unit':value['unit']};values=[]
            for row in rows:
                test=row['result']['test'] if row['result'] else None
                v=test['clinical_forecast'].get(field,{}).get('rmse') if test else None
                values.append(number(v));record[row['key']]=v
            last=ref_test['carry_forward'][field]['rmse'];record['carry_forward']=last;rmse_rows.append(record)
            lines.append('| '+' | '.join([field,value['unit']]+values+[number(last)])+' |')
    lines+=['','[下载MAE表](per_field_mae.csv) · [下载RMSE表](per_field_rmse.csv)','',
            '## 6. GCS未来分类准确率','',
            '| 模型 | 睁眼 | 运动 | 语言 |','|---|---:|---:|---:|']
    for row in rows:
        test=row['result']['test'] if row['result'] else None
        values=[f"{test['clinical_forecast'][field]['accuracy']*100:.2f}%" if test else '—' for field in ['gcs_eye','gcs_motor','gcs_verbal']]
        lines.append('| '+' | '.join([row['name']]+values)+' |')
    if ref_test:lines.append('| '+' | '.join(['保持最近值']+[f"{ref_test['carry_forward'][field]['accuracy']*100:.2f}%" for field in ['gcs_eye','gcs_motor','gcs_verbal']])+' |')
    lines+=['','## 7. 历史更短时的未来预测','',
            '| 模型 | 6小时历史患者数 | 6小时后至48小时误差 | 12小时历史患者数 | 12小时后至48小时误差 |','|---|---:|---:|---:|---:|']
    for row in rows:
        result=row['result'];extra=result['test_early_contexts'] if result else None
        lines.append('| '+' | '.join([row['name'],str(extra['6']['patients']) if extra else '—',number(extra['6']['forecast']['reconstruction_error'] if extra else None),str(extra['12']['patients']) if extra else '—',number(extra['12']['forecast']['reconstruction_error'] if extra else None)])+' |')
    lines+=['','## 8. 配置、检查与输出','',
            '[实验设计](../../BASELINE_COMPARISON.zh-CN.md) · [任务配置与来源](manifest.json) · [运行队列](queue.json) · [完整比较数据](comparison.json)','',
            '每个模型目录保存best.pt、last.pt、history.json、metrics.json、hardware.json和swanlab.json。启动前的CUDA检查、完整48小时预检查保存在checks目录，执行源码固定在source_snapshot。每30秒更新本报告。','',
            '```bash','bash scripts/run_mimic_deterministic_matched.sh' if manifest.get('additional_model_names') else 'bash scripts/run_mimic_baselines.sh','```','']
    if any(row['history'] for row in rows[1:]):lines+=['![训练与验证曲线](training_curves.png)','']
    (out/'summary.md').write_text('\n'.join(lines))
    atomic(out/'comparison.json',{'dataset_sha256':manifest.get('dataset_sha256'),'models':[{k:v for k,v in row.items() if k!='history'} for row in rows],'future_mae':mae_rows,'future_rmse':rmse_rows})
    import csv
    for name,records in [('mae',mae_rows),('rmse',rmse_rows)]:
        if not records:continue
        with (out/f'per_field_{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    # CPU figure rendering from already recorded metrics.
    signature=[len(row['history']) for row in rows[1:]]
    if signature!=read(out/'plot_signature.json') and any(signature):
        plot(out,rows);atomic(out/'plot_signature.json',signature)


def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for row in rows:
        history=row['history']
        if not history:continue
        x=[h['epoch'] for h in history]
        label=row['name'].replace('deterministic_22545（同预算）','Deterministic ODE-AE (matched)')
        axes[0].plot(x,[h['validation']['overall']['reconstruction_error'] for h in history],label=label)
        axes[1].plot(x,[h['validation']['forecast']['reconstruction_error'] for h in history],label=label)
    for ax,title in zip(axes,['Validation overall','Validation forecast']):
        ax.set(xlabel='Epoch',ylabel='Mixed-field error',title=title);ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=180)
    plt.close(fig)
