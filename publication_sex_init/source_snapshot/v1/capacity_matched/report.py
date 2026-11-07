"""Saved-metric reporting for native and equal-budget baseline runs."""
import csv
from datetime import datetime,timezone
import os
from pathlib import Path

from paper_odeae.scaling_study import atomic
from mimic_experiment.report import read

ROOT=Path(__file__).resolve().parents[1]
NAMES={'ode_ae':'ODE-AE（概率版）','ode_rnn':'ODE-RNN','latent_ode':'Latent ODE'}


def report(output,manifest):
    output=Path(output);queue=read(output/'queue.json',{});all_rows=[]
    for dataset,definition in manifest['datasets'].items():
        out=Path(definition['output']);rows=[]
        paths=[('reference','latent2x both','参照',Path(definition['reference']))]
        for kind in ['ode_ae','ode_rnn','latent_ode']:
            paths += [(kind+'_native',NAMES[kind],'原容量',Path(definition['native'][kind])),(kind,NAMES[kind],'同预算',out/kind/'seed_0')]
        for key,title,capacity,path in paths:
            history=read(path/'history.json',[]);final=read(path/'metrics.json');hw=read(path/'hardware.json',{})
            best=min(history,key=lambda h:h['validation']['overall']['reconstruction_error']) if history else None
            status=queue.get('jobs',{}).get(dataset+'_'+key,{})
            rows.append({'key':key,'name':title,'capacity':capacity,'path':str(path),'hardware':hw,
                         'status':'complete' if final else status.get('status','waiting'),'phase':status.get('phase','waiting'),
                         'gpu':hw.get('physical_gpu',status.get('physical_gpu')),'epochs':final['epochs_run'] if final else len(history),
                         'best_epoch':final['best_epoch'] if final else best['epoch'] if best else None,
                         'validation':final['validation'] if final else best['validation'] if best else None,'result':final,'history':history})
        lines=[f'# {dataset.upper()}：与latent2x both相同参数预算的基线比较','',
               '## 1. 本轮比较什么','',
               f"本轮为概率ODE-AE、ODE-RNN、Latent ODE新增同预算版本，目标为 **{definition['target']:,}个总参数和可训练参数**。latent2x both及原容量模型读取已有结果，新版从seed 0初始化训练。新版本保持各模型原有编码方向、生成方式、概率假设、损失与训练预算。",'',
               'ODE-AE仍是6维Gaussian状态和独立编码/生成ODE；ODE-RNN使用38维递归状态及两层隐藏ODE；Latent ODE保留32维随机状态，编码状态调为48维。通过网络隐藏宽度精确匹配预算。旧的冻结二分类头服务于此前的其他任务，新版只保留当前混合字段预测所用网络。','',
               '## 2. 参数与训练进度','',
               '| 模型 | 容量 | 总参数 | 可训练参数 | 物理GPU | 阶段 | 已训练 / 最佳轮次 |','|---|---|---:|---:|---|---|---:|']
        if queue.get('status')=='paused_by_user':
            lines[2:2]=['同预算基线实验已按要求暂停，已有结果和恢复检查点保留，不会自动恢复。GPU已分配给MWM实验。','']
        elif read(output/'resume_request.json'):
            lines[2:2]=['当前主模型为latent2x both。此前暂停的同预算baseline从last.pt继续训练；已完成的实验保留原结果。MWM相关训练已暂停。','']
        for row in rows:
            h=row['hardware'];count=h.get('parameters');active=h.get('trainable_parameters')
            lines.append('| '+' | '.join([row['name'],row['capacity'],f'{count:,}' if count is not None else '—',f'{active:,}' if active is not None else '—',str(row['gpu']) if row['gpu'] is not None else '等待空闲卡',row['status']+(f" / {row['phase']}" if row['status']!='complete' else ''),f"{row['epochs']} / {row['best_epoch'] or '—'}"])+' |')
        lines+=['','## 3. 主要指标','',
                '| 模型 | 容量 | 验证历史 | 验证未来 | 验证整体 | 测试历史 | 测试未来 | 测试整体 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            values=[]
            for split in ['validation','test']:
                metrics=row['validation'] if split=='validation' else row['result']['test'] if row['result'] else None
                values += [f"{metrics[k]['reconstruction_error']:.6f}" if metrics else '—' for k in ['interpolation','forecast','overall']]
            lines.append('| '+' | '.join([row['name'],row['capacity']]+values)+' |')
        lines+=['','## 4. 调整容量后的变化','',
                '以下差值为同预算模型减去原容量模型；负值表示扩大或重新分配容量后误差下降。','',
                '| 模型 | 历史误差差值 | 未来误差差值 | 整体误差差值 |','|---|---:|---:|---:|']
        for kind in NAMES:
            native=next(r for r in rows if r['key']==kind+'_native');matched=next(r for r in rows if r['key']==kind)
            values=[f"{matched['result']['test'][k]['reconstruction_error']-native['result']['test'][k]['reconstruction_error']:.6f}" if matched['result'] else '—' for k in ['interpolation','forecast','overall']]
            lines.append('| '+' | '.join([NAMES[kind]]+values)+' |')
        if dataset=='mimic':
            lines+=['','## 5. 不同预测范围与数值标准化误差','',
                    '| 模型 | 容量 | 未来6小时 | 未来12小时 | 未来24小时 | 数值标准化MSE |','|---|---|---:|---:|---:|---:|']
            for row in rows:
                t=row['result']['test'] if row['result'] else None
                values=[f"{t[f'forecast_{h}h']['reconstruction_error']:.6f}" if t else '—' for h in [6,12,24]]
                values += [f"{t['mean_standardized_numeric_mse']:.6f}" if t else '—'];lines.append('| '+' | '.join([row['name'],row['capacity']]+values)+' |')
            clinical=rows[0]['result']['test']['clinical_forecast']
            for metric in ['mae','rmse']:
                with (out/f'per_field_{metric}.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['field','unit']+[r['key'] for r in rows])
                    for field,value in clinical.items():
                        if metric not in value:continue
                        writer.writerow([field,value['unit']]+[r['result']['test']['clinical_forecast'][field][metric] if r['result'] else '' for r in rows])
            lines+=['','[逐字段MAE](per_field_mae.csv) · [逐字段RMSE](per_field_rmse.csv)','']
        lines+=['','## 配置与执行记录','',
                f"同一数据缓存、患者划分和评测代码，数据SHA256为`{definition['dataset_sha256']}`。新增模型按各自原训练策略执行，Adam学习率0.003、原学习率衰减、输入dropout 0.1、300轮上限、50轮下限、40轮早停耐心。验证整体数据误差选模，最佳检查点重新载入后测试。",'',
                '新实验使用ode环境，独立单卡，接入SwanLab。队列轮询物理GPU 0、1、2、3，只在空闲卡上启动作业。既有实验继续运行；新结果分目录保存。','',
                f"[设计与宽度分配]({os.path.relpath(ROOT/'capacity_matched/README.zh-CN.md',out)}) · [运行队列]({os.path.relpath(output/'queue.json',out)}) · [全部配置与源码]({os.path.relpath(output/'manifest.json',out)}) · [结构化比较](comparison.json)",'']
        signature=[len(r['history']) for r in rows]
        if any(len(r['history']) for r in rows if r['capacity']=='同预算'):
            if signature!=read(out/'plot_signature.json'):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(1,2,figsize=(13,4.5))
                for row in rows:
                    if row['capacity']=='原容量':continue
                    for ax,region in zip(axes,['overall','forecast']):ax.plot([h['epoch'] for h in row['history']],[h['validation'][region]['reconstruction_error'] for h in row['history']],label=row['key'])
                for ax,title in zip(axes,['Validation overall','Validation forecast']):ax.set(xlabel='Epoch',ylabel='Mixed-field error',title=title);ax.legend();ax.grid(alpha=.15)
                fig.tight_layout()
                for ext in ['png','svg','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=160)
                plt.close(fig);atomic(out/'plot_signature.json',signature)
            lines+=['![训练与验证曲线](training_curves.png)','']
        (out/'summary.md').write_text('\n'.join(lines));records=[{k:v for k,v in r.items() if k!='history'} for r in rows]
        atomic(out/'comparison.json',{'dataset':dataset,'target_parameters':definition['target'],'models':records})
        all_rows.append({'dataset':dataset,'report':str(out/'summary.md'),'models':records})
    lines=['# ADNI与MIMIC：三种基线的同预算实验','',
           'ADNI按44,875参数匹配，MIMIC按43,280参数匹配。六组新实验按空闲GPU排队，所有任务均为单卡训练，使用ode环境和SwanLab。','',
           '| 作业 | 状态 | 阶段 | 物理GPU |','|---|---|---|---:|']
    if queue.get('status')=='paused_by_user':
        lines[2:2]=['未完成的同预算baseline已暂停，恢复检查点保留，不会自动恢复。GPU已分配给MWM实验。','']
    elif read(output/'resume_request.json'):
        lines[2:2]=['已恢复未完成的三组同参数量baseline，从原检查点续训。主模型使用latent2x both，MWM全部暂停。','']
    for job in manifest['jobs']:
        state=queue.get('jobs',{}).get(job['id'],{});lines.append('| '+' | '.join([job['id'],state.get('status','waiting'),state.get('phase','checks'),str(state.get('physical_gpu','等待空闲卡'))])+' |')
    lines+=['']+[f"- [{group['dataset'].upper()}：原容量与同预算指标]({os.path.relpath(group['report'],output)})" for group in all_rows]
    lines += ['', '[设计与参数分配](../../README.zh-CN.md) · [队列](queue.json) · [六组CUDA预检查](architecture_checks.json)','']
    (output/'summary.md').write_text('\n'.join(lines));atomic(output/'comparison.json',{'updated_at':datetime.now(timezone.utc).isoformat(),'datasets':all_rows})
