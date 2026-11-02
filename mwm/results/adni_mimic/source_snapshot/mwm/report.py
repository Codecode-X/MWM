"""Readable progress tables for MWM and its two supervision ablations."""
import json
from pathlib import Path


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def report(out,manifest):
    queue=read(out/'queue.json',{});all_rows=[];resources=read(out/'resource_allocation.json',{})
    for dataset in ['adni','mimic']:
        jobs=[j for j in manifest['jobs'] if j['dataset']==dataset];folder=Path(jobs[0]['config']['comparison_output'])
        lines=[f'# {dataset.upper()}：MWM训练与比较','',
               '## 1. 本轮怎样训练','',
               '三组均按检查时间顺序读取历史。首次观测年龄用于INIT，首次观测经UPDATE形成初始估计；此后每次先用ODE预测当前检查，再吸收当前检查。历史截止点之后只做ODE预测。',
               '',f"每组使用单张RTX 4090、seed 0、batch {jobs[0]['config']['batch_size']}，每个batch执行一次梯度更新，无梯度累积。最多300轮、至少50轮，验证分数连续40轮未改善时早停。全部训练接入SwanLab。",'',
               '三组统一用验证集未加权的逐次预测MSE＋交叉熵选择检查点。A的计数权重和B的NLL只作用于训练，三组的选择标准相同。',
               '', '## 2. 损失和参数量','',
               '| 版本 | 数据监督 | INIT / UPDATE / Decoder / ODE隐藏宽度 | 参数量 |',
               '|---|---|---|---:|']
        if queue.get('active_datasets')==['adni']:
            lines[2:2]=[('本轮GPU分配：GPU 0运行MWM主版本，GPU 1运行A，GPU 3运行B；GPU 2保留备用。' if dataset=='adni'
                         else 'MIMIC的MWM实验已暂停，检查点保留；当前GPU 0、1、3用于ADNI，GPU 2备用。'),'']
        elif resources.get('assignment',{}).get('2')=='mimic_main':
            lines[2:2]=['GPU 0、1、3分别训练ADNI的主版本、A、B；GPU 2恢复MIMIC主版本。两数据集训练batch均为128，每个batch一次反向传播和参数更新，无梯度累积。','']
        for j in jobs:
            variant=j['config']['variant'];description={'main':'数值MSE＋类别交叉熵；全部观测同权','A':'在主版本上，乘sigmoid(N−3)；N为已吸收的观测条数','B':'数值Gaussian NLL＋类别交叉熵；输出每时刻一个σ'}[variant]
            lines.append(f"| MWM {variant} | {description} | {'159 / 160 / 158 / 187' if variant=='B' else '159 / 159 / 159 / 188'} | {j['config']['target_parameters']:,} |")
        lines+=['','三组均保留0.01 Jacobian和0.02共享ODE参数L2，移除旧的0.01 weak_prior独立项。B用少量隐藏宽度调整容纳29个σ头参数，总参数量与主版本完全相同。B的σ使用softplus＋0.01，初始值为1；log σ按每个有效数值字段计入NLL。','',
                '## 3. 当前进度','', '| 版本 | 状态 | 物理GPU | 已训练 / 最佳轮次 | 验证逐次预测整体误差 |', '|---|---|---:|---:|---:|']
        for j in jobs:
            cfg=j['config'];run=Path(cfg['output']);metrics=read(run/'metrics.json');history=read(run/'history.json',[])
            state=queue.get('jobs',{}).get(j['id'],{});hardware=read(run/'hardware.json',{})
            best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error']) if history else None
            row={'id':j['id'],'config':cfg,'status':metrics['status'] if metrics else state.get('status','waiting'),
                 'phase':state.get('phase'),'hardware':hardware,'epochs':len(history),'best_epoch':metrics['best_epoch'] if metrics else best['epoch'] if best else None,
                 'metrics':metrics,'validation_best_causal':best['validation']['overall']['reconstruction_error'] if best else None}
            all_rows.append(row)
            lines.append('| '+' | '.join([f"MWM {cfg['variant']}",row['status']+' / '+str(row['phase']),str(hardware.get('physical_gpu',state.get('physical_gpu','—'))),
                                           f"{len(history)} / {row['best_epoch'] or '—'}",f"{row['validation_best_causal']:.6f}" if best else '—'])+' |')
        lines+=['','## 4. 共同测试任务','',
                '测试历史重建从历史截止点向过去自由积分；测试未来预测从同一截止点向未来自由积分。两者均不吸收测试目标。逐次预测历史误差另外记录，包含首次UPDATE后的读出和其余历史检查UPDATE前的预测。', '',
                '| 模型 | 参数量 | 最佳轮次 | 测试历史回顾 | 测试未来 | 测试整体 | 测试历史逐次预测 |','|---|---:|---:|---:|---:|---:|---:|']
        ref=Path(jobs[0]['config']['reference_output']);refm=read(ref/'metrics.json');refh=read(ref/'hardware.json',{})
        if refm:
            t=refm['test'];lines.append('| '+' | '.join(['latent2x both参考',str(refh.get('parameters','—')),str(refm['best_epoch'])]+[f"{t[k]['reconstruction_error']:.6f}" for k in ['interpolation','forecast','overall']]+['—'])+' |')
        for row in [r for r in all_rows if r['config']['dataset_name']==dataset]:
            metrics=row['metrics'];test=metrics['test'] if metrics else None
            values=[f"MWM {row['config']['variant']}",str(row['config']['target_parameters']),str(row['best_epoch'] or '—')]
            values += [f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation','forecast','overall']]
            values += [f"{metrics['causal_test']['interpolation']['reconstruction_error']:.6f}" if metrics else '—']
            lines.append('| '+' | '.join(values)+' |')
        lines+=['','参考组按旧的历史回顾＋未来整体误差选检查点；MWM按新的逐次预测整体误差选择。表格共用数据划分、归一化和测试目标；参考组与MWM之间同时包含初始化及监督方式的变化。', '',
                '## 5. 数据和运行记录','',
                f"数据SHA256：`{jobs[0]['config']['dataset_sha256']}`。",'',
                '每组保存config.json、hardware.json、history.json、best.pt、last.pt、metrics.json和swanlab.json。hardware记录实际张量设备和物理GPU。', '',
                '[方法与损失说明](../../../MWM_METHOD.zh-CN.md)' if dataset=='adni' else '[方法与损失说明](../../../paper_odeae/MWM_METHOD.zh-CN.md)', '',
                '[全局队列](../../../mwm/results/adni_mimic/queue.json)','']
        if dataset=='adni' and resources.get('archived_adni_batch32'):
            import os
            lines+=['旧batch32结果已归档，本轮三组从seed 0重新开始，以batch128完成整次训练。', '',
                    f"[旧batch32结果]({os.path.relpath(Path(resources['archived_adni_batch32'])/'summary.md',folder)})",'']
        # Relative links are resolved from the actual dataset result directory.
        import os
        lines=[line.replace('../../../MWM_METHOD.zh-CN.md',os.path.relpath(Path(manifest['project_root'])/'paper_odeae/MWM_METHOD.zh-CN.md',folder))
                    .replace('../../../paper_odeae/MWM_METHOD.zh-CN.md',os.path.relpath(Path(manifest['project_root'])/'paper_odeae/MWM_METHOD.zh-CN.md',folder))
                    .replace('../../../mwm/results/adni_mimic/queue.json',os.path.relpath(out/'queue.json',folder)) for line in lines]
        (folder/'summary.md').write_text('\n'.join(lines))
    (out/'comparison.json').write_text(json.dumps({'models':all_rows},indent=2))
    (out/'summary.md').write_text('# MWM：ADNI与MIMIC实验\n\n每个版本独占一张空闲GPU，主版本与A、B共六组。\n\n'
        + ('当前只运行ADNI主版本、A、B，使用GPU 0、1、3；GPU 2备用。MIMIC暂停。\n\n' if queue.get('active_datasets')==['adni'] else '')
        + ('当前GPU 0/1/3运行ADNI主版本/A/B，GPU 2运行MIMIC主版本；均为batch128、无梯度累积。MIMIC A/B保持暂停。\n\n' if resources.get('assignment',{}).get('2')=='mimic_main' else '')
        + '\n'.join(f"- [{d.upper()}结果]({__import__('os').path.relpath(Path(next(j['config']['comparison_output'] for j in manifest['jobs'] if j['dataset']==d))/'summary.md',out)})" for d in ['adni','mimic'])
        +'\n\n[设备与状态](queue.json) · [CUDA检查](architecture_checks.json) · [固定配置](manifest.json)\n')
