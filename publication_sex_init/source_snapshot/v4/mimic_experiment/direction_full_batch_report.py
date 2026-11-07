"""CPU reporting of the new single-GPU, no-accumulation direction comparison."""
from pathlib import Path

from .report import read
from .prepare import atomic


def generate(out):
    out=Path(out);queue=read(out/'queue.json',{});manifest=read(out/'manifest.json',{});root=Path(__file__).resolve().parents[1]
    rows=[]
    for key,title,path in [('forward_shared','latent2x both（正向ODE，单卡）',out/'forward_shared/seed_0'),
                           ('reverse_shared','反向共享ODE（单卡）',out/'reverse_shared/seed_0'),
                           ('historical_ddp','历史latent2x both（四卡DDP）',root/'mimic_experiment/results/latent2x_both_cuda/seed_0')]:
        history=read(path/'history.json',[]);final=read(path/'metrics.json');best=min(history,key=lambda x:x['validation']['overall']['reconstruction_error']) if history else None
        checkpoint_path=path/'checkpoint_evaluation_epoch_128/metrics.json'
        checkpoint_result=read(checkpoint_path) if key=='reverse_shared' and not final else None
        if checkpoint_result and (checkpoint_result.get('status')!='checkpoint_evaluated_while_paused' or checkpoint_result.get('best_epoch')!=128):
            raise RuntimeError('Unexpected paused checkpoint evaluation')
        result=final or checkpoint_result
        job=queue.get('jobs',{}).get(key,{})
        rows.append({'key':key,'name':title,'status':'complete' if final else job.get('status','waiting'),'phase':job.get('phase','waiting'),
                     'epochs':final['epochs_run'] if final else len(history),'best_epoch':result['best_epoch'] if result else best['epoch'] if best else None,
                     'validation':result['validation'] if result else best['validation'] if best else None,'result':result,
                     'evaluation_source':str(checkpoint_path) if checkpoint_result else str(path/'metrics.json') if final else None,
                     'evaluation_kind':'paused_epoch128_checkpoint' if checkpoint_result else 'completed_run' if final else None,
                     'hardware':read(path/'hardware.json',{}),'history':history,'path':str(path)})
    lines=['# MIMIC：正向与反向共享ODE的单卡整批训练比较','',
           '## 1. 本轮怎样公平比较','',
           '两组均从seed 0重新初始化，使用相同数据、43,280参数、28维状态和训练代码。每个优化步骤一次处理完整128人batch，执行一次前向、一次反向传播和一次参数更新，无梯度累积。正向组使用GPU 3，反向组使用GPU 2，均为ode环境。','',
           '两组的数据顺序、历史截止点选择、输入dropout、学习率、优化器、损失项权重与早停设置相同。主历史读取顺序、生成起点及weak_prior所需的计算路径在下文逐项说明。两组保留0.01 weak prior、0.01 Jacobian和0.02共享ODE参数L2。','',
           '训练最多300轮、至少50轮，验证整体数据误差连续40轮未改善时早停。主要测试读取前24小时，预测到48小时；验证与测试保留原共同评价器的固定查询批次。全部训练接入SwanLab。','',
           '旧注意力组和旧反向组已按用户要求停止，旧检查点与日志保留。历史四卡结果作为背景列出；本轮方向收益只使用两组新单卡结果计算。','',
           ]
    lines+=Path(__file__).with_name('direction_full_batch_method.zh-CN.md').read_text().splitlines()
    lines+=['',
           '## 7. 训练进度与主要结果','',
           '| 模型 | 状态 | 已训练 / 最佳轮次 | 验证整体 | 测试历史 | 测试未来 | 测试整体 | 数值标准化MSE |','|---|---|---:|---:|---:|---:|---:|---:|']
    if queue.get('status')=='paused_by_user':
        lines[2:2]=['反向共享ODE保持暂停，最佳与恢复检查点保留，不会自动恢复。','']
    for row in rows:
        val=row['validation'];test=row['result']['test'] if row['result'] else None
        status=row['status']+(f" / {row['phase']}" if row['status']!='complete' else '')
        if row['evaluation_kind']=='paused_epoch128_checkpoint':status+='（第128轮已测试）'
        values=[row['name'],status,f"{row['epochs']} / {row['best_epoch'] or '—'}",f"{val['overall']['reconstruction_error']:.6f}" if val else '—']
        values += [f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation','forecast','overall']]
        values += [f"{test['mean_standardized_numeric_mse']:.6f}" if test else '—'];lines.append('| '+' | '.join(values)+' |')
    if rows[1]['evaluation_kind']=='paused_epoch128_checkpoint':
        lines+=['','反向组已训练134轮，本表暂用暂停时验证最佳的第128轮检查点。该检查点在GPU 3上完成验证和测试，沿用前24小时历史、预测至48小时，以及共同评价器的固定批次；未执行任何参数更新。','',
                '[第128轮完整指标](reverse_shared/seed_0/checkpoint_evaluation_epoch_128/metrics.json) · [评估设备与检查点记录](reverse_shared/seed_0/checkpoint_evaluation_epoch_128/hardware.json)','']
    lines+=['','## 8. 本轮方向收益','',
            '| 比较 | 历史误差差值 | 未来误差差值 | 整体误差差值 | 正向未来相对误差降低 |','|---|---:|---:|---:|---:|']
    if rows[0]['result'] and rows[1]['result']:
        forward=rows[0]['result']['test'];reverse=rows[1]['result']['test']
        differences=[reverse[k]['reconstruction_error']-forward[k]['reconstruction_error'] for k in ['interpolation','forecast','overall']]
        values=[f'{v:.6f}' for v in differences]+[f"{100*differences[1]/reverse['forecast']['reconstruction_error']:.2f}%"]
    else:values=['—']*4
    label='反向第128轮−正向最佳（新单卡）' if rows[1]['evaluation_kind']=='paused_epoch128_checkpoint' else '反向−正向（新单卡）'
    lines.append('| '+' | '.join([label]+values)+' |')
    lines+=['','## 9. 原始训练与设备记录','',
            '| 模型 | 实际物理GPU | 参数量 | 训练batch | 梯度累积次数 | 最大训练显存GiB |','|---|---|---:|---:|---:|---:|']
    for row in rows:
        h=row['hardware'];peak=max((x.get('peak_memory_bytes',0) for x in row['history']),default=0)
        lines.append('| '+' | '.join([row['name'],str(h.get('physical_gpu','—')),str(h.get('parameters','—')),str(h.get('training_batch_size',h.get('global_batch_size','—'))),str(h.get('gradient_accumulation_steps','—')),f'{peak/2**30:.2f}' if peak else '—'])+' |')
    lines+=['','[运行队列](queue.json) · [固定配置与来源](manifest.json) · [完整指标](comparison.json) · [设计说明](../../DIRECTION_FULL_BATCH.zh-CN.md)','',
            '每组目录保存best.pt、last.pt、history.json、metrics.json、hardware.json和swanlab.json。训练结束重新载入最佳检查点，核对验证分数并完成测试；同时核对评测患者数和有效观测数。','',
            '```bash','bash scripts/run_mimic_direction_full_batch.sh','```','']
    if (root/'mimic_experiment/results/weak_prior_ablation_cuda/manifest.json').exists():
        lines+=['[新增weak_prior=0消融：第40轮前后稳定性比较](../weak_prior_ablation_cuda/summary.md)','']
    if (root/'mimic_experiment/results/init_covariate_ablation_cuda/manifest.json').exists():
        lines+=['[标准正向模型INIT输入消融：首次观测年龄、性别](../init_covariate_ablation_cuda/summary.md)','']
    signature=[len(r['history']) for r in rows]
    if any(signature[:2]):
        if signature!=read(out/'plot_signature.json'):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for row in rows:
                for ax,region in zip(axes,['overall','forecast']):ax.plot([h['epoch'] for h in row['history']],[h['validation'][region]['reconstruction_error'] for h in row['history']],label=row['key'])
            for ax,title in zip(axes,['Validation overall','Validation forecast']):ax.set(xlabel='Epoch',ylabel='Mixed-field error',title=title);ax.legend();ax.grid(alpha=.15)
            fig.tight_layout()
            for ext in ['png','svg','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=160)
            plt.close(fig);atomic(out/'plot_signature.json',signature)
        lines+=['![训练曲线](training_curves.png)','']
    diagnostics=out/'gradient_sensitivity_analysis/summary_section.zh-CN.md'
    if diagnostics.exists():lines+=['',diagnostics.read_text()]
    (out/'summary.md').write_text('\n'.join(lines));atomic(out/'comparison.json',{'models':[{k:v for k,v in r.items() if k!='history'} for r in rows]})
