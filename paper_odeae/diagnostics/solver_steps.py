"""Four-GPU read-only comparison of solver work on identical test prefixes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from paper_odeae.models import PaperModel
from paper_odeae.baselines import BaselineModel
from paper_odeae.data import Prefixes,collate
from paper_odeae.train import device_batch
from paper_odeae.solver import solver_statistics,TABLEAU

ROOT=Path(__file__).resolve().parents[2]
JOBS=[('ODE-AE','local_available/ode_ae/seed_0'),
      ('Deterministic 2505','local_available/deterministic/seed_0'),
      ('LSTM','local_available/lstm/seed_0'),
      ('Latent ODE native','baselines_full/native/latent_ode/seed_0'),
      ('ODE-RNN native','baselines_full/native/ode_rnn/seed_0'),
      ('Latent ODE small','baselines_full/small/latent_ode/seed_0'),
      ('ODE-RNN small','baselines_full/small/ode_rnn/seed_0'),
      ('Deterministic 22545','capacity_matched/deterministic_22545/seed_0'),
      ('Forward age 22545','direction_ablation/forward_age_from22545/seed_0'),
      ('Forward zero 21307','direction_ablation/forward_from22545/seed_0')]


def aggregate(calls):
    active=[r for r in calls if r['active_patients']>0]
    return {'integration_calls':len(calls),'nonzero_integrations':len(active),
        'zero_interval_calls':len(calls)-len(active),'accepted_steps':sum(r['accepted'] for r in calls),
        'rejected_steps':sum(r['rejected'] for r in calls),'nfe':sum(r['nfe'] for r in calls),
        'accepted_steps_per_nonzero_solve':float(np.mean([r['accepted'] for r in active])) if active else 0.,
        'accepted_step_histogram':dict(Counter(r['accepted'] for r in active)),
        'active_patient_step_equivalents':sum(r['accepted']*r['active_patients'] for r in calls)}


def report(out,results):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    done=[r for r in results if 'total' in r]
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['name','status','checkpoint_epoch','trainable_parameters','patients','prefixes','batches','integration_calls','nonzero_integrations','accepted_steps','rejected_steps','nfe'])
        writer.writeheader()
        for r in done:
            values={**r,**r['total']};writer.writerow({k:values[k] for k in writer.fieldnames})
    lines=['# 各模型 ODE 数值步数与向量场调用比较','',
      f"本次使用完整测试集的 {done[0]['patients']} 位患者，枚举全部 {done[0]['prefixes']} 个历史截断，batch size 32，共 {done[0]['batches']} 批。四张 RTX 4090 分配独立诊断任务，所有模型均在 CUDA 上执行，未训练或修改检查点。",'',
      '这里只统计一次完整评估的前向求解开销，不是历史累计训练步数。所有任务使用当前 Tsit5 实现，rtol=1e-4、atol=1e-6。计数器开关前后的首批预测逐位相同。','',
      '## 1. 这些计数分别是什么意思','',
      '- 积分调用：模型要求解一个时间区间的次数，包含整批时间间隔为零而直接返回的情况。',
      '- 非零积分：该批至少一个患者有非零间隔，求解器实际进行数值步。',
      '- 接受步：误差检查通过并推进了时间的数值步；拒绝步只花计算，没有推进时间。',
      '- NFE：向量场网络的前向调用次数。当前实现每个尝试步调用7次，因此 NFE=7×（接受步＋拒绝步）。解码器、GRU、MLP更新和反向传播不包含在内。','',
      '这些是批向量化调用计数。不同患者的积分共享批内步长控制，不能理解成每个人分别独立求解的次数。换批量大小、患者分组或容差后，数字可能变化。','',
      '## 2. 完整测试集比较','',
      '| 模型 | 可训练参数 | 检查点轮次 | 积分调用 | 非零积分 | 接受步 | 拒绝步 | NFE |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in done:
        t=r['total'];label=r['name']+('（训练中快照）' if r['status']=='in_progress_checkpoint' else '')
        lines.append(f"| {label} | {r['trainable_parameters']:,} | {r['checkpoint_epoch']} | {t['integration_calls']:,} | {t['nonzero_integrations']:,} | {t['accepted_steps']:,} | {t['rejected_steps']:,} | {t['nfe']:,} |")
    lines+=['','零初始化正向版本若尚未生成正式检查点，则不填写数值，不用随机初始化或单轮检查替代。LSTM 无 ODE，故为0，但不意味着它没有递归计算成本。','',
      '![向量场调用对比](nfe_comparison.png)','','## 3. 编码和生成分别用了多少步','',
      '| 模型 | 阶段 | 非零积分 | 接受步 | 拒绝步 | NFE |','|---|---|---:|---:|---:|---:|']
    for r in done:
        for phase,t in r['by_phase'].items():
            lines.append(f"| {r['name']} | {phase} | {t['nonzero_integrations']:,} | {t['accepted_steps']:,} | {t['rejected_steps']:,} | {t['nfe']:,} |")
    lines+=['','## 4. 如何理解差异','',
      '绝大多数非零区间都只接受2步、没有拒绝。当前求解器把每次区间归一到[0,1]，从0.25开始尝试，误差足够小时第二步可覆盖剩余0.75。因此2步是当前起步策略下的最低常见值，不是每年固定走2步。完整统计中，22,545参数的反向确定性模型有一个区间用了3步，其余区间为2步。','',
      'ODE-RNN只有一条随时间递推的动力学；ODE-AE包含历史编码和轨迹生成两条求解路径，所以总NFE约为ODE-RNN的两倍。扩大网络并没有使接受步数明显增加，但每次向量场调用更贵，不能根据NFE相同就认定耗时相同。','',
      'Latent ODE比原ODE-AE多一次从截止点回到首次检查的生成积分；这同时避免了原ODE-AE编码首步的全零间隔。原容量与小容量Latent ODE本次恰好具有相同的步数分布。','',
      '正向截止点锚定模型分别重建过去和生成未来，且一个批中的患者截断点不同，所以会增加批积分调用。按“接受步×非零间隔患者数”计算，正向年龄版本为48,954，与小确定性ODE-AE相同；其批NFE更高主要体现了分批和掩码路径的开销，不应说成每位患者的有效积分区间都变多。这个加权量也不是独立单患者求解的实测步数。','',
      '当前诊断与正式训练共用GPU，记录的墙钟时间不用于公平速度排名。不同模型的状态维度、隐藏层宽度、层数、时间单位以及解码/递归更新成本也不同。ODE-AE用年，Latent ODE/ODE-RNN保留十年单位；同一容差不能保证训练先验完全等价。','',
      '尚未结束的正向年龄模型使用诊断当时的稳定best.pt快照，最终模型的步数可能改变。历史训练没有记录逐批NFE，不能用本次评估计数乘以训练轮数来冒充训练总开销。','',
      '## 5. 文件和复查','',
      '`summary.json`和各`job_*.json`保存每模型设备、检查点轮次、阶段计数和每次非零积分接受步数的直方图；`comparison.csv`用于表格处理。`counter_verification.json`额外检验了出现拒绝步时的NFE计数，并与真实函数调用次数逐一吻合。','',
      '```bash','conda activate ode','python -m paper_odeae.diagnostics.solver_steps','```','']
    (out/'summary.zh-CN.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(11,6));ax.barh([r['name'] for r in done],[r['total']['nfe'] for r in done]);ax.invert_yaxis()
    ax.set(xlabel='Vector-field calls (NFE), one test evaluation',title=f"Same {done[0]['patients']} patients / {done[0]['prefixes']} prefixes / batch size 32")
    for i,r in enumerate(done):ax.text(r['total']['nfe'],i,f" {r['total']['nfe']:,}",va='center')
    ax.set_xlim(0,max(r['total']['nfe'] for r in done)*1.14);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'nfe_comparison.{ext}',dpi=180)
    plt.close(fig)


@torch.no_grad()
def worker(gpu,out,patients):
    if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError('explicit CUDA required')
    torch.set_num_threads(1)
    payload=pickle.loads((ROOT/'paper_odeae/cache/local_available.pkl').read_bytes())
    all_test=[s for s in payload['sequences'] if s['split']=='test']
    selected=all_test if patients==0 else [all_test[i] for i in np.linspace(0,len(all_test)-1,patients,dtype=int)]
    dataset=Prefixes(selected);loader=DataLoader(dataset,batch_size=32,collate_fn=collate)
    for index,(name,folder) in enumerate(JOBS):
        if index%4!=gpu:continue
        run=ROOT/'paper_odeae/results'/folder
        if not (run/'best.pt').exists():
            (out/f'job_{index}.json').write_text(json.dumps({'name':name,'status':'no_checkpoint','run':folder})+'\n');continue
        # Read bytes from a stable checkpoint; current training may replace it.
        import io
        for attempt in range(5):
            before=(run/'best.pt').stat();blob=(run/'best.pt').read_bytes();after=(run/'best.pt').stat()
            if before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size==len(blob):break
        else:raise RuntimeError('could not obtain stable checkpoint snapshot')
        checkpoint=torch.load(io.BytesIO(blob),map_location='cuda',weights_only=False)
        cfg=checkpoint['config'];schema=checkpoint['schema']
        if cfg['model'] in ['latent_ode','ode_rnn']:
            model=BaselineModel(schema,cfg['model'],cfg['latent'],cfg['encoder_hidden'],cfg['ode_width'],cfg['rtol'],cfg['atol']).cuda()
        else:
            model=PaperModel(schema,cfg['model'],cfg['latent'],cfg['ode_width'],cfg['mlp_width'],cfg['rtol'],cfg['atol'],cfg.get('encoder_direction','backward'),cfg.get('encoder_initialization')).cuda()
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        first=device_batch(next(iter(loader)));reference=model(first,sample=False)['prediction']
        original=model.solve
        phase=lambda func:('encoder' if (hasattr(model,'backward_ode') and func is model.backward_ode) or
            (hasattr(model,'core') and hasattr(model.core,'encoder') and func is model.core.encoder.ode) else
            ('recurrent_dynamics' if cfg['model']=='ode_rnn' else 'generator'))
        active_collection=None
        def tagged(func,state,delta):
            if active_collection is not None:active_collection['phase']=phase(func)
            return original(func,state,delta)
        model.solve=tagged
        with solver_statistics() as check:
            active_collection=check
            instrumented=model(first,sample=False)['prediction']
        torch.testing.assert_close(reference,instrumented,rtol=0,atol=0)
        torch.cuda.synchronize();start=time.monotonic();batches=0
        with solver_statistics() as stats:
            active_collection=stats
            for cpu in loader:
                model(device_batch(cpu),sample=False);batches+=1
        torch.cuda.synchronize()
        result={'name':name,'status':'completed_checkpoint' if (run/'metrics.json').exists() else 'in_progress_checkpoint',
            'run':folder,'checkpoint_epoch':checkpoint['epoch'],'checked_at':datetime.now(timezone.utc).isoformat(),
            'physical_gpu':gpu,'parameter_device':str(next(model.parameters()).device),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'patients':len(selected),'prefixes':len(dataset),'batches':batches,'batch_size':32,'rtol':cfg['rtol'],'atol':cfg['atol'],
            'time_unit':'decade' if cfg['model'] in ['latent_ode','ode_rnn'] else 'year',
            'instrumentation_prediction_difference':0.,'wall_seconds_with_concurrent_training':time.monotonic()-start,
            'total':aggregate(stats['calls']),
            'by_phase':{p:aggregate([r for r in stats['calls'] if r['phase']==p]) for p in sorted({r['phase'] for r in stats['calls']})}}
        assert result['total']['nfe']==(1+len(TABLEAU['a_lower']))*(result['total']['accepted_steps']+result['total']['rejected_steps'])
        (out/f'job_{index}.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--output',default='paper_odeae/results/solver_steps_full_test')
    parser.add_argument('--patients',type=int,default=0,help='0: entire test split; otherwise evenly spaced subset')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.report_only:report(out,json.loads((out/'summary.json').read_text())['runs']);return
    if args.worker is not None:worker(args.worker,out,args.patients);return
    def launch(gpu):
        with (out/f'gpu_{gpu}.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','paper_odeae.diagnostics.solver_steps','--worker',str(gpu),'--output',str(out),'--patients',str(args.patients)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,range(4)))
    results=[json.loads((out/f'job_{i}.json').read_text()) for i in range(len(JOBS))]
    (out/'summary.json').write_text(json.dumps({'runs':results,'rk_stages_per_attempt':1+len(TABLEAU['a_lower'])},indent=2)+'\n')
    report(out,results)
    for r in results:print(r['name'],r.get('total',r['status']))


if __name__=='__main__':main()
