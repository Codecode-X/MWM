"""Compare 1x/2x/4x/8x/16x latent budgets with the latent4x pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .scaling_study import ROOT,atomic,identity,parameter_count
from .prior_study import read

DEFAULT=ROOT/'paper_odeae/results/shared_matched_latent_scaling'
FOUR=ROOT/'paper_odeae/results/shared_matched_scaling_4x'
BASE=ROOT/'paper_odeae/results/forward_age_scaling/shared_matched/seed_0'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);(out/'configs').mkdir(exist_ok=True)
    base=read(FOUR/'latent_4x/seed_0/config.json')
    assert base['latent']==73 and parameter_count(base)==90550
    jobs=[]
    for factor,gpu in [(2,0),(8,0),(16,3)]:
        target=22545*factor
        d=min(range(6,2049),key=lambda d:abs(parameter_count({**base,'latent':d})-target))
        name=f'latent_{factor}x'
        cfg={**base,'latent':d,'variant':name,'target_parameters':parameter_count({**base,'latent':d}),
             'output':str(out/name/'seed_0'),'ablation_family':'latent_scaling',
             'capacity_change':json.dumps({'latent':d},sort_keys=True),
             'ablation':f'latent4x continuation: target {factor}x original shared_matched total parameters; hidden widths fixed'}
        path=out/'configs'/f'{name}.json'
        if path.exists() and read(path)!=cfg:raise ValueError('config changed')
        atomic(path,cfg)
        jobs.append({'name':name,'factor':factor,'latent':d,'parameters':cfg['target_parameters'],
                     'physical_gpu':gpu,'config':str(path),'output':cfg['output']})
    references=[]
    for name,path in [('shared_matched',BASE),('latent_4x',FOUR/'latent_4x/seed_0')]:
        hw=read(path/'hardware.json');metrics=read(path/'metrics.json')
        assert hw['dataset_sha256']==digest(base['data']) and metrics['checkpoint_reload_difference']==0
        dest=out/'references'/name;dest.mkdir(parents=True,exist_ok=True)
        for file in ['config.json','hardware.json','metrics.json']:
            if (dest/file).exists() and (dest/file).read_bytes()!=(path/file).read_bytes():raise ValueError('reference changed')
            shutil.copy2(path/file,dest/file)
        references.append({'name':name,'source':str(path),'copied':str(dest),'metrics_sha256':digest(dest/'metrics.json')})
    snapshot=out/'source_snapshot';source=FOUR/'source_snapshot'
    hashes=read(source/'source_hashes.json')
    for name,expected in hashes.items():
        if digest(source/name)!=expected:raise ValueError('latent4x source snapshot changed')
    if not snapshot.exists():shutil.copytree(source,snapshot)
    for name,expected in hashes.items():
        if digest(snapshot/name)!=expected:raise ValueError('frozen source differs from latent4x')
    manifest={'jobs':jobs,'references':references,'seed':0,'dataset':base['data'],'dataset_sha256':digest(base['data']),
              'hidden_mlp_width':159,'shared_ode_width':188,'source_snapshot':'exact copy of latent4x training pipeline',
              'source_hashes':hashes,'reference_retrained':False,'selection':'original validation overall reconstruction',
              'execution':'independent single-GPU jobs; latent2x GPU0; completed latent8x GPU0 and latent16x GPU3 reused'}
    previous=read(out/'manifest.json')
    if previous is not None and previous!=manifest:
        stable=lambda m:{k:v for k,v in m.items() if k not in ['jobs','execution']}
        old_jobs={j['name']:j for j in previous['jobs']};new_jobs={j['name']:j for j in jobs}
        if stable(previous)!=stable(manifest) or set(new_jobs)-set(old_jobs)!={'latent_2x'} or any(new_jobs.get(k)!=v for k,v in old_jobs.items()):
            raise ValueError('manifest changed beyond adding latent2x')
        archive=out/'manifest.before_latent2x.json'
        if archive.exists() and read(archive)!=previous:raise ValueError('preserve previous manifest archive')
        atomic(archive,previous)
        for file in ['summary.md','summary.json','completion_verification.json']:
            source=out/file;dest=out/(file+'.before_latent2x')
            if source.exists() and not dest.exists():shutil.copy2(source,dest)
    atomic(out/'manifest.json',manifest)
    return manifest,snapshot


def row(path):
    result={'status':'queued'};history=read(path/'history.json',[])
    if history:
        best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error'])
        result.update(status='running',epochs=len(history),best_epoch=best['epoch'],validation=best['validation'],training_components=best['training_components'],
                      mean_epoch_seconds=sum(r['epoch_seconds'] for r in history)/len(history),peak_memory_bytes=max(r['peak_memory_bytes'] for r in history))
    metrics=read(path/'metrics.json')
    if metrics:result.update(status='complete',epochs=metrics['epochs_run'],best_epoch=metrics['best_epoch'],validation=metrics['validation'],test=metrics['test'])
    track=read(path/'swanlab.json',{})
    if result['status']=='queued' and track.get('status')=='running':result['status']='running'
    if (path/'failure.json').exists():result['status']='failed'
    return result


def report(out,manifest):
    rows=[]
    for reference in manifest['references']:
        path=Path(reference['copied']);cfg=read(path/'config.json')
        rows.append({'name':reference['name'],'latent':cfg['latent'],'parameters':cfg['target_parameters'],'physical_gpu':'已有结果',**row(path)})
    for job in manifest['jobs']:rows.append({**job,**row(Path(job['output']))})
    rows.sort(key=lambda r:r['parameters'])
    atomic(out/'summary.json',{'updated_at':datetime.now(timezone.utc).isoformat(),'queue':read(out/'queue.json',{}),'runs':rows})
    lines=['# shared_matched：潜在状态继续扩容','',
           '沿用 latent4x 的单卡训练方案，比较原始 shared_matched（22,545参数）的约2倍、4倍、8倍和16倍总参数预算。只增加潜在状态维度；INIT、UPDATE、DECODER隐藏宽度159，共享ODE隐藏宽度188保持固定。','',
           '| 模型 | 潜在维度 | 参数量 | GPU | 状态 | 轮数/最佳轮次 | 验证整体 | 验证预测 | 测试历史重构 | 测试预测 | 测试整体 |',
           '|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|']
    def metric(r,s,k):return f"{r[s][k]['reconstruction_error']:.5f}" if s in r else '—'
    for r in rows:
        lines.append(f"| {r['name']} | {r['latent']} | {r['parameters']:,} | {r['physical_gpu']} | {r['status']} | {r.get('epochs','—')}/{r.get('best_epoch','—')} | "+' | '.join(metric(r,s,k) for s,k in [('validation','overall'),('validation','forecast'),('test','interpolation'),('test','forecast'),('test','overall')])+' |')
    lines+=['','原基线、latent4x及已完成的latent8x/16x复用已有结果，补充latent2x在GPU 0训练。所有容量组使用与latent4x完全相同的冻结训练代码，seed 0、batch 32、原学习率和损失、最多300轮及耐心值40保持一致。新组不加入weak_prior或Jacobian损失。',
            '每组按原验证整体误差选择检查点；完整训练结束后填写测试成绩。参数量随状态维度增大而增加，相连网络的输入输出层同步扩大。','',
            '训练接入SwanLab。设计、参数预算和运行命令见 [实验说明](../../docs/LATENT_SCALE_UP.zh-CN.md)。','']
    if all(r['status']=='complete' for r in rows):
        lines+=['','全部完成，按验证整体误差排序：'+'、'.join(r['name'] for r in sorted(rows,key=lambda r:r['validation']['overall']['reconstruction_error']))+'。']
    temp=out/f'summary.md.{os.getpid()}.tmp';temp.write_text('\n'.join(lines));temp.replace(out/'summary.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(DEFAULT));p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('already running')
    manifest,snapshot=prepare(out);report(out,manifest)
    if args.prepare_only:return
    active_gpus=sorted({j['physical_gpu'] for j in manifest['jobs'] if not (Path(j['output'])/'metrics.json').exists()})
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','physical_gpus':active_gpus};atomic(out/'queue.json',state)
    def worker(job):
        dest=Path(job['output']);dest.mkdir(parents=True,exist_ok=True)
        try:
            if digest(manifest['dataset'])!=manifest['dataset_sha256']:raise ValueError('dataset changed')
            if (dest/'metrics.json').exists():
                if read(dest/'config.json')!=read(Path(job['config'])):raise ValueError('completed config differs')
                return
            if (dest/'history.json').exists():raise ValueError('preserve interrupted run; choose a new output')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['physical_gpu']),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
            for smoke in [True,False]:
                target=out/'checks'/job['name'] if smoke else dest;target.mkdir(parents=True,exist_ok=True)
                if smoke and (target/'metrics.json').exists():continue
                command=[sys.executable,'-u','-m','paper_odeae.train','--config',job['config'],'--output',str(target)]
                if smoke:command.append('--smoke')
                print(f"GPU{job['physical_gpu']}: {job['name']}, smoke={smoke}",flush=True)
                with (target/'train.log').open('w') as log:
                    subprocess.run(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            (dest/'failure.json').unlink(missing_ok=True)
        except Exception as error:
            atomic(dest/'failure.json',{'error':f'{type(error).__name__}: {error}','job':job});raise
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(worker,job) for job in manifest['jobs']}
            while pending:
                done,pending=wait(pending,timeout=30)
                for future in done:future.result()
                report(out,manifest)
        verification=[]
        for job in manifest['jobs']:
            hw=read(Path(job['output'])/'hardware.json');metrics=read(Path(job['output'])/'metrics.json')
            assert hw['trainable_parameters']==job['parameters'] and hw['physical_gpu']==str(job['physical_gpu'])
            assert hw['dataset_sha256']==manifest['dataset_sha256'] and hw['parameter_device']=='cuda:0'
            assert metrics['checkpoint_reload_difference']<=1e-6
            verification.append({'name':job['name'],'hardware':hw,'reload_difference':metrics['checkpoint_reload_difference']})
        atomic(out/'completion_verification.json',{'runs':verification,'references_retrained':False})
        state['status']='complete';atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);report(out,manifest);raise


if __name__=='__main__':main()
