"""Three single-GPU loss ablations on latent2x, with the live baseline reused."""
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

from .scaling_study import ROOT,atomic,identity
from .prior_study import read

DEFAULT=ROOT/'paper_odeae/results/latent2x_loss_ablation'
BASE=ROOT/'paper_odeae/results/shared_matched_latent_scaling/latent_2x/seed_0'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);(out/'configs').mkdir(exist_ok=True)
    base=read(BASE/'config.json');assert base['latent']==28 and base['target_parameters']==44875
    jobs=[]
    for name,prior,jac,gpu in [('both',.01,.01,1),('weak_prior',.01,0.,2),('jacobian',0.,.01,3)]:
        cfg={**base,'study':'latent2x_loss_ablation','variant':name,'prior_weight':prior,'jacobian_weight':jac,
             'collect_prior_predictions':True,'collect_jacobian_states':True,
             'output':str(out/name/'seed_0'),'ablation_family':'latent2x_losses',
             'ablation':'latent2x fixed architecture, pre-update supervision and exact Jacobian independent/combined',
             'capacity_change':'none'}
        path=out/'configs'/f'{name}.json'
        if path.exists() and read(path)!=cfg:raise ValueError('config changed')
        atomic(path,cfg);jobs.append({'name':name,'prior_weight':prior,'jacobian_weight':jac,'physical_gpu':gpu,'config':str(path),'output':cfg['output']})
    baseline_config=out/'baseline_config.json'
    if baseline_config.exists() and read(baseline_config)!=base:raise ValueError('baseline config changed')
    atomic(baseline_config,base)
    manifest={'jobs':jobs,'baseline':str(BASE),'baseline_config_sha256':digest(BASE/'config.json'),'parameters':44875,'latent':28,
              'baseline_retrained':False,'dataset':base['data'],'dataset_sha256':digest(base['data']),
              'selection':'original validation overall reconstruction; added losses excluded from selection score',
              'execution':'three independent single-GPU jobs: both=1, prior=2, jacobian=3; original baseline continues on GPU0'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise ValueError('manifest changed')
    atomic(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot'
    files=['paper_odeae/__init__.py','paper_odeae/models.py','paper_odeae/train.py','paper_odeae/jacobian.py',
           'paper_odeae/data.py','paper_odeae/solver.py','paper_odeae/diagnostics/jacobian_checkpoint.py',
           'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json',
           'src/__init__.py','src/tracking.py','src/swanlab_local.py']
    if not (snapshot/'source_hashes.json').exists():
        for name in files:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        atomic(snapshot/'source_hashes.json',{name:digest(snapshot/name) for name in files})
    return manifest,snapshot


def row(path):
    result={'status':'queued'};history=read(path/'history.json',[])
    if history:
        best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error'])
        result.update(status='running',epochs=len(history),best_epoch=best['epoch'],validation=best['validation'],training_components=best.get('training_components',{}),
                      mean_epoch_seconds=sum(r.get('epoch_seconds',0) for r in history)/len(history),peak_memory_bytes=max(r.get('peak_memory_bytes',0) for r in history))
    metrics=read(path/'metrics.json')
    if metrics:result.update(status='complete',epochs=metrics['epochs_run'],best_epoch=metrics['best_epoch'],validation=metrics['validation'],test=metrics['test'])
    track=read(path/'swanlab.json',{})
    if result['status']=='queued' and track.get('status')=='running':result['status']='running'
    if (path/'failure.json').exists():result['status']='failed'
    return result


def report(out,manifest):
    rows=[{'name':'latent2x baseline','prior_weight':0.,'jacobian_weight':0.,'physical_gpu':0,**row(BASE)}]
    diagnostic=read(out/'baseline_diagnostics.json')
    if diagnostic:
        for split in ['validation','test']:
            rows[0][split]={**rows[0][split],'prior':diagnostic[split]['prior'],'dynamics':diagnostic[split]['dynamics']}
    rows += [{**job,**row(Path(job['output']))} for job in manifest['jobs']]
    atomic(out/'summary.json',{'updated_at':datetime.now(timezone.utc).isoformat(),'queue':read(out/'queue.json',{}),'runs':rows})
    lines=['# latent2x：weak_prior与Jacobian独立及联合消融','',
           '四组均为28维状态、44,875参数。基线使用已有latent2x训练；三组从相同seed 0初始化从头训练。隐藏宽度、共享向量场、数据、batch 32、学习率和早停设置保持一致。','',
           '| 模型 | weak_prior权重 | Jacobian权重 | GPU | 状态 | 轮数/最佳轮次 |','|---|---:|---:|---:|---|---|']
    for r in rows:lines.append(f"| {r['name']} | {r['prior_weight']} | {r['jacobian_weight']} | {r['physical_gpu']} | {r['status']} | {r.get('epochs','—')}/{r.get('best_epoch','—')} |")
    lines+=['','## 原评测任务','',
            '| 模型 | 验证整体 | 验证未来预测 | 测试历史重构 | 测试未来预测 | 测试整体 |',
            '|---|---:|---:|---:|---:|---:|']
    def val(r,split,metric):return f"{r[split][metric]['reconstruction_error']:.5f}" if split in r else '—'
    for r in rows:lines.append('| '+r['name']+' | '+' | '.join(val(r,s,k) for s,k in [('validation','overall'),('validation','forecast'),('test','interpolation'),('test','forecast'),('test','overall')])+' |')
    lines+=['','## 校正前预测与局部敏感性','',
            '| 模型 | 验证校正前误差 | 测试校正前误差 | 验证平均Jacobian平方范数 | 测试平均Jacobian平方范数 |',
            '|---|---:|---:|---:|---:|']
    for r in rows:
        def prior(split):return f"{r[split]['prior']['reconstruction_error']:.5f}" if 'prior' in r.get(split,{}) else '—'
        def jac(split):return f"{r[split]['dynamics']['jacobian_frobenius_squared']:.6f}" if 'dynamics' in r.get(split,{}) else '—'
        lines.append(f"| {r['name']} | {prior('validation')} | {prior('test')} | {jac('validation')} | {jac('test')} |")
    lines+=['','weak_prior使用历史第2次及以后、当前观测更新前的MSE＋交叉熵。Jacobian计算共享向量场完整28×28导数，784项平方求和，再对有效生成状态取平均；采样位置停止梯度，正则训练向量场。','',
            '联合组的损失为原损失＋0.01×weak_prior＋0.01×Jacobian。所有组仍按原验证整体误差选择检查点，测试列在完整训练后填写。','',
            '已有latent2x不重训，三组新训练接入SwanLab。完整配置、定义和运行方式见 [实验设计](../../docs/LATENT2X_LOSS_ABLATION.zh-CN.md)。','']
    if all(r['status']=='complete' for r in rows):
        lines+=['','全部完成。按验证整体误差排序：'+'、'.join(r['name'] for r in sorted(rows,key=lambda r:r['validation']['overall']['reconstruction_error']))+'。']
    temp=out/f'summary.md.{os.getpid()}.tmp';temp.write_text('\n'.join(lines));temp.replace(out/'summary.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve();old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('already running')
    manifest,snapshot=prepare(out);report(out,manifest)
    if args.prepare_only:return
    for name,expected in read(snapshot/'source_hashes.json').items():
        if digest(snapshot/name)!=expected:raise ValueError('frozen source changed')
    if digest(manifest['dataset'])!=manifest['dataset_sha256']:raise ValueError('data changed')
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','physical_gpus':[1,2,3]};atomic(out/'queue.json',state)
    def execute(job):
        dest=Path(job['output']);dest.mkdir(parents=True,exist_ok=True)
        try:
            if (dest/'metrics.json').exists():
                if read(dest/'config.json')!=read(Path(job['config'])):raise ValueError('completed configuration differs')
                return
            if (dest/'history.json').exists():raise ValueError('preserve interrupted run')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['physical_gpu']),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
            for smoke in [True,False]:
                target=out/'checks'/job['name'] if smoke else dest;target.mkdir(parents=True,exist_ok=True)
                if smoke and (target/'metrics.json').exists():continue
                cmd=[sys.executable,'-u','-m','paper_odeae.train','--config',job['config'],'--output',str(target)]
                if smoke:cmd.append('--smoke')
                print(f"GPU {job['physical_gpu']}: {job['name']}, smoke={smoke}",flush=True)
                with (target/'train.log').open('w') as log:subprocess.run(cmd,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            (dest/'failure.json').unlink(missing_ok=True)
        except Exception as error:
            atomic(dest/'failure.json',{'error':f'{type(error).__name__}: {error}','job':job});raise
    try:
        failures=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending={pool.submit(execute,job) for job in manifest['jobs']}
            while pending:
                done,pending=wait(pending,timeout=30)
                for future in done:
                    try:future.result()
                    except Exception as error:failures.append(str(error))
                report(out,manifest)
        if failures:raise RuntimeError('; '.join(failures))
        state['status']='waiting_for_baseline';atomic(out/'queue.json',state)
        while not (BASE/'metrics.json').exists():time.sleep(30);report(out,manifest)
        if not (out/'baseline_diagnostics.json').exists():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            cmd=[sys.executable,'-u','-m','paper_odeae.diagnostics.jacobian_checkpoint','--checkpoint',str(BASE/'best.pt'),'--output',str(out/'baseline_diagnostics.json')]
            with (out/'baseline_diagnostics.log').open('w') as stream:subprocess.run(cmd,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        verification=[]
        for job in manifest['jobs']:
            path=Path(job['output']);hw=read(path/'hardware.json');m=read(path/'metrics.json')
            assert hw['trainable_parameters']==44875 and hw['physical_gpu']==str(job['physical_gpu']) and hw['parameter_device']=='cuda:0'
            assert hw['dataset_sha256']==manifest['dataset_sha256'] and m['checkpoint_reload_difference']<=1e-6
            verification.append({'name':job['name'],'hardware':hw,'reload_difference':m['checkpoint_reload_difference']})
        atomic(out/'completion_verification.json',{'runs':verification,'baseline_retrained':False})
        state['status']='complete';atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);report(out,manifest);raise


if __name__=='__main__':main()
