"""Single-GPU Jacobian ablation and shared baseline/prior/Jacobian report."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .scaling_study import ROOT,atomic,identity
from .prior_study import read

DEFAULT=ROOT/'paper_odeae/results/shared_matched_jacobian_single'
WEAK=ROOT/'paper_odeae/results/shared_matched_prior_weak_single'
REFERENCE=ROOT/'paper_odeae/results/forward_age_scaling/shared_matched/seed_0'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out,gpu):
    out.mkdir(parents=True,exist_ok=True)
    base=read(REFERENCE/'config.json')
    cfg={**base,'study':'shared_matched_jacobian_single','variant':'jacobian',
         'prior_weight':0.,'collect_prior_predictions':True,'jacobian_weight':.01,'collect_jacobian_states':True,
         'output':str(out/'jacobian/seed_0'),'ablation_family':'jacobian_regularization','capacity_change':'none',
         'ablation':'exact mean squared Frobenius Jacobian at all valid generated visit states; detached locations'}
    if (out/'config.json').exists() and read(out/'config.json')!=cfg:raise ValueError('configuration changed')
    atomic(out/'config.json',cfg)
    manifest={'baseline':str(REFERENCE),'weak_prior':str(WEAK/'weak_prior/seed_0'),'baseline_retrained':False,
              'parameters':22545,'prior_weight':0.,'jacobian_weight':.01,'physical_gpu':gpu,
              'output':cfg['output'],'dataset':cfg['data'],'dataset_sha256':digest(cfg['data']),
              'regularizer':'exact ||W2 diag(1-tanh(W1z+b1)^2) W1||_F^2; mean over valid generated states; z detached'}
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


def run_row(path):
    row={'status':'queued'}
    history=read(path/'history.json',[])
    if history:
        best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error'])
        row.update(status='running',epochs=len(history),best_epoch=best['epoch'],validation=best['validation'],training_components=best.get('training_components',{}))
    metrics=read(path/'metrics.json')
    if metrics:row.update(status='complete',epochs=metrics['epochs_run'],best_epoch=metrics['best_epoch'],validation=metrics['validation'],test=metrics['test'])
    return row


def report(out,manifest):
    queue=read(out/'queue.json',{});base=read(REFERENCE/'metrics.json')
    base={**base,'epochs':base['epochs_run'],'status':'已有结果'}
    weak=run_row(WEAK/'weak_prior/seed_0');jac=run_row(Path(manifest['output']))
    if jac['status']!='complete' and queue.get('status') in ['checking','reference_evaluation','failed']:jac['status']=queue['status']
    for name,row in [('baseline',base),('weak_prior',weak)]:
        diagnostic=read(out/'reference_diagnostics'/f'{name}.json')
        if diagnostic:
            row['diagnostic_devices']={'physical_gpu':diagnostic['physical_gpu'],'device':diagnostic['device']}
            for split in ['validation','test']:
                row[split]={**row[split],'dynamics':diagnostic[split]['dynamics'],'prior':diagnostic[split]['prior']}
    summary={'updated_at':datetime.now(timezone.utc).isoformat(),'queue':queue,'baseline_retrained':False,
             'baseline':base,'weak_prior':weak,'jacobian':jac}
    lines=['# shared_matched：单卡损失消融比较','',
           '三组均为22,545参数、seed 0、相同数据与训练设置。基线复用已有 shared_matched；weak_prior 在 GPU 0 继续训练；新增 Jacobian 在 GPU 1 单卡训练。','',
           '| 模型 | 新增损失 | λ | 状态 | 轮数/最佳轮次 | 验证整体 | 测试预测 | 测试整体 | 验证校正前 | 测试校正前 |',
           '|---|---|---:|---|---|---:|---:|---:|---:|---:|']
    def val(row,split,key):return f"{row[split][key]['reconstruction_error']:.5f}" if split in row and key in row[split] else '—'
    rows=[('shared_matched','无',0,base),('weak_prior','校正前预测',.01,weak),('jacobian','速度场 Jacobian',.01,jac)]
    for name,loss,weight,row in rows:
        lines.append(f"| {name} | {loss} | {weight} | {row['status']} | {row.get('epochs','—')}/{row.get('best_epoch','—')} | "+' | '.join(val(row,s,k) for s,k in [('validation','overall'),('test','forecast'),('test','overall'),('validation','prior'),('test','prior')])+' |')
    lines+=['','两项新增损失分别做独立消融：weak_prior 优化读入当前观测之前的预测；jacobian 优化速度场在邻近状态上的变化平缓程度。Jacobian 组的 prior_weight=0，没有叠加 weak_prior。','',
            'Jacobian 采用完整6×6矩阵的精确计算，36项平方求和，再对批内有效生成状态取平均。状态覆盖历史与未来的实际检查时间，补齐位置排除；采样位置停止梯度，正则直接训练共享向量场参数。','',
            '| 模型 | 验证平均 Jacobian 平方范数 | 测试平均 Jacobian 平方范数 |','|---|---:|---:|']
    for name,_,_,row in rows:
        def metric(split):
            value=row.get(split,{}).get('dynamics',{}).get('jacobian_frobenius_squared')
            return '—' if value is None else f'{value:.6f}'
        lines.append(f'| {name} | {metric("validation")} | {metric("test")} |')
    lines+=['','Jacobian 指标反映模型在生成状态附近对扰动的局部敏感度；测试预测与整体误差反映预测效果。基线和 weak_prior 的此项指标由已有检查点补算，训练记录保持原样。所有组仍按原验证整体误差选择检查点。',
            '训练通过 SwanLab 记录 jacobian_frobenius_squared 与 weighted_jacobian。设计和公式见 [Jacobian 消融说明](../../docs/JACOBIAN_REGULARIZATION.zh-CN.md)。','']
    for destination in [out,WEAK]:
        atomic(destination/'summary.json',summary)
        temp=destination/f'summary.md.{os.getpid()}.tmp';temp.write_text('\n'.join(lines));temp.replace(destination/'summary.md')
    if queue.get('owns_weak_monitor'):
        state=read(WEAK/'queue.json',{})
        state.update(status='complete' if weak['status']=='complete' else 'running',pid=os.getpid(),start_time=identity(os.getpid()))
        atomic(WEAK/'queue.json',state)


def adopt_weak(out):
    old=read(WEAK/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):
        command=Path('/proc',str(old['pid']),'cmdline').read_bytes().decode().replace('\0',' ')
        if '-m paper_odeae.prior_study' not in command:raise RuntimeError('weak report already has another monitor')
        atomic(out/'previous_weak_monitor.json',old)
        # Stop the report controller only. Its training worker keeps running.
        os.kill(old['pid'],signal.SIGTERM)
    state={**old,'pid':os.getpid(),'start_time':identity(os.getpid()),'monitor':'paper_odeae.jacobian_study','baseline_retrained':False}
    atomic(WEAK/'queue.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT))
    parser.add_argument('--gpu',type=int,default=1,choices=[0,1,2,3]);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve()
    old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('Jacobian study already running')
    manifest,snapshot=prepare(out,args.gpu)
    if args.prepare_only:return
    for name,expected in read(snapshot/'source_hashes.json').items():
        if digest(snapshot/name)!=expected:raise ValueError('frozen source changed')
    if digest(manifest['dataset'])!=manifest['dataset_sha256']:raise ValueError('dataset changed')
    adopt_weak(out)
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'checking','owns_weak_monitor':True,'physical_gpu':args.gpu}
    atomic(out/'queue.json',state);report(out,manifest)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
    def execute(command,log):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            child=subprocess.Popen(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:time.sleep(10);report(out,manifest)
                if child.returncode:raise RuntimeError(f'job failed; inspect {log}')
            finally:
                if child.poll() is None:child.terminate();child.wait()
    def diagnostic(name,path):
        target=out/'reference_diagnostics'/f'{name}.json'
        if target.exists():return
        command=[sys.executable,'-u','-m','paper_odeae.diagnostics.jacobian_checkpoint','--checkpoint',str(path/'best.pt'),'--output',str(target)]
        execute(command,target.with_suffix('.log'))
    try:
        for smoke in [True,False]:
            target=out/'checks/single_cuda_smoke' if smoke else Path(manifest['output'])
            if (target/'metrics.json').exists():continue
            if (target/'history.json').exists():raise ValueError(f'preserve interrupted task: {target}')
            if not smoke:
                state['status']='reference_evaluation';atomic(out/'queue.json',state)
                diagnostic('baseline',REFERENCE)
            state['status']='checking' if smoke else 'running';atomic(out/'queue.json',state)
            command=[sys.executable,'-u','-m','paper_odeae.train','--config',str(out/'config.json'),'--output',str(target)]
            if smoke:command.append('--smoke')
            execute(command,target/'train.log')
        state['status']='waiting_for_weak_checkpoint';atomic(out/'queue.json',state)
        while not (WEAK/'weak_prior/seed_0/metrics.json').exists():time.sleep(30);report(out,manifest)
        diagnostic('weak_prior',WEAK/'weak_prior/seed_0')
        state['status']='complete';atomic(out/'queue.json',state);report(out,manifest)
        hardware=read(Path(manifest['output'])/'hardware.json');metrics=read(Path(manifest['output'])/'metrics.json')
        assert hardware['physical_gpu']==str(args.gpu) and hardware['trainable_parameters']==22545
        assert hardware['dataset_sha256']==manifest['dataset_sha256'] and metrics['checkpoint_reload_difference']<=1e-6
        atomic(out/'completion_verification.json',{'hardware':hardware,'checkpoint_reload_difference':metrics['checkpoint_reload_difference'],'baseline_retrained':False})
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);report(out,manifest);raise


if __name__=='__main__':main()
