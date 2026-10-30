"""One single-GPU weak prior-loss run; reuse the existing lambda=0 baseline."""
import argparse
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

DEFAULT=ROOT/'paper_odeae/results/shared_matched_prior_weak_single'
REFERENCE=ROOT/'paper_odeae/results/forward_age_scaling/shared_matched/seed_0'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    base=json.loads((REFERENCE/'config.json').read_text())
    cfg={**base,'study':'shared_matched_prior_single','variant':'weak_prior',
         'prior_weight':.01,'collect_prior_predictions':True,'output':str(out/'weak_prior/seed_0'),
         'ablation_family':'prior_supervision','ablation':'single GPU; pre-update history supervision, first visit excluded',
         'capacity_change':'none'}
    path=out/'config.json'
    if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('config changed')
    atomic(path,cfg)
    ref=out/'baseline_reference';ref.mkdir(exist_ok=True)
    for name in ['config.json','metrics.json','hardware.json','schema.json']:
        if (ref/name).exists() and (ref/name).read_bytes()!=(REFERENCE/name).read_bytes():raise ValueError('reference changed')
        shutil.copy2(REFERENCE/name,ref/name)
    manifest={'baseline':str(REFERENCE),'baseline_metrics_sha256':digest(ref/'metrics.json'),
              'baseline_retrained':False,'config':str(path),'output':cfg['output'],
              'dataset':cfg['data'],'dataset_sha256':digest(cfg['data']),
              'physical_gpu':0,'parameters':22545,'prior_weight':.01,
              'execution':'one single-GPU job, no new lambda=0 training'}
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('manifest changed')
    atomic(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot'
    files=['paper_odeae/__init__.py','paper_odeae/models.py','paper_odeae/train.py','paper_odeae/jacobian.py',
           'paper_odeae/data.py','paper_odeae/solver.py','paper_odeae/reference/features46.json',
           'paper_odeae/reference/tsit5_tableau.json','src/__init__.py','src/tracking.py','src/swanlab_local.py']
    if not (snapshot/'source_hashes.json').exists():
        for name in files:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        atomic(snapshot/'source_hashes.json',{name:digest(snapshot/name) for name in files})
    return manifest,snapshot


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def report(out,manifest):
    base=read(out/'baseline_reference/metrics.json');path=Path(manifest['output'])
    q=read(out/'queue.json',{});row={'status':q.get('status','prepared')}
    history=read(path/'history.json',[])
    if history:
        best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error'])
        row.update(epochs=len(history),best_epoch=best['epoch'],validation=best['validation'],training_components=best['training_components'])
    metrics=read(path/'metrics.json')
    if metrics:row.update(status='complete',epochs=metrics['epochs_run'],best_epoch=metrics['best_epoch'],validation=metrics['validation'],test=metrics['test'])
    atomic(out/'summary.json',{'queue':q,'baseline_retrained':False,'baseline':base,'weak_prior':row,
                              'updated_at':datetime.now(timezone.utc).isoformat()})
    def val(item,split,region):
        return f"{item[split][region]['reconstruction_error']:.5f}" if split in item and region in item[split] else '—'
    lines=['# shared_matched：单卡弱校正前预测约束','',
           '新增一组 λ=0.01 实验，在释放出的物理 GPU 0 上运行。λ=0 直接复用原单卡 shared_matched，不重跑训练。参数量均为22,545，其余训练条件保持一致。','',
           '| 模型 | λ | 状态 | 轮数/最佳轮次 | 验证整体 | 测试预测 | 测试整体 | 验证校正前 | 测试校正前 |',
           '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for name,weight,item,status,epochs,best in [('已有 shared_matched',0,base,'复用已完成结果',base['epochs_run'],base['best_epoch']),('weak_prior',.01,row,row['status'],row.get('epochs','—'),row.get('best_epoch','—'))]:
        lines.append(f'| {name} | {weight} | {status} | {epochs}/{best} | '+' | '.join(val(item,s,k) for s,k in [('validation','overall'),('test','forecast'),('test','overall'),('validation','prior'),('test','prior')])+' |')
    if metrics:
        lines+=['','相对已有单卡基线：']
        for region in ['forecast','overall']:
            old=base['test'][region]['reconstruction_error'];new=metrics['test'][region]['reconstruction_error']
            lines.append(f'- 测试 {region}：{old:.5f} → {new:.5f}，误差降低 {(old-new)/old*100:.2f}%。')
        hw=read(path/'hardware.json')
        assert hw['physical_gpu']=='0' and hw['parameters']==22545 and hw['parameter_device']=='cuda:0'
        assert hw['dataset_sha256']==manifest['dataset_sha256'] and metrics['checkpoint_reload_difference']<=1e-6
        atomic(out/'completion_verification.json',{'hardware':hw,'checkpoint_reload_difference':metrics['checkpoint_reload_difference']})
    lines+=['','新增损失只使用历史第2次及以后有真实观测的字段，预测发生在当前观测进入UPDATE之前。首次检查、未来区域和缺失标签不参与。只增加监督项，不直接惩罚状态跳变。',
            '选择检查点仍使用原验证整体误差，不把新增损失加进验证选择分数。旧基线没有保存校正前指标，对应单元格留空；不为此重跑基线。','']
    temp=out/f'summary.md.{os.getpid()}.tmp';temp.write_text('\n'.join(lines));temp.replace(out/'summary.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT))
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();out=Path(args.output).resolve()
    old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('already running')
    manifest,snapshot=prepare(out);report(out,manifest)
    if args.prepare_only:return
    for name,expected in read(snapshot/'source_hashes.json').items():
        if digest(snapshot/name)!=expected:raise ValueError('snapshot changed')
    if digest(manifest['dataset'])!=manifest['dataset_sha256']:raise ValueError('data changed')
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'checking','physical_gpu':0,'baseline_retrained':False}
    atomic(out/'queue.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
    try:
        for smoke in [True,False]:
            target=out/'checks/single_cuda_smoke' if smoke else Path(manifest['output'])
            target.mkdir(parents=True,exist_ok=True)
            if (target/'metrics.json').exists():continue
            if (target/'history.json').exists():raise ValueError(f'preserve interrupted task: {target}')
            cmd=[sys.executable,'-u','-m','paper_odeae.train','--config',manifest['config'],'--output',str(target)]
            if smoke:cmd.append('--smoke')
            state['status']='checking' if smoke else 'running';atomic(out/'queue.json',state);report(out,manifest)
            with (target/'train.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:time.sleep(10);report(out,manifest)
                    if child.returncode:raise RuntimeError(f'training failed: inspect {target}/train.log')
                finally:
                    if child.poll() is None:child.terminate();child.wait()
        state['status']='complete';atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);report(out,manifest);raise


if __name__=='__main__':main()
