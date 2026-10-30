"""Schedule the ADNI HealthFlux comparison on free physical RTX 4090 GPUs."""
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
import threading
import time

from paper_odeae.scaling_study import ROOT,atomic,identity
from paper_odeae.prior_study import read

DEFAULT=ROOT/'healthflux/results/adni_comparison'
BASE=ROOT/'paper_odeae/results/forward_age_scaling/shared_matched/seed_0'
REFERENCES=[('shared_matched (6D)',BASE),
 ('shared_matched latent2x (28D)',ROOT/'paper_odeae/results/shared_matched_latent_scaling/latent_2x/seed_0'),
 ('shared_matched latent4x (73D)',ROOT/'paper_odeae/results/shared_matched_scaling_4x/latent_4x/seed_0'),
 ('ODE-AE',ROOT/'paper_odeae/results/local_available/ode_ae/seed_0'),
 ('deterministic ODE-AE 22545',ROOT/'paper_odeae/results/capacity_matched/deterministic_22545/seed_0'),
 ('ODE-RNN',ROOT/'paper_odeae/results/baselines_full/native/ode_rnn/seed_0'),
 ('Latent ODE',ROOT/'paper_odeae/results/baselines_full/native/latent_ode/seed_0')]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);(out/'configs').mkdir(exist_ok=True)
    original=read(BASE/'config.json');jobs=[]
    for name,beta in [('common_task',1.),('paper_kl',.01)]:
        cfg={**original,'model':'healthflux','study':'healthflux_adni','variant':name,
             'deterministic_dim':256,'stochastic_dim':128,'context_dim':128,'age_dim':64,
             'observation_dim':256,'internal_width':384,'decoder_widths':[1024,512],'dropout':.1,
             'kl_weight':beta,'output':str(out/name/'seed_0'),'paper_version':3,
             'scope':'local_available','input_zscore_fit':'train only','ablation':'common task; beta=1 versus paper beta=.01',
             'capacity_change':'native HealthFlux 256+128D; no parameter-count matching'}
        for key in ['target_parameters','latent','ode_width','mlp_width','encoder_direction','encoder_initialization','shared_vector_field','ablation_family','rtol','atol']:cfg.pop(key,None)
        path=out/'configs'/f'{name}.json'
        if path.exists() and read(path)!=cfg:raise ValueError('Preserve existing configuration')
        atomic(path,cfg);jobs.append({'name':name,'config':str(path),'output':cfg['output'],'kl_weight':beta})
    manifest={'created_by':'healthflux.study','dataset':original['data'],'dataset_sha256':digest(original['data']),
              'environment':'healthflux','baseline_retrained':False,'jobs':jobs,
              'paper':read(ROOT/'healthflux/reference/source_audit.json'),
              'reference_runs':[{'name':name,'path':str(path),'metrics_sha256':digest(path/'metrics.json')} for name,path in REFERENCES],
              'selection':'original ADNI validation overall mixed-field reconstruction error',
              'hardware_policy':'free physical GPU among 0,1,2,3; one worker per GPU; existing training preserved'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise ValueError('Manifest changed')
    atomic(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot'
    files=['healthflux/__init__.py','healthflux/model.py','healthflux/train.py','healthflux/study.py',
           'healthflux/tests/__init__.py','healthflux/tests/test_model.py',
           'paper_odeae/__init__.py','paper_odeae/data.py','paper_odeae/models.py','paper_odeae/train.py',
           'paper_odeae/solver.py','paper_odeae/jacobian.py','paper_odeae/scaling_study.py','paper_odeae/prior_study.py',
           'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json',
           'src/__init__.py','src/tracking.py','src/swanlab_local.py']
    if not (snapshot/'source_hashes.json').exists():
        for name in files:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        atomic(snapshot/'source_hashes.json',{name:digest(snapshot/name) for name in files})
    return manifest,snapshot


def row(path):
    hardware=read(path/'hardware.json',{});history=read(path/'history.json',[])
    result={'status':'queued','parameters':hardware.get('trainable_parameters'),'total_parameters':hardware.get('parameters'),
            'physical_gpu':hardware.get('physical_gpu'),'path':str(path)}
    track=read(path/'swanlab.json',{})
    if track.get('status')=='running':result['status']='running'
    if history:
        best=min(history,key=lambda item:item['validation']['overall']['reconstruction_error'])
        result.update(status='running',epochs=len(history),best_epoch=best['epoch'],validation=best['validation'],
                      mean_epoch_seconds=sum(item.get('epoch_seconds',0) for item in history)/len(history))
    metrics=read(path/'metrics.json')
    if metrics:result.update(status='complete',epochs=metrics['epochs_run'],best_epoch=metrics['best_epoch'],validation=metrics['validation'],test=metrics['test'])
    if (path/'failure.json').exists():result['status']='failed'
    return result


def report(out,manifest):
    refs=[{'name':name,**row(path)} for name,path in REFERENCES]
    rows=[{'name':'HealthFlux/'+job['name'],'kl_weight':job['kl_weight'],**row(Path(job['output']))} for job in manifest['jobs']]
    queue=read(out/'queue.json',{})
    atomic(out/'summary.json',{'updated_at':datetime.now(timezone.utc).isoformat(),'queue':queue,'runs':rows,'references':refs})
    lines=['# HealthFlux 在 ADNI 共同任务上的比较','',
           '## 1. 比较的数据和任务','',
           '两组 HealthFlux 使用同一份 ADNI 40 字段缓存、2093/393/414 位训练/验证/测试患者，枚举相同历史截止点。数值字段使用相同 min-max 目标，类别字段使用相同交叉熵，缺失字段与区域计数直接调用已有 evaluate 函数。','',
           'HealthFlux 按原文保留 256 维确定性状态、128 维随机状态、384 宽 MLP、年龄和静态背景条件、Gaussian 先验/后验、GRU＋LayerNorm，以及 RK4。新增的任务适配是用已有健康状态输出 ADNI 40 字段，在历史截止处向后重构、向前预测。未来不进行观测更新。','',
           '## 2. 训练设置','',
           '两组从相同 seed 0 初始化，同用 batch 32、Adam 0.003、最多300轮、最少50轮和耐心值40。common_task 的目标是 MSE＋CCE＋1×KL＋0.02×ODE L2，与已有概率模型比较协议一致。paper_kl 只将 KL 系数改为论文的0.01。已有基线直接复用，未重训。','',
           '| 新实验 | KL系数 | GPU | 状态 | 轮数/最佳轮次 |','|---|---:|---|---|---|']
    for r in rows:
        jobname=r['name'].split('/')[-1];state=queue.get('jobs',{}).get(jobname,{})
        lines.append(f"| {r['name']} | {r['kl_weight']} | {r.get('physical_gpu') or state.get('physical_gpu','待空闲')} | {r['status'] if r['status']!='queued' else state.get('status','queued')} | {r.get('epochs','—')}/{r.get('best_epoch','—')} |")
    lines+=['','## 3. 同一评分函数的完整对比','',
            '历史重构列使用截止点已经获得的历史；未来预测列只使用截止点之前的信息。整体误差按全部有效字段重新汇总。下表保留各模型实际容量，HealthFlux采用论文原容量。','',
            '| 模型 | 可训练参数 | 总参数 | 状态 | 轮数 | 最佳验证整体 | 测试历史 | 测试未来 | 测试整体 |',
            '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    def score(r,split,key):return f"{r[split][key]['reconstruction_error']:.5f}" if split in r else '—'
    for r in refs+rows:
        param=lambda k:f"{r[k]:,}" if r.get(k) is not None else '—'
        lines.append(f"| {r['name']} | {param('parameters')} | {param('total_parameters')} | {r['status']} | {r.get('epochs','—')} | "+' | '.join(score(r,s,k) for s,k in [('validation','overall'),('test','interpolation'),('test','forecast'),('test','overall')])+' |')
    if all(r['status']=='complete' for r in rows):
        best=min(rows,key=lambda r:r['validation']['overall']['reconstruction_error'])
        lines+=['',f"按原验证整体误差，两组 HealthFlux 中选择 {best['name']}。"]
        for reference in refs[:3]:
            old=reference['test']['forecast']['reconstruction_error'];new=best['test']['forecast']['reconstruction_error']
            lines.append(f"与 {reference['name']} 相比，测试未来误差变化为 {(new/old-1)*100:+.2f}%（负值为误差下降）。")
    lines+=['','## 4. 实现和运行记录','',
            '论文原训练预测下一临床事件及时间；本表使用共同的ADNI字段重构/预测目标。实现依据论文v3 Methods公式2–6，公开代码检索与逐项适配见 [设计和实现说明](../../README.zh-CN.md)。','',
            '检查点、history、实际CUDA设备、参数量、SwanLab、数据哈希和冻结源码保存在各实验目录。测试结果在完整训练及最佳检查点重载核对之后填入表格。','',
            '运行命令：`bash scripts/run_healthflux_adni.sh`。队列仅占用空闲GPU；正在运行的其他实验继续保留。','']
    path=out/f'summary.md.{os.getpid()}.tmp';path.write_text('\n'.join(lines));path.replace(out/'summary.md')


def free_gpus():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=15)
    mapping={line.split(',')[1].strip():int(line.split(',')[0]) for line in gpu.splitlines()}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15)
    busy={mapping[line.split(',')[0].strip()] for line in processes.splitlines() if ',' in line and line.split(',')[0].strip() in mapping}
    return [g for g in range(4) if g not in busy]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve();old=read(out/'queue.json',{})
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('HealthFlux queue already running')
    if Path(sys.prefix).name!='healthflux':raise RuntimeError('Dedicated healthflux environment required')
    manifest,snapshot=prepare(out);report(out,manifest)
    if args.prepare_only:return
    for name,expected in read(snapshot/'source_hashes.json').items():
        if digest(snapshot/name)!=expected:raise RuntimeError('Frozen source changed')
    if digest(manifest['dataset'])!=manifest['dataset_sha256']:raise RuntimeError('Dataset changed')
    lock=threading.RLock();held=set();state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running',
        'jobs':{job['name']:{'status':'waiting_for_free_gpu'} for job in manifest['jobs']}}
    def update(name,**kwargs):
        with lock:state['jobs'][name].update(kwargs);atomic(out/'queue.json',state)
    def execute(job):
        name=job['name'];dest=Path(job['output']);dest.mkdir(parents=True,exist_ok=True);gpu=None
        try:
            if (dest/'metrics.json').exists():update(name,status='complete');return
            if (dest/'history.json').exists():raise RuntimeError('Preserve incomplete training run')
            while gpu is None:
                with lock:
                    candidates=[g for g in free_gpus() if g not in held]
                    if candidates:gpu=candidates[0];held.add(gpu)
                if gpu is None:time.sleep(20)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
            for smoke in [True,False]:
                target=out/'checks'/name if smoke else dest;target.mkdir(parents=True,exist_ok=True)
                if smoke and (target/'metrics.json').exists():continue
                update(name,status='checking' if smoke else 'running',physical_gpu=gpu)
                cmd=[sys.executable,'-u','-m','healthflux.train','--config',job['config'],'--output',str(target)]
                if smoke:cmd.append('--smoke')
                print(f'GPU {gpu}: HealthFlux {name}, smoke={smoke}',flush=True)
                with (target/'train.log').open('w') as stream:subprocess.run(cmd,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            hw=read(dest/'hardware.json');metrics=read(dest/'metrics.json')
            assert hw['dataset_sha256']==manifest['dataset_sha256'] and hw['environment']=='healthflux'
            assert hw['parameter_device']=='cuda:0' and metrics['checkpoint_reload_difference']<=1e-6
            update(name,status='complete');(dest/'failure.json').unlink(missing_ok=True)
        except Exception as error:
            atomic(dest/'failure.json',{'error':f'{type(error).__name__}: {error}'});update(name,status='failed',error=str(error));raise
        finally:
            if gpu is not None:
                with lock:held.discard(gpu)
    atomic(out/'queue.json',state)
    try:
        errors=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(execute,job) for job in manifest['jobs']}
            while pending:
                done,pending=wait(pending,timeout=30)
                for task in done:
                    try:task.result()
                    except Exception as error:errors.append(str(error))
                report(out,manifest)
        if errors:raise RuntimeError('; '.join(errors))
        state['status']='complete';atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as error:
        state.update(status='failed',error=str(error));atomic(out/'queue.json',state);report(out,manifest);raise


if __name__=='__main__':main()
