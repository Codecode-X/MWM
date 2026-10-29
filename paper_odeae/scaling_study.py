"""Prepare, validate and run the forward_age sharing/capacity study."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Queue,Empty
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'paper_odeae/results/forward_age_scaling'


def atomic(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except FileNotFoundError:return None


def parameter_count(cfg):
    d=cfg['latent'];f=49
    def network(a,b,w):return (a+1)*w+(w+1)*b
    count=network(1,d,cfg.get('init_width',cfg['mlp_width']))+network(d+f,d,cfg.get('update_width',cfg['mlp_width']))+network(d,f,cfg.get('decoder_width',cfg['mlp_width']))
    count+=network(d,d,cfg.get('generator_ode_width',cfg['ode_width']))
    if not cfg.get('shared_vector_field'):count+=network(d,d,cfg.get('encoder_ode_width',cfg['ode_width']))
    return count


def definitions():
    return [
      ('baseline','reference',{}),
      ('shared_tied','sharing',{'shared_vector_field':True}),
      ('shared_tied_l2x2','sharing_regularization_control',{'shared_vector_field':True,'shared_l2_multiplier':2.}),
      ('shared_matched','sharing_parameter_control',{'shared_vector_field':True,'mlp_width':159,'ode_width':188}),
      ('init_2x','allocation_2x',{'init_width':2967}),
      ('update_2x','allocation_2x',{'update_width':517}),
      ('decoder_2x','allocation_2x',{'decoder_width':556}),
      ('encoder_ode_2x','allocation_2x',{'encoder_ode_width':1849}),
      ('generator_ode_2x','allocation_2x',{'generator_ode_width':1849}),
      ('both_odes_2x','allocation_2x',{'ode_width':983}),
      ('all_mlps_2x','allocation_2x',{'mlp_width':333}),
      ('all_networks_2x','allocation_2x',{'mlp_width':308,'ode_width':236}),
      ('init_4x','dose_response_4x',{'init_width':8608}),
      ('update_4x','dose_response_4x',{'update_width':1245}),
      ('decoder_4x','dose_response_4x',{'decoder_width':1362}),
      ('encoder_ode_4x','dose_response_4x',{'encoder_ode_width':5321}),
      ('generator_ode_4x','dose_response_4x',{'generator_ode_width':5321}),
      ('both_odes_4x','dose_response_4x',{'ode_width':2719}),
      ('all_mlps_4x','dose_response_4x',{'mlp_width':691}),
      ('all_networks_4x','dose_response_4x',{'mlp_width':616,'ode_width':472}),
      ('latent_12','latent_fixed_width',{'latent':12}),
      ('latent_24','latent_fixed_width',{'latent':24}),
      ('latent_48','latent_fixed_width',{'latent':48}),
      ('latent_24_matched','latent_parameter_control',{'latent':24,'mlp_width':77,'ode_width':73})]


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'paper_odeae/configs/deterministic_forward_age_from22545.json').read_text())
    base['data']=str((ROOT/base['data']).resolve())
    dataset_hash=hashlib.sha256(Path(base['data']).read_bytes()).hexdigest()
    jobs=[];(out/'configs').mkdir(exist_ok=True)
    for name,family,changes in definitions():
        cfg={**base,**changes,'study':'forward_age_scaling','variant':name,'ablation_family':family,
             'seed':0,'output':str(out/name/'seed_0'),'smoke':False}
        cfg['target_parameters']=parameter_count(cfg)
        cfg['ablation']='forward_age sharing and parameter allocation; see study design'
        cfg['capacity_change']=json.dumps(changes,sort_keys=True)
        path=out/'configs'/f'{name}.json'
        if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError(f'configuration changed: {path}')
        atomic(path,cfg)
        jobs.append({'name':name,'family':family,'parameters':cfg['target_parameters'],'changes':changes,'config':str(path),'output':cfg['output']})
    manifest={'dataset_sha256':dataset_hash,'dataset':base['data'],'jobs':jobs,'seed':0,
        'execution':'four independent single-GPU workers; global batch 32 for each job',
        'selection':'checkpoint by validation overall; model ranking by validation forecast of that checkpoint; test descriptive only'}
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('study manifest changed')
    atomic(path,manifest)
    snapshot=out/'source_snapshot'
    if not (snapshot/'source_hashes.json').exists():
        files=['paper_odeae/__init__.py','paper_odeae/train.py','paper_odeae/models.py','paper_odeae/jacobian.py','paper_odeae/data.py','paper_odeae/solver.py',
               'paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json',
               'src/__init__.py','src/tracking.py','src/swanlab_local.py']
        hashes={}
        for name in files:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
            hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        atomic(snapshot/'source_hashes.json',hashes)
    return manifest,snapshot


def report(out,manifest):
    rows=[]
    for job in manifest['jobs']:
        path=Path(job['output']);row={**job,'status':'queued'}
        for filename in ['history.json','metrics.json']:
            try:value=json.loads((path/filename).read_text())
            except (FileNotFoundError,json.JSONDecodeError):continue
            if filename=='history.json' and value:
                best=min(value,key=lambda x:x['validation']['overall']['reconstruction_error'])
                row.update(epochs=len(value),best_epoch=best['epoch'],validation_overall=best['validation']['overall']['reconstruction_error'],validation_forecast=best['validation']['forecast']['reconstruction_error'],status='incomplete')
                row['best_training_components']=best.get('training_components',{})
                row['mean_epoch_seconds']=sum(r.get('epoch_seconds',0) for r in value)/len(value)
                row['peak_memory_bytes']=max(r.get('peak_memory_bytes',0) for r in value)
                row['budget_limited']=len(value)==300
                row['best_near_budget_end']=best['epoch']>=290
            elif filename=='metrics.json':
                row.update(status='complete',test={k:v['reconstruction_error'] for k,v in value['test'].items()},elapsed_seconds=value['elapsed_seconds'],checkpoint_reload_difference=value['checkpoint_reload_difference'])
        try:
            tracking=json.loads((path/'swanlab.json').read_text())
            if row['status']!='complete':row['status']=tracking['status']
        except (FileNotFoundError,json.JSONDecodeError):pass
        if (path/'failure.json').exists() and row['status']!='complete':row['status']='failed'
        rows.append(row)
    queue_state=json.loads((out/'queue.json').read_text()) if (out/'queue.json').exists() else {}
    stopped=queue_state.get('status')=='stopped_by_user'
    if stopped:
        for row in rows:
            if row['status']!='complete':
                row['status']='stopped_by_user' if 'epochs' in row else 'cancelled_by_user'
    baseline=rows[0]
    for row in rows:
        if 'validation_forecast' in baseline and 'validation_forecast' in row:
            row['validation_forecast_gain']=baseline['validation_forecast']-row['validation_forecast']
            additional=row['parameters']-baseline['parameters']
            if additional>0:row['gain_per_10k_parameters']=row['validation_forecast_gain']/additional*10000
    atomic(out/'summary.json',{'updated_at':datetime.now(timezone.utc).isoformat(),'queue_status':queue_state.get('status'),'runs':rows})
    lines=['# forward_age 共享向量场与参数扩容实验','',
       '全部任务使用forward_age、seed 0、相同数据与训练设置；四卡各运行一个独立实验。先用同一执行方式重新训练baseline，旧DDP结果只作外部参照。', '',
       '每个检查点按验证整体误差选择；优先根据该检查点的验证预测误差比较方案。测试结果只在完整训练后填写，不用测试集决定下一组结构。未完成任务的相对收益仍会变化。','',
       '| 组别 | 参数量 | 状态 | 轮数 | 最佳验证整体 | 对应验证预测 | 测试预测 | 测试整体 | 每新增万参数的验证预测收益 |',
       '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    if stopped:
        lines[2:2]=['用户已停止本轮消融。已完成结果保留；stopped_by_user 表示训练中断，cancelled_by_user 表示取消尚未开始的任务。中断任务没有完整测试结果。','']
    for r in rows:
        fmt=lambda x:'—' if x is None else f'{x:.5f}'
        lines.append(f"| {r['name']} | {r['parameters']:,} | {r['status']} | {r.get('epochs','—')} | {fmt(r.get('validation_overall'))} | {fmt(r.get('validation_forecast'))} | {fmt(r.get('test',{}).get('forecast'))} | {fmt(r.get('test',{}).get('overall'))} | {fmt(r.get('gain_per_10k_parameters'))} |")
    completed=[r for r in rows if r['status']=='complete' and 'validation_forecast' in r]
    if completed:
        lines+=['','已完成组按验证预测误差排序：'+', '.join(r['name'] for r in sorted(completed,key=lambda r:r['validation_forecast']))+'。','']
    if len(completed)==len(rows):
        best=min(completed,key=lambda r:r['validation_forecast'])
        allocation=[r for r in completed if r['family']=='allocation_2x']
        efficient=max((r for r in completed if 'gain_per_10k_parameters' in r),key=lambda r:r['gain_per_10k_parameters'])
        budget_best=min(allocation,key=lambda r:r['validation_forecast'])
        lines+=['','## 完成后的自动比较','',
            f"验证预测误差最低：{best['name']}；约4.5万参数组中最低：{budget_best['name']}；每新增万参数的验证预测收益最高：{efficient['name']}。",'',
            '这里列的是固定seed 0的观察结果。若最佳检查点接近300轮上限，应先检查收敛再推断扩容无效。测试成绩只用于最终确认，不反过来挑模型。','',
            '| 配置 | 平均每轮秒数 | 峰值已分配显存 MiB | 达到300轮 | 最优轮次≥290 |','|---|---:|---:|---|---|']
        for r in rows:lines.append(f"| {r['name']} | {r.get('mean_epoch_seconds',0):.2f} | {r.get('peak_memory_bytes',0)/2**20:.1f} | {r.get('budget_limited')} | {r.get('best_near_budget_end')} |")
        verify=[]
        for r in rows:
            path=Path(r['output']);hw=json.loads((path/'hardware.json').read_text());m=json.loads((path/'metrics.json').read_text())
            assert hw['trainable_parameters']==r['parameters']
            assert hw['dataset_sha256']==manifest['dataset_sha256']
            assert hw['parameter_device'].startswith('cuda')
            assert m['checkpoint_reload_difference']<=1e-6
            verify.append({'name':r['name'],'physical_gpu':hw['physical_gpu'],'device':hw['parameter_device'],'parameters':r['parameters'],'checkpoint_reload_difference':m['checkpoint_reload_difference']})
        atomic(out/'completion_verification.json',{'status':'all_complete','runs':verify})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(13,7))
        for r in rows:
            ax.scatter(r['parameters'],r['validation_forecast']);ax.annotate(r['name'],(r['parameters'],r['validation_forecast']),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(xlabel='Trainable parameters',ylabel='Validation forecast error (lower is better)',title='forward_age sharing and capacity study, seed 0')
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'capacity_vs_validation.{ext}',dpi=180)
        plt.close(fig)
    lines+=['','收益为baseline误差减本组误差，正数代表改善；分母是新增可训练参数。共享和等参组不以该比值比较。所有组固定L2系数，容量变化也会改变正则总量，需结合summary.json中的训练重构与正则分量解释。单种子结果不等于显著差异。','']
    target=out/f'summary.md.{os.getpid()}.tmp';target.write_text('\n'.join(lines));target.replace(out/'summary.md')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args();out=Path(args.output).resolve()
    manifest,snapshot=prepare(out);report(out,manifest)
    if args.prepare_only:return
    old=json.loads((out/'queue.json').read_text()) if (out/'queue.json').exists() else {}
    if old.get('pid') and identity(old['pid'])==old.get('start_time'):raise RuntimeError('study already running')
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_existing_queue'}
    atomic(out/'queue.json',state)
    dependency=ROOT/'paper_odeae/results/direction_ablation/queue.json'
    try:
        if dependency.exists():
            dep=json.loads(dependency.read_text());pid=dep['pid'];start=identity(pid)
            state['dependency_pid']=pid;atomic(out/'queue.json',state)
            while start is not None and identity(pid)==start:
                time.sleep(30);report(out,manifest)
            dep=json.loads(dependency.read_text())
            if dep['status']!='complete':raise RuntimeError('existing ablation queue did not finish successfully')
        hashes=json.loads((snapshot/'source_hashes.json').read_text())
        for name,expected in hashes.items():
            if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=expected:raise ValueError('source snapshot changed')
        if hashlib.sha256(Path(manifest['dataset']).read_bytes()).hexdigest()!=manifest['dataset_sha256']:raise ValueError('dataset changed')
        jobs=Queue()
        for job in manifest['jobs']:jobs.put(job)
        failures=Queue()
        def execute(job,gpu):
            cfg=json.loads(Path(job['config']).read_text());destination=Path(job['output'])
            if hashlib.sha256(Path(manifest['dataset']).read_bytes()).hexdigest()!=manifest['dataset_sha256']:raise ValueError('dataset changed')
            if (destination/'metrics.json').exists():
                saved=json.loads((destination/'config.json').read_text())
                if saved!=cfg:raise ValueError('completed configuration differs')
                hw=json.loads((destination/'hardware.json').read_text())
                if hw['dataset_sha256']!=manifest['dataset_sha256']:raise ValueError('completed dataset differs')
                return
            if (destination/'history.json').exists():raise ValueError(f'preserve interrupted task: {destination}')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONPATH=str(snapshot),SWANLAB_LOG_DIR=str(ROOT/'swanlog'))
            for smoke in [True,False]:
                if args.smoke_only and not smoke:continue
                target=out/'checks'/job['name'] if smoke else destination
                target.mkdir(parents=True,exist_ok=True)
                if smoke and (target/'metrics.json').exists():continue
                command=[sys.executable,'-u','-m','paper_odeae.train','--config',job['config'],'--output',str(target)]
                if smoke:command.append('--smoke')
                print(f'GPU {gpu}: {job["name"]}, smoke={smoke}',flush=True)
                with (target/'train.log').open('w') as stream:
                    subprocess.run(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            (destination/'failure.json').unlink(missing_ok=True)
        def worker(gpu):
            while True:
                try:job=jobs.get_nowait()
                except Empty:return
                try:execute(job,gpu)
                except Exception as error:
                    destination=Path(job['output']);destination.mkdir(parents=True,exist_ok=True)
                    failure={'variant':job['name'],'physical_gpu':gpu,'error':f'{type(error).__name__}: {error}'}
                    atomic(destination/'failure.json',failure);failures.put(failure)
                    print(json.dumps(failure),flush=True)
        state['status']='running_smokes' if args.smoke_only else 'running';atomic(out/'queue.json',state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(worker,gpu) for gpu in range(4)}
            while pending:
                done,pending=wait(pending,timeout=60)
                for future in done:future.result()
                report(out,manifest)
        if not failures.empty():raise RuntimeError(f'{failures.qsize()} study tasks failed; inspect failure.json and logs')
        state['status']='smokes_complete' if args.smoke_only else 'complete';atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');atomic(out/'queue.json',state);raise


if __name__=='__main__':main()
