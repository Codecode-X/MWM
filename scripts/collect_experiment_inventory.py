"""Read experiment artifacts for the consolidated report. No model execution."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ROOTS=['paper_odeae/results','mimic_experiment/results','healthflux/results','ablation/results','CausalMed/outputs','results','mwm/results']
EXCLUDE={'source_snapshot','analysis_source','recoveries','dispatch_backups','stop_backups','postprocessing_versions','.git','__pycache__','node_modules'}


def read(p,default=None):
    if not p.exists():return default
    try:return json.loads(p.read_text())
    except (ValueError,OSError):return default


def region(s,key):
    value=s.get(key,{}) if isinstance(s,dict) else {}
    return value.get('reconstruction_error') if isinstance(value,dict) else value


def dataset(path,cfg):
    if path.startswith('CausalMed/'):return 'CausalMed γ=1'
    if str(cfg.get('dataset','')).lower()=='adni':return 'ADNI'
    if str(cfg.get('dataset','')).lower()=='mimic':return 'MIMIC'
    if 'local_available' in str(cfg.get('data','')):return 'ADNI'
    if path.startswith('mimic_experiment/'):return 'MIMIC'
    if path.startswith('paper_odeae/') or path.startswith('healthflux/') or path.startswith('results/adni_cuda/'):return 'ADNI'
    if path.startswith('ablation/results/alpha01_components/'):return 'ADNI' if '/adni_' in path else 'MIMIC'
    if 'mimic' in str(cfg.get('data','')).lower():return 'MIMIC'
    if 'adni' in str(cfg.get('data','')).lower() or 'local_available' in str(cfg.get('data','')):return 'ADNI'
    return '早期非反事实合成/工程验证'


def get_parameters(hw,cfg):
    for k in ['parameters','model_parameters','total_parameters','total']:
        if isinstance(hw.get(k),int):return hw[k]
    if isinstance(hw.get('parameter_counts'),dict):
        return hw['parameter_counts'].get('total')
    return cfg.get('target_parameters')


def collect():
    runs=[];docs=[];artifacts=[];excluded=Counter()
    for root in ROOTS:
        for parent,dirs,files in os.walk(ROOT/root):
            for d in list(dirs):
                if d in EXCLUDE:dirs.remove(d);excluded[d]+=1
            folder=Path(parent);relative=folder.relative_to(ROOT).as_posix();parts=folder.parts
            for name in files:
                p=folder/name
                if name.endswith('.md'):docs.append(p.relative_to(ROOT).as_posix())
                if name in ['metrics.json','summary.json','comparison.json','statistics.json','verification.json']:
                    artifacts.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            config_path=next((folder/name for name in ['launch_config.json','config.json'] if (folder/name).exists()),None)
            if config_path is None:continue
            cfg=read(config_path,{})
            if not isinstance(cfg,dict):continue
            h=read(folder/'history.json',[])
            if isinstance(h,dict):h=h.get('history',[])
            if not isinstance(h,list):h=[]
            metrics_path=next((folder/name for name in ['metrics.json','metrics_150.json','training_complete.json'] if (folder/name).exists()),None)
            metrics=read(metrics_path,{}) if metrics_path else {}
            if not h and not metrics:continue
            if not isinstance(metrics,dict):continue
            test=metrics.get('test',{});val=metrics.get('validation',{})
            if not h and not test and not val:continue
            hw=read(folder/'hardware.json',{});stop=read(folder/'stop_request.json',{});queue=read(folder/'queue.json',{})
            latest=h[-1] if h else {}
            best_epoch=metrics.get('best_epoch',metrics.get('checkpoint_epoch'))
            best=None
            scored=[row for row in h if isinstance(row,dict) and region(row.get('validation',{}),'overall') is not None]
            if best_epoch is None and scored:
                best=min(scored,key=lambda r:region(r['validation'],'overall'));best_epoch=best.get('epoch')
            elif best_epoch is not None:best=next((r for r in h if r.get('epoch')==best_epoch),None)
            if not val and best:val=best.get('validation',{})
            if test:status='已测试'
            elif stop:status='已停止'
            elif queue.get('status') in ['paused_by_user','stopped_by_user','cancelled_by_user','user_paused','user_stopped']:status='已停止'
            elif h:status='有训练记录，未生成最终测试'
            else:status=str(metrics.get('status','有结果记录'))
            role='正式训练'
            if any(('smoke' in x.lower() or 'preflight' in x.lower() or x=='checks') for x in parts):role='预检/冒烟'
            elif 'baseline_reference' in parts or 'references' in parts:role='复用参照'
            elif 'pilot' in relative or 'pre_age_input' in relative:role='早期试跑'
            elif '20261101T094505Z' in relative:role='归档训练'
            record={'path':relative,'dataset':dataset(relative,cfg),'role':role,'status':status,
                    'seed':cfg.get('seed'),'model':cfg.get('model',cfg.get('variant')),
                    'parameters':get_parameters(hw,cfg),'latent':cfg.get('latent'),
                    'epochs_run':metrics.get('epochs_run',metrics.get('epochs',latest.get('epoch',len(h)))),
                    'best_epoch':best_epoch,'configured_max_epochs':cfg.get('max_epochs'),
                    'validation_overall':region(val,'overall'),'validation_future':region(val,'forecast'),
                    'test_history':region(test,'interpolation'),'test_future':region(test,'forecast'),'test_overall':region(test,'overall'),
                    'test_macro_mse_z':test.get('macro_mse_z') if isinstance(test,dict) else None,
                    'standardized_numeric_mse':test.get('mean_standardized_numeric_mse') if isinstance(test,dict) else None,
                    'physical_gpu':hw.get('physical_gpu',cfg.get('physical_gpu')),
                    'parameter_device':hw.get('parameter_device',hw.get('device')),
                    'config':config_path.relative_to(ROOT).as_posix(),
                    'metrics':metrics_path.relative_to(ROOT).as_posix() if metrics_path else None,
                    'history':(folder/'history.json').relative_to(ROOT).as_posix() if h else None}
            if isinstance(test,dict):
                for k in ['auroc','average_precision','ap','brier']:
                    if k in test:record['test_'+k]=test[k]
            runs.append(record)
    # Queue snapshots supply current states, but a completed metrics file wins.
    live=read(ROOT/'ablation/results/alpha01_components/queue.json',{}).get('jobs',{})
    for r in runs:
        if r['path'].startswith('ablation/results/alpha01_components/'):
            name=Path(r['path']).name
            if not r['test_overall'] and name in live:
                r['status']={'running':'运行中','stopped_by_user':'已停止','queued':'候补'}.get(live[name]['status'],live[name]['status'])
    timestamp=datetime.now(timezone.utc).isoformat()
    inventory={'snapshot_utc':timestamp,'scope':'ADNI, MIMIC, CausalMed plus separately labeled historical engineering checks',
               'method':'Filesystem JSON/history/statistical summaries only; no checkpoint loading, training or model evaluation',
               'runs':sorted(runs,key=lambda r:(r['dataset'],r['path'])),'documents':sorted(set(docs)),
               'artifact_fingerprints':artifacts,'excluded_duplicate_directories':dict(excluded),
               'counts':{'runs':len(runs),'by_dataset':dict(Counter(r['dataset'] for r in runs)),
                         'by_role':dict(Counter(r['role'] for r in runs)),'documents':len(set(docs))}}
    target=ROOT/'reproduction/experiment_inventory.json';target.write_text(json.dumps(inventory,indent=2,ensure_ascii=False))
    keys=sorted(set().union(*(r.keys() for r in runs)))
    with (ROOT/'reproduction/experiment_inventory.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(inventory['runs'])
    return inventory


if __name__=='__main__':
    inventory=collect();print(json.dumps(inventory['counts'],indent=2,ensure_ascii=False))
