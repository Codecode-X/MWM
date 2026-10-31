"""MIMIC replication of the ADNI equal-capacity history encoder ablations."""
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from paper_odeae.scaling_study import identity
from .prepare import atomic
from .report import read

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'mimic_experiment/results/history_encoder_ablation_cuda'
REFERENCE=ROOT/'mimic_experiment/results/latent2x_both_cuda'
ADNI=ROOT/'paper_odeae/results/latent2x_history_encoder_ablation'
GPUS={'reverse_shared':2,'attention_time':3}
NAMES={'reference':'latent2x both（正向ODE）','reverse_shared':'反向共享ODE','attention_time':'注意力＋实际时间编码'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    base=read(REFERENCE/'config.json');hw=read(REFERENCE/'seed_0/hardware.json')
    if hw['parameters']!=43280 or base['latent']!=28 or not read(REFERENCE/'seed_0/metrics.json'):
        raise RuntimeError('Use the completed 43,280-parameter MIMIC both reference')
    sha=digest(base['data'])
    audit=read(ROOT/'mimic_experiment/results/baselines_single_cuda/data_verification.json')
    if sha!=hw['dataset_sha256'] or sha!=audit['dataset_sha256'] or audit['status']!='passed':raise RuntimeError('Reference MIMIC data changed')
    if read(ADNI/'queue.json')['status']!='complete':raise RuntimeError('ADNI source ablations are incomplete')
    configs={};(out/'configs').mkdir(exist_ok=True)
    for variant,gpu in GPUS.items():
        cfg={**base,'study':'mimic_history_encoder_ablation','variant':variant,'history_encoder_variant':variant,
             'encoder_direction':'backward' if variant=='reverse_shared' else 'attention','output':str(out/variant/'seed_0'),
             'comparison_output':str(out),'physical_gpu':gpu,'dataset_sha256':sha,'time_unit':'hour','attention_time_unit':'hour',
             'reference_output':str(REFERENCE),'adni_output':str(ADNI),'verify_reference':variant=='reverse_shared',
             'prior_semantics':'original forward causal supervision; auxiliary state is not the main reverse state' if variant=='reverse_shared' else
                               'causal attention state through j-1; shared ODE predicts j before observing j',
             'capacity_change':'none; original networks and widths retained' if variant=='reverse_shared' else
                               'only replace UPDATE 16,087 parameters with matched attention 16,087; embedding width42, feedforward169',
             'generator_anchor':'first history visit' if variant=='reverse_shared' else 'history cutoff',
             'history_semantics':'history-conditioned reconstruction','future_semantics':'free shared ODE without future observation updates',
             'micro_batch_size':32,'gradient_partitions':4,'source_adni_variant':str(ADNI/variant/'seed_0')}
        config=out/'configs'/f'{variant}.json'
        if config.exists() and read(config)!=cfg:raise RuntimeError('Preserve existing MIMIC ablation configuration')
        atomic(config,cfg);configs[variant]=cfg
    source=out/'source_snapshot';source.mkdir(exist_ok=True)
    files=list(read(ROOT/'mimic_experiment/results/baselines_single_cuda/manifest.json')['source_hashes'])
    files+=['paper_odeae/history_encoders.py','mimic_experiment/train_history_encoder.py',
            'mimic_experiment/history_encoder_checks.py','mimic_experiment/history_encoder_study.py']
    if not (source/'source_hashes.json').exists():
        for name in files:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        atomic(source/'source_hashes.json',{name:digest(source/name) for name in files})
    hashes=read(source/'source_hashes.json')
    for name,sha_ in hashes.items():
        if digest(source/name)!=sha_:raise RuntimeError('Frozen MIMIC history encoder source changed')
    manifest={'reference_output':str(REFERENCE),'reference_checkpoint_sha256':digest(REFERENCE/'seed_0/best.pt'),
              'dataset':base['data'],'dataset_sha256':sha,'reference_retrained':False,'adni_output':str(ADNI),
              'adni_metrics_sha256':{v:digest(ADNI/v/'seed_0/metrics.json') for v in GPUS},
              'configs':configs,'parameters':43280,'latent':28,'common_parameters':{'init':4798,'shared_ode':10744,'decoder':11651},
              'history_encoder_parameters':16087,'attention':{'embedding_hidden':42,'feedforward_hidden':169,'heads':4,'periods_hours':[.5,48.]},
              'source_sha256':hashes,'physical_gpus':GPUS,'seed':0,'effective_batch':128,'gradient_partitions':4,
              'loss_weights':{'prior':.01,'jacobian':.01,'ode_l2':.02},'dropout':'reproduce the original four rank streams and local padding'}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Preserve existing experiment manifest')
    atomic(out/'manifest.json',manifest);atomic(out/'data_verification.json',{**audit,'dataset_hash_rechecked':True})
    return configs,source,manifest


def collect(out):
    queue=read(out/'queue.json',{});rows=[]
    for key,path in [('reference',REFERENCE/'seed_0')]+[(v,out/v/'seed_0') for v in GPUS]:
        history=read(path/'history.json',[]);result=read(path/'metrics.json');best=min(history,key=lambda x:x['validation']['overall']['reconstruction_error']) if history else None
        state=queue.get('jobs',{}).get(key,{})
        rows.append({'key':key,'name':NAMES[key],'path':str(path),'status':'complete' if result else state.get('status','waiting'),
                     'phase':state.get('phase','training'),'epochs':result['epochs_run'] if result else len(history),
                     'best_epoch':result['best_epoch'] if result else best['epoch'] if best else None,
                     'validation':result['validation'] if result else best['validation'] if best else None,
                     'result':result,'hardware':read(path/'hardware.json',{}),'history':history})
    return rows


def report(out,manifest):
    rows=collect(out)
    def num(v):return f'{v:.6f}' if v is not None else '—'
    lines=['# MIMIC：历史读取方向与ODE编码作用的等参数消融','',
           '## 1. 沿用ADNI的两个问题','',
           '本轮继续比较正向ODE编码、反向共享ODE编码，以及因果注意力加实际时间编码。参照使用已完成的MIMIC latent2x both；新增两组从seed 0独立训练。三组均为28维状态、43,280参数。','',
           '| 模型 | 主历史编码 | 生成起点 | 物理GPU |','|---|---|---|---:|',
           '| latent2x both | 正向ODE＋UPDATE | 最后一次历史检查 | 已完成四卡参照 |',
           '| reverse_shared | 同一个ODE反向编码＋相同UPDATE | 第一次历史检查 | 2 |',
           '| attention_time | 因果注意力＋实际经过小时编码 | 最后一次历史检查 | 3 |','',
           '反向组保留全部原有网络；主状态反向读到起点后，沿同一速度场向前自由生成。weak prior保留同参数辅助正向扫描，只用于监督。注意力组替换历史编码模块，历史状态形成过程不调用ODE；共享ODE负责主生成和校正前预测。','',
           '## 2. 保持相同容量、数据与训练','',
           '两组保留INIT宽159、生成ODE宽188、解码器宽159。注意力编码模块与原UPDATE均为16,087参数：观测嵌入44→42→28，四头注意力，前馈28→169→28及两个LayerNorm。时间编码使用固定0.5至48小时周期，没有可学习参数。','',
           '同一chronology_v2缓存，训练21,517人、验证4,649人、测试4,536人；31个字段、44维展开输出。训练随机选择6/12/18/24小时历史，主要测试读前24小时、预测到48小时。归一化只使用训练患者。','',
           '新增两组使用ode环境单卡训练，有效batch 128，四份最多32人的分片累积梯度。患者分片、各分片的补齐长度和输入dropout随机流复现四卡参照。数据损失与weak prior按整批字段计数归约，Jacobian按整批有效状态归约，ODE L2每步计一次。','',
           '保留0.01 weak prior、0.01 Jacobian、0.02 ODE L2；Adam学习率0.003及原衰减，最多300轮、至少50轮、早停耐心40轮。验证整体数据误差选模，重新载入最佳检查点后测试，全部训练接入SwanLab。','',
           '## 3. 训练进度与主要结果','',
           '| 模型 | 阶段 | 已训练 / 最佳轮次 | 验证整体 | 测试历史 | 测试未来 | 测试整体 | 未来数值标准化MSE |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        v=row['validation'];t=row['result']['test'] if row['result'] else None
        values=[row['name'],row['status']+(f" / {row['phase']}" if row['status']!='complete' else ''),f"{row['epochs']} / {row['best_epoch'] or '—'}",num(v['overall']['reconstruction_error'] if v else None)]
        values += [num(t[k]['reconstruction_error']) if t else '—' for k in ['interpolation','forecast','overall']]
        values += [num(t['mean_standardized_numeric_mse']) if t else '—'];lines.append('| '+' | '.join(values)+' |')
    lines+=['','## 4. 未来不同范围与较短历史','',
            '| 模型 | 未来6小时 | 未来12小时 | 未来24小时 | 仅6小时历史后的未来误差 | 仅12小时历史后的未来误差 |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        result=row['result'];t=result['test'] if result else None
        values=[num(t[f'forecast_{h}h']['reconstruction_error']) if t else '—' for h in [6,12,24]]
        values += [num(result['test_early_contexts'][str(h)]['forecast']['reconstruction_error']) if result else '—' for h in [6,12]]
        lines.append('| '+' | '.join([row['name']]+values)+' |')
    lines+=['','## 5. 在两个数据集上分别量化收益','',
            '差值＝消融测试误差−本数据集正向参照误差；未来相对误差降低＝未来差值÷消融未来误差。正值表示正向ODE方案更好。','',
            '| 数据集 | 消融 | 历史误差差值 | 未来误差差值 | 整体误差差值 | 未来相对误差降低 |','|---|---|---:|---:|---:|---:|']
    adni_reference=read(ROOT/'paper_odeae/results/latent2x_loss_ablation/both/seed_0/metrics.json')['test']
    for dataset in ['ADNI','MIMIC']:
        reference=adni_reference if dataset=='ADNI' else rows[0]['result']['test']
        for key in GPUS:
            target=read(ADNI/key/'seed_0/metrics.json')['test'] if dataset=='ADNI' else next(r['result']['test'] if r['result'] else None for r in rows if r['key']==key)
            differences=[target[k]['reconstruction_error']-reference[k]['reconstruction_error'] for k in ['interpolation','forecast','overall']] if target else None
            values=[num(v) for v in differences]+[f"{100*differences[1]/target['forecast']['reconstruction_error']:.2f}%"] if target else ['—']*4
            lines.append('| '+' | '.join([dataset,NAMES[key]]+values)+' |')
    lines+=['','## 6. 临床单位误差与动力学','',
            '逐字段MAE、RMSE、GCS准确率及保持最近值的对照保存在[完整比较数据](comparison.json)。数值字段表另提供[MAE下载](per_field_mae.csv)和[RMSE下载](per_field_rmse.csv)。','',
            '| 模型 | 测试校正前预测误差 | 平均Jacobian平方范数 | GCS睁眼准确率 | GCS运动准确率 | GCS语言准确率 |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        t=row['result']['test'] if row['result'] else None
        values=[num(t['prior']['reconstruction_error']) if t else '—',num(t['dynamics']['jacobian_frobenius_squared']) if t else '—']
        values += [f"{100*t['clinical_forecast'][f]['accuracy']:.2f}%" if t else '—' for f in ['gcs_eye','gcs_motor','gcs_verbal']]
        lines.append('| '+' | '.join([row['name']]+values)+' |')
    lines+=['','## 7. 设计、检查与输出','',
            '[实验设计](../../HISTORY_ENCODER_ABLATION.zh-CN.md) · [ADNI结果](../../../paper_odeae/results/latent2x_history_encoder_ablation/summary.md) · [配置和源码清单](manifest.json) · [运行队列](queue.json)','',
            '检查包括参数匹配、共有网络初始权重一致、共享场身份、时间单位、未来信息隔离、注意力因果性、全部参数的有限梯度，以及分片损失与整批损失的梯度一致性。参照检查点在CUDA上重新评价并核对已完成结果。训练完成后逐组核对测试患者数与观测数。','',
            '```bash','bash scripts/run_mimic_history_encoder_ablation.sh','```','']
    import csv
    fields=rows[0]['result']['test']['clinical_forecast']
    for metric in ['mae','rmse']:
        with (out/f'per_field_{metric}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['field','unit']+[r['key'] for r in rows]+['carry_forward'])
            for field,value in fields.items():
                if metric not in value:continue
                values=[r['result']['test']['clinical_forecast'][field][metric] if r['result'] else '' for r in rows]
                writer.writerow([field,value['unit']]+values+[rows[0]['result']['test']['carry_forward'][field][metric]])
    signature=[len(row['history']) for row in rows]
    if any(signature[1:]):
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
        lines+=['![训练与验证曲线](training_curves.png)','']
    (out/'summary.md').write_text('\n'.join(lines));atomic(out/'comparison.json',{'dataset_sha256':manifest['dataset_sha256'],'updated_at':datetime.now(timezone.utc).isoformat(),'models':[{k:v for k,v in r.items() if k!='history'} for r in rows]})


def verify_completed(out,key):
    run=out/key/'seed_0';result=read(run/'metrics.json');reference=read(REFERENCE/'seed_0/metrics.json');hw=read(run/'hardware.json')
    assert result['status']=='complete' and result['checkpoint_reload_difference']<=1e-6
    assert hw['parameters']==hw['trainable_parameters']==43280 and hw['world_size']==1
    assert hw['parameter_device']==hw['prediction_device']=='cuda:0' and set(hw['actual_tensor_devices'].values())=={'cuda:0'}
    for split in ['validation','test']:
        assert result[split]['patients']==reference[split]['patients']
        for region in ['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']:
            assert result[split][region]['observed_scalars']==reference[split][region]['observed_scalars']
    for horizon in ['6','12']:assert result['test_early_contexts'][horizon]['patients']==reference['test_early_contexts'][horizon]['patients']
    atomic(run/'completion_verification.json',{'status':'passed','same_test_patients_and_observation_counts':True,'parameters':43280,'physical_gpu':hw['physical_gpu'],'device':'cuda:0'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode for orchestration')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('MIMIC history encoder study already running')
    configs,source,manifest=prepare(out);workers={};streams={}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
           'orchestrator_source_sha256':digest(Path(__file__)),
           'jobs':{v:{'status':'waiting','phase':'checks','physical_gpu':g} for v,g in GPUS.items()}}
    atomic(out/'queue.json',state);report(out,manifest)
    try:
        for stage in ['checks','smoke','training']:
            for key,cfg in configs.items():
                folder=Path(cfg['output']) if stage=='training' else out/'checks'/key/stage;folder.mkdir(parents=True,exist_ok=True)
                expected=folder/('checks.json' if stage=='checks' else 'metrics.json')
                if expected.exists():
                    wanted='passed' if stage=='checks' else 'cuda_smoke_passed' if stage=='smoke' else 'complete'
                    if read(expected)['status']!=wanted:raise RuntimeError('Incomplete previous stage')
                    if stage=='training':verify_completed(out,key);state['jobs'][key].update(status='complete',phase='tested')
                    continue
                while subprocess.check_output(['nvidia-smi','-i',str(cfg['physical_gpu']),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():report(out,manifest);time.sleep(10)
                config=out/'configs'/f'{key}.json'
                command=[sys.executable,'-u','-m','mimic_experiment.history_encoder_checks','--config',str(config),'--output',str(folder)] if stage=='checks' else [sys.executable,'-u','-m','mimic_experiment.train_history_encoder','--config',str(config),'--stage',stage]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['physical_gpu']),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'mimic-history-{key}-{stage}-seed0')
                stream=(folder/'run.log').open('a');streams[key]=stream
                worker=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers[key]=(worker,expected);state['jobs'][key].update(status='running',phase=stage,pid=worker.pid)
                atomic(out/'queue.json',state);print(f'{key}: {stage}, physical GPU {cfg["physical_gpu"]}, PID {worker.pid}',flush=True)
            last=0
            while workers:
                for key,(worker,expected) in list(workers.items()):
                    code=worker.poll()
                    if code is None:continue
                    streams.pop(key).close();workers.pop(key)
                    if code or not expected.exists():
                        state['jobs'][key].update(status='failed',exit_code=code);atomic(out/'queue.json',state);raise RuntimeError(f'{key}/{stage} failed; inspect run.log')
                    if stage=='training':verify_completed(out,key)
                    state['jobs'][key].update(status='complete' if stage=='training' else 'ready',phase='tested' if stage=='training' else stage+'_passed')
                    atomic(out/'queue.json',state);print(f'{key}: {stage} passed',flush=True)
                if time.monotonic()-last>=30:report(out,manifest);last=time.monotonic()
                time.sleep(2)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        for key in workers:state['jobs'][key]['status']='interrupted'
        atomic(out/'queue.json',state);report(out,manifest);raise
    finally:
        for worker,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in streams.values():stream.close()


if __name__=='__main__':main()
