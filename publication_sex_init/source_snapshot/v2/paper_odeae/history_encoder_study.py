"""Two equal-capacity, single-GPU ADNI history-encoder ablations."""
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .scaling_study import ROOT,atomic,identity
from .prior_study import read

DEFAULT=ROOT/'paper_odeae/results/latent2x_history_encoder_ablation'
BASE=ROOT/'paper_odeae/results/latent2x_loss_ablation/both/seed_0'
VARIANTS={'reverse_shared':2,'attention_time':3}
NAMES={'reference':'latent2x both（正向ODE）','reverse_shared':'反向共享ODE','attention_time':'因果注意力＋连续时间编码'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    base=read(BASE/'config.json');hardware=read(BASE/'hardware.json')
    if base['latent']!=28 or base['target_parameters']!=44875 or not read(BASE/'metrics.json'):
        raise RuntimeError('Use the completed 28D latent2x both reference')
    sha=digest(base['data'])
    if sha!=hardware['dataset_sha256']:raise RuntimeError('ADNI reference dataset changed')
    if hardware['parameter_breakdown']!={'init':4798,'decoder':12451,'update':16882,'backward_ode':10744}:
        raise RuntimeError('Reference network allocation changed')
    (out/'configs').mkdir(exist_ok=True);configs={}
    for variant,gpu in VARIANTS.items():
        cfg={**base,'study':'latent2x_history_encoder_ablation','variant':variant,'history_encoder_variant':variant,
             'encoder_direction':'backward' if variant=='reverse_shared' else 'attention',
             'output':str(out/variant/'seed_0'),'physical_gpu':gpu,'dataset_sha256':sha,'resume':True,
             'reference_output':str(BASE),'reference_model_source':str(BASE.parent.parent/'source_snapshot/paper_odeae/models.py'),
             'verify_reference_checkpoint':variant=='reverse_shared','generator_anchor':'first history visit' if variant=='reverse_shared' else 'history cutoff',
             'prior_semantics':'original forward causal one-step supervision; auxiliary pass does not supply the main reverse state' if variant=='reverse_shared' else
                               'causal attention prefix through j-1; shared ODE predicts j before reading its observation',
             'capacity_change':'none; all original weights and widths retained' if variant=='reverse_shared' else
                               'replace only 16,882-parameter UPDATE with an exactly 16,882-parameter attention history encoder',
             'ablation_family':'history_encoder','ablation':'main history encoding strategy; common 28D state, field, decoder and losses'}
        path=out/'configs'/f'{variant}.json'
        if path.exists() and read(path)!=cfg:raise RuntimeError('Preserve existing ablation configuration')
        atomic(path,cfg);configs[variant]=cfg
    source=out/'source_snapshot';source.mkdir(exist_ok=True)
    files=['paper_odeae/__init__.py','paper_odeae/models.py','paper_odeae/history_encoders.py','paper_odeae/history_encoder_checks.py',
           'paper_odeae/history_encoder_study.py','paper_odeae/train.py','paper_odeae/data.py','paper_odeae/solver.py','paper_odeae/jacobian.py',
           'paper_odeae/scaling_study.py','paper_odeae/prior_study.py','paper_odeae/reference/features46.json','paper_odeae/reference/tsit5_tableau.json',
           'src/__init__.py','src/tracking.py','src/swanlab_local.py']
    if not (source/'source_hashes.json').exists():
        for name in files:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        atomic(source/'source_hashes.json',{name:digest(source/name) for name in files})
    hashes=read(source/'source_hashes.json')
    for name,sha_ in hashes.items():
        if digest(source/name)!=sha_:raise RuntimeError('Frozen history encoder source changed')
    manifest={'reference':str(BASE),'reference_checkpoint_sha256':digest(BASE/'best.pt'),'reference_config_sha256':digest(BASE/'config.json'),
              'dataset':base['data'],'dataset_sha256':sha,'parameters':44875,'latent':28,'configs':configs,
              'source_sha256':hashes,'physical_gpus':VARIANTS,'seed':0,'reference_retrained':False,
              'common_parameters':{'init':4798,'decoder':12451,'shared_ode':10744},'history_encoder_parameters':16882,
              'attention':{'heads':4,'embedding_hidden':59,'feedforward_hidden':156,'time_encoding':'fixed sin/cos of real elapsed years; periods .25..20 years'},
              'loss_weights':{'prior':.01,'jacobian':.01,'ode_l2':.02}}
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise RuntimeError('Manifest changed')
    atomic(out/'manifest.json',manifest)
    return configs,source,manifest


def report(out,manifest):
    queue=read(out/'queue.json',{});rows=[]
    for variant,path in [('reference',BASE)]+[(name,Path(cfg['output'])) for name,cfg in manifest['configs'].items()]:
        history=read(path/'history.json',[]);metrics=read(path/'metrics.json');best=min(history,key=lambda h:h['validation']['overall']['reconstruction_error']) if history else None
        state=queue.get('jobs',{}).get(variant,{})
        rows.append({'variant':variant,'name':NAMES[variant],'path':str(path),'status':'complete' if metrics else state.get('status','waiting'),
                     'phase':state.get('phase','training'),'epochs':metrics['epochs_run'] if metrics else len(history),
                     'best_epoch':metrics['best_epoch'] if metrics else best['epoch'] if best else None,
                     'validation':metrics['validation'] if metrics else best['validation'] if best else None,
                     'test':metrics['test'] if metrics else None,'hardware':read(path/'hardware.json',{}),'history':history})
    lines=['# ADNI：历史读取方向与ODE编码作用的等参数消融','',
           '## 1. 比较从哪里开始','',
           '参照是已经完成训练的latent2x both。它使用28维状态、44,875参数，按时间从前往后吸收观测，从最后一个历史状态出发重构过去并预测未来。本轮复用其检查点和结果，新增两组独立单卡实验。','',
           '数据保持为同一ADNI缓存：训练2,093人、验证393人、测试414人，40个有效字段、49维展开输出，按训练集min-max归一化。训练与评测继续枚举每位患者的所有非空历史前缀。','',
           '## 2. 每组具体改变什么','',
           '| 模型 | 主历史编码 | 生成起点 | 状态维度 | 总参数 | 物理GPU |','|---|---|---|---:|---:|---:|',
           '| latent2x both | 正向ODE＋UPDATE | 最后一次历史检查 | 28 | 44,875 | 历史实验1 |',
           '| reverse_shared | 反向ODE＋相同UPDATE | 第一次历史检查 | 28 | 44,875 | 2 |',
           '| attention_time | 因果自注意力＋真实时间编码 | 最后一次历史检查 | 28 | 44,875 | 3 |','',
           '反向组保留原有全部网络、隐藏宽度和共享关系。主编码从历史末端读到起点，再从起点沿同一个ODE向前生成完整轨迹。','',
           '注意力组只替换16,882参数的历史观测编码模块。它使用四头注意力，观测嵌入隐藏宽59，前馈隐藏宽156；固定正余弦编码使用实际相隔年数。因果掩码让每个前缀状态只读取当前及此前观测。INIT、28维状态、188宽生成ODE和159宽解码器全部保留。','',
           '两组均保持0.01 weak prior、0.01 Jacobian和0.02 ODE L2。反向组用同一套参数做辅助正向扫描，继续计算原定义的因果weak prior；辅助状态只用于这项监督。注意力组从截至上一次观测的注意力状态出发，用共享ODE预测下一次观测，计算同一含义的weak prior；ODE不参与注意力历史状态的形成。','',
           '## 3. 相同训练预算与当前进度','',
           'seed 0、batch 32、Adam学习率0.003、按0.97^(step/2000)衰减、输入dropout 0.1、梯度上限10。最多300轮、至少50轮，验证整体数据误差连续40轮未改善时早停。每组使用ode环境与一张4090，接入SwanLab。','',
           '| 模型 | 阶段 | 已训练 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |','|---|---|---:|---:|---:|---:|']
    def num(v):return f'{v:.6f}' if v is not None else '—'
    for row in rows:
        v=row['validation'];values=[num(v[k]['reconstruction_error']) if v else '—' for k in ['interpolation','forecast','overall']]
        lines.append('| '+' | '.join([row['name'],row['status']+(f" / {row['phase']}" if row['status']!='complete' else ''),f"{row['epochs']} / {row['best_epoch'] or '—'}"]+values)+' |')
    lines+=['','## 4. 完整测试结果','',
            '| 模型 | 测试历史重构 | 测试未来预测 | 测试整体 | 校正前预测误差 | 平均Jacobian平方范数 |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        t=row['test'];values=[num(t[k]['reconstruction_error']) if t else '—' for k in ['interpolation','forecast','overall','prior']]
        values.append(num(t['dynamics']['jacobian_frobenius_squared']) if t else '—');lines.append('| '+' | '.join([row['name']]+values)+' |')
    lines+=['','## 5. 正向ODE编码带来多少收益','',
            '对每个消融分别计算：误差差值＝消融误差−参照误差；正向ODE方案相对误差降低＝差值÷消融误差。正值表示参照方案更好。','',
            '| 对比 | 历史误差差值 | 未来误差差值 | 整体误差差值 | 未来相对误差降低 |','|---|---:|---:|---:|---:|']
    reference=rows[0]['test']
    for row in rows[1:]:
        t=row['test'];delta=[t[k]['reconstruction_error']-reference[k]['reconstruction_error'] for k in ['interpolation','forecast','overall']] if t else None
        relative=100*delta[1]/t['forecast']['reconstruction_error'] if t else None
        lines.append('| '+' | '.join([row['name']]+([num(v) for v in delta]+[f'{relative:.2f}%'] if t else ['—']*4))+' |')
    lines+=['','方向组用于比较“正向估计截止状态”与“反向估计起点再生成”的整套读取方式。注意力组用于比较同样的截止点预测任务下，ODE递推编码和直接时间注意力编码的差异。','',
            '## 6. 逐字段结果、检查和复现','',
            '[实验设计说明](../../HISTORY_ENCODER_ABLATION.zh-CN.md) · [固定配置与源码](manifest.json) · [运行队列](queue.json) · [完整指标与差值](comparison.json)','',
            '每组先通过CUDA正确性检查和小规模训练测试，再进入正式训练。检查覆盖等参数、共有网络初始权重一致、共享场身份、时间单位、未来信息隔离、注意力因果性，以及所有参数的有限梯度。反向组还重新评价参照检查点，核对共同评测代码。','',
            '训练结束自动重新载入最佳检查点，核对验证分数并测试。每组目录保存best.pt、last.pt、history.json、hardware.json、swanlab.json和metrics.json。逐字段MSE或交叉熵保存在comparison.json的validation/test/per_field中。','',
            '```bash','bash scripts/run_adni_history_encoder_ablation.sh','```','']
    if any(row['history'] for row in rows[1:]):
        signature=[len(row['history']) for row in rows]
        if signature!=read(out/'plot_signature.json'):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for row in rows:
                x=[h['epoch'] for h in row['history']]
                for ax,region in zip(axes,['overall','forecast']):
                    ax.plot(x,[h['validation'][region]['reconstruction_error'] for h in row['history']],label=row['variant'])
            for ax,title in zip(axes,['Validation overall','Validation forecast']):
                ax.set(xlabel='Epoch',ylabel='Mixed-field error',title=title);ax.legend();ax.grid(alpha=.15)
            fig.tight_layout()
            for ext in ['png','svg','pdf']:fig.savefig(out/f'training_curves.{ext}',dpi=160)
            plt.close(fig);atomic(out/'plot_signature.json',signature)
        lines+=['![训练与验证曲线](training_curves.png)','']
    (out/'summary.md').write_text('\n'.join(lines));atomic(out/'comparison.json',{'updated_at':datetime.now(timezone.utc).isoformat(),
           'dataset_sha256':manifest['dataset_sha256'],'models':[{k:v for k,v in row.items() if k!='history'} for row in rows]})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args()
    if Path(sys.prefix).name!='ode':raise RuntimeError('Use ode for orchestration')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);previous=read(out/'queue.json',{})
    if previous.get('pid') and identity(previous['pid'])==previous.get('start_time'):raise RuntimeError('History encoder study already running')
    configs,source,manifest=prepare(out);workers={};streams={}
    state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
           'jobs':{key:{'status':'waiting','phase':'checks','physical_gpu':gpu} for key,gpu in VARIANTS.items()}}
    atomic(out/'queue.json',state);report(out,manifest)
    try:
        for stage in ['checks','smoke','training']:
            for variant,cfg in configs.items():
                folder=Path(cfg['output']) if stage=='training' else out/'checks'/variant/stage;folder.mkdir(parents=True,exist_ok=True)
                expected=folder/('checks.json' if stage=='checks' else 'metrics.json')
                if expected.exists():
                    status=read(expected)['status'];valid='passed' if stage=='checks' else 'cuda_smoke_passed' if stage=='smoke' else 'local_or_paper_contract_run_completed'
                    if status!=valid:raise RuntimeError(f'{variant}/{stage}: incomplete previous output')
                    continue
                while subprocess.check_output(['nvidia-smi','-i',str(cfg['physical_gpu']),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    report(out,manifest);time.sleep(10)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['physical_gpu']),PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         SWANLAB_LOG_DIR=str(ROOT/'swanlog'),SWANLAB_EXPERIMENT_NAME=f'adni-history-{variant}-{stage}-seed0')
                config=out/'configs'/f'{variant}.json'
                if stage=='checks':command=[sys.executable,'-u','-m','paper_odeae.history_encoder_checks','--config',str(config),'--output',str(folder)]
                else:command=[sys.executable,'-u','-m','paper_odeae.train','--config',str(config),'--output',str(folder)]+(['--smoke'] if stage=='smoke' else [])
                stream=(folder/'run.log').open('a');streams[variant]=stream
                worker=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers[variant]=(worker,expected);state['jobs'][variant].update(status='running',phase=stage,pid=worker.pid)
                atomic(out/'queue.json',state);print(f'{variant}: {stage}, physical GPU {cfg["physical_gpu"]}, PID {worker.pid}',flush=True)
            last=0
            while workers:
                for variant,(worker,expected) in list(workers.items()):
                    code=worker.poll()
                    if code is None:continue
                    streams.pop(variant).close();workers.pop(variant)
                    if code or not expected.exists():
                        state['jobs'][variant].update(status='failed',exit_code=code);atomic(out/'queue.json',state)
                        raise RuntimeError(f'{variant}/{stage} failed; inspect its run.log')
                    state['jobs'][variant].update(status='complete' if stage=='training' else 'ready',phase='tested' if stage=='training' else stage+'_passed')
                    atomic(out/'queue.json',state);print(f'{variant}: {stage} passed',flush=True)
                if time.monotonic()-last>=30:report(out,manifest);last=time.monotonic()
                time.sleep(2)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'queue.json',state);report(out,manifest)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        for variant in workers:state['jobs'][variant]['status']='interrupted'
        atomic(out/'queue.json',state);report(out,manifest);raise
    finally:
        for worker,_ in workers.values():
            if worker.poll() is None:worker.terminate();worker.wait()
        for stream in streams.values():stream.close()


if __name__=='__main__':main()
