"""Four single-GPU 4x-capacity ablations, starting from shared_matched."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .scaling_study import ROOT, atomic, identity, parameter_count

DEFAULT = ROOT / 'paper_odeae/results/shared_matched_scaling_4x'
REFERENCE = ROOT / 'paper_odeae/results/forward_age_scaling/shared_matched/seed_0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(base):
    """Change one capacity setting at a time; choose nearest integer budget."""
    target = 4 * parameter_count(base)
    result = []
    for name, key in [('init_4x', 'init_width'), ('update_4x', 'update_width'),
                      ('ode_4x', 'ode_width'), ('latent_4x', 'latent')]:
        # Each count is affine in this single setting with all others fixed.
        unit = parameter_count({**base, key: 2}) - parameter_count({**base, key: 1})
        intercept = parameter_count({**base, key: 1}) - unit
        value = max(1, round((target - intercept) / unit))
        result.append((name, {key: value}))
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((REFERENCE / 'config.json').read_text())
    assert base['shared_vector_field'] and base['latent'] == 6
    assert parameter_count(base) == 22545
    hw = json.loads((REFERENCE / 'hardware.json').read_text())
    assert hw['model_source_sha256'] == digest(ROOT / 'paper_odeae/models.py')
    assert hw['dataset_sha256'] == digest(base['data'])
    (out / 'configs').mkdir(exist_ok=True)
    jobs = []
    for gpu, (name, changes) in enumerate(definitions(base)):
        cfg = {**base, **changes, 'study': 'shared_matched_scaling', 'variant': name,
               'ablation_family': 'single_component_4x', 'output': str(out / name / 'seed_0'),
               'capacity_change': json.dumps(changes, sort_keys=True),
               'ablation': 'shared_matched; target 4x total parameters; one capacity setting changed'}
        cfg['target_parameters'] = parameter_count(cfg)
        config = out / 'configs' / f'{name}.json'
        if config.exists() and json.loads(config.read_text()) != cfg:
            raise ValueError(f'configuration changed: {config}')
        atomic(config, cfg)
        jobs.append({'name': name, 'physical_gpu': gpu, 'changes': changes,
                     'parameters': cfg['target_parameters'], 'config': str(config), 'output': cfg['output']})
    reference = out / 'baseline_reference'
    reference.mkdir(exist_ok=True)
    for name in ['metrics.json', 'hardware.json', 'config.json', 'schema.json']:
        target = reference / name
        if target.exists() and target.read_bytes() != (REFERENCE / name).read_bytes():
            raise ValueError('reference results changed')
        shutil.copy2(REFERENCE / name, target)
    manifest = {'study': 'shared_matched_scaling', 'baseline_parameters': 22545,
                'target_parameters': 90180, 'baseline_original': str(REFERENCE),
                'baseline_reference': str(reference), 'baseline_retrained': False,
                'baseline_metrics_sha256': digest(reference / 'metrics.json'),
                'dataset': base['data'], 'dataset_sha256': hw['dataset_sha256'], 'jobs': jobs,
                'selection': 'checkpoint by validation overall; compare validation forecast at that checkpoint',
                'execution': 'one fixed independent worker on each physical RTX 4090 GPU 0,1,2,3'}
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('manifest changed')
    atomic(path, manifest)
    snapshot = out / 'source_snapshot'
    files = ['paper_odeae/__init__.py', 'paper_odeae/train.py', 'paper_odeae/models.py', 'paper_odeae/jacobian.py',
             'paper_odeae/data.py', 'paper_odeae/solver.py', 'paper_odeae/reference/features46.json',
             'paper_odeae/reference/tsit5_tableau.json', 'src/__init__.py', 'src/tracking.py', 'src/swanlab_local.py']
    if not (snapshot / 'source_hashes.json').exists():
        for name in files:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        atomic(snapshot / 'source_hashes.json', {name: digest(snapshot / name) for name in files})
    return manifest, snapshot


def report(out, manifest):
    baseline = json.loads((Path(manifest['baseline_reference']) / 'metrics.json').read_text())
    rows = []
    state = json.loads((out / 'queue.json').read_text()) if (out / 'queue.json').exists() else {}
    for job in manifest['jobs']:
        row = {**job, 'status': 'queued'}
        path = Path(job['output'])
        if (path / 'history.json').exists():
            try:
                history = json.loads((path / 'history.json').read_text())
            except json.JSONDecodeError:
                # Training may be writing the next epoch; refresh next tick.
                history = []
            if history:
                best = min(history, key=lambda r: r['validation']['overall']['reconstruction_error'])
                row.update(status='running', epochs=len(history), best_epoch=best['epoch'],
                           validation=best['validation'], training_components=best.get('training_components'),
                           mean_epoch_seconds=sum(r['epoch_seconds'] for r in history)/len(history),
                           peak_memory_bytes=max(r['peak_memory_bytes'] for r in history))
        if (path / 'metrics.json').exists():
            try:
                metrics = json.loads((path / 'metrics.json').read_text())
            except json.JSONDecodeError:
                metrics = None
            if metrics:
                row.update(status='complete', epochs=metrics['epochs_run'], best_epoch=metrics['best_epoch'],
                           validation=metrics['validation'], test=metrics['test'],
                           checkpoint_reload_difference=metrics['checkpoint_reload_difference'])
        if (path / 'failure.json').exists():
            row['status'] = 'failed'
        if (path / 'stopped_by_user.json').exists() and row['status'] != 'complete':
            row['status'] = 'stopped_by_user'
        if state.get('status') == 'stopped_by_user' and row['status'] != 'complete':
            row['status'] = 'stopped_by_user' if 'epochs' in row else 'cancelled_by_user'
        if 'validation' in row:
            old = baseline['validation']['forecast']['reconstruction_error']
            new = row['validation']['forecast']['reconstruction_error']
            row['validation_forecast_reduction_percent'] = (old-new)/old*100
        rows.append(row)
    atomic(out / 'summary.json', {'updated_at': datetime.now(timezone.utc).isoformat(),
                                 'queue_status': state.get('status'), 'baseline': baseline, 'runs': rows})
    lines = ['# shared_matched：总参数量扩大至约 4 倍', '',
             '基线为已完成的 shared_matched（22,545 参数）。每组只改变一种容量设置，目标总参数量 90,180。四张 RTX 4090 各运行一组，全部使用 seed 0 和 SwanLab。', '',
             '先按验证整体误差选检查点，再比较该检查点的验证未来预测误差；测试成绩只在训练完整结束后填写。', '',
             '| 组别 | 总参数量 | 状态 | 轮数 / 最佳轮次 | 验证整体 | 验证预测 | 测试历史重构 | 测试预测 | 测试整体 |',
             '|---|---:|---|---|---:|---:|---:|---:|---:|']
    def metric(m, split, region):
        return f"{m[split][region]['reconstruction_error']:.5f}" if split in m else '—'
    lines.append(f"| shared_matched 基线 | 22,545 | 已完成，复用原结果 | {baseline['epochs_run']} / {baseline['best_epoch']} | " +
                 ' | '.join(metric(baseline, s, r) for s, r in [('validation','overall'),('validation','forecast'),('test','interpolation'),('test','forecast'),('test','overall')]) + ' |')
    for row in rows:
        lines.append(f"| {row['name']} | {row['parameters']:,} | {row['status']} | {row.get('epochs','—')} / {row.get('best_epoch','—')} | " +
                     ' | '.join(metric(row, s, r) for s, r in [('validation','overall'),('validation','forecast'),('test','interpolation'),('test','forecast'),('test','overall')]) + ' |')
    if any(r['status']=='stopped_by_user' for r in rows):
        lines += ['', 'init_4x 已按用户要求停止，保留历史和检查点。其余三组继续运行；停止组不参加完整测试排名。']
    if all(r['status'] == 'complete' for r in rows):
        ranked = sorted(rows, key=lambda r: r['validation']['forecast']['reconstruction_error'])
        lines += ['', '四个扩容组按验证预测误差排序：' + '、'.join(r['name'] for r in ranked) + '。',
                  '扩大后的组仍需与基线比较；四组中最优不代表比基线更好。']
        verification = []
        for row in rows:
            hw = json.loads((Path(row['output']) / 'hardware.json').read_text())
            assert hw['trainable_parameters'] == row['parameters']
            assert hw['dataset_sha256'] == manifest['dataset_sha256']
            assert hw['physical_gpu'] == str(row['physical_gpu'])
            assert hw['parameter_device'].startswith('cuda') and hw['shared_vector_field']
            assert row['checkpoint_reload_difference'] <= 1e-6
            verification.append({'name': row['name'], 'hardware': hw})
        atomic(out / 'completion_verification.json', {'status': 'complete', 'runs': verification})
    lines += ['', 'latent 组保持所有隐藏层宽度不变，只扩大状态维度；这会自然增大与状态相连的 INIT、UPDATE、ODE 和 DECODER 的输入/输出层。它衡量状态空间容量，不能解释成只增大了某一个网络。', '',
              '其余学习率、批量、损失权重、缺失处理和数据划分均保持不变。共享 ODE 的 L2 只计算一次，固定系数 0.02；参数变多也会改变正则总量，需结合训练分量分析。当前为单种子对照。', '']
    temp = out / f'summary.md.{os.getpid()}.tmp'
    temp.write_text('\n'.join(lines)); temp.replace(out / 'summary.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(DEFAULT))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--monitor-existing', action='store_true')
    args = parser.parse_args(); out = Path(args.output).resolve()
    old = json.loads((out / 'queue.json').read_text()) if (out / 'queue.json').exists() else {}
    if old.get('pid') and identity(old['pid']) == old.get('start_time'):
        raise RuntimeError('study already running')
    if args.monitor_existing:
        manifest=json.loads((out/'manifest.json').read_text())
        children={}
        for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]:
            parts=line.strip().split(None,1)
            if len(parts)<2:continue
            for job in manifest['jobs']:
                if parts[1].startswith(str(Path(sys.executable))) and '--config '+job['config']+' ' in parts[1]:
                    children[job['name']]={'pid':int(parts[0]),'start_time':identity(int(parts[0]))}
        state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'monitoring_remaining',
               'workers':children,'reason':'init_4x stopped by user; retain the other three running workers'}
        atomic(out/'queue.json',state)
        while any(identity(w['pid'])==w['start_time'] for w in children.values()):
            report(out,manifest);__import__('time').sleep(30)
        missing=[j['name'] for j in manifest['jobs'] if not (Path(j['output'])/'stopped_by_user.json').exists() and not (Path(j['output'])/'metrics.json').exists()]
        state.update(status='failed' if missing else 'complete_with_user_stopped',missing=missing)
        atomic(out/'queue.json',state);report(out,manifest)
        return
    manifest, snapshot = prepare(out)
    report(out, manifest)
    if args.prepare_only:
        return
    for name, expected in json.loads((snapshot / 'source_hashes.json').read_text()).items():
        if digest(snapshot / name) != expected:
            raise ValueError('frozen source changed')
    state = {'pid': os.getpid(), 'start_time': identity(os.getpid()),
             'status': 'running_smokes' if args.smoke_only else 'running'}
    atomic(out / 'queue.json', state)
    def execute(job):
        path = Path(job['output']); path.mkdir(parents=True, exist_ok=True)
        try:
            if digest(manifest['dataset']) != manifest['dataset_sha256']:
                raise ValueError('dataset changed')
            cfg = json.loads(Path(job['config']).read_text())
            if (path / 'metrics.json').exists():
                if json.loads((path / 'config.json').read_text()) != cfg:
                    raise ValueError('completed config changed')
                return
            if (path / 'history.json').exists():
                raise ValueError('preserve interrupted training; use a new output directory')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['physical_gpu']), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONPATH=str(snapshot), SWANLAB_LOG_DIR=str(ROOT / 'swanlog'))
            for smoke in [True, False]:
                if args.smoke_only and not smoke:
                    continue
                target = out / 'checks' / job['name'] if smoke else path
                target.mkdir(parents=True, exist_ok=True)
                if smoke and (target / 'metrics.json').exists():
                    smoke_cfg = json.loads((target / 'config.json').read_text())
                    smoke_hw = json.loads((target / 'hardware.json').read_text())
                    assert smoke_hw['dataset_sha256'] == manifest['dataset_sha256']
                    assert smoke_cfg['target_parameters'] == job['parameters']
                    continue
                cmd = [sys.executable, '-u', '-m', 'paper_odeae.train', '--config', job['config'], '--output', str(target)]
                if smoke:
                    cmd.append('--smoke')
                print(f"GPU {job['physical_gpu']}: {job['name']}, smoke={smoke}", flush=True)
                with (target / 'train.log').open('w') as stream:
                    subprocess.run(cmd, cwd=snapshot, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            (path / 'failure.json').unlink(missing_ok=True)
        except Exception as error:
            atomic(path / 'failure.json', {'error': f'{type(error).__name__}: {error}', 'job': job})
            raise
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(execute, job) for job in manifest['jobs']}
            while pending:
                done, pending = wait(pending, timeout=30)
                for future in done:
                    future.result()
                report(out, manifest)
        state['status'] = 'smokes_complete' if args.smoke_only else 'complete'
        atomic(out / 'queue.json', state); report(out, manifest)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic(out / 'queue.json', state); report(out, manifest)
        raise


if __name__ == '__main__':
    main()
