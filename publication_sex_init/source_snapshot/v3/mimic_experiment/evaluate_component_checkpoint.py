"""Evaluate one immutable ADNI component-ablation snapshot on CUDA only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from paper_odeae.train import evaluate, seed_all
from component_ablation_model import make_model
from component_ablation_loss import jacobian_regularization


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


class ClinicalMetrics:
    def __init__(self, schema, metadata, train):
        self.schema = schema
        self.scale = metadata['scale']
        self.std = []
        for j, field in enumerate(schema):
            values = np.concatenate([s['values'][:, j][s['mask'][:, j].astype(bool)] for s in train])
            self.std.append(float(np.std(values.astype(np.float64))))
        self.totals = {region: torch.zeros(3, len(schema), device='cuda', dtype=torch.float64)
                       for region in ['interpolation', 'forecast', 'overall']}
        self.confusions = {region: {f['name']: torch.zeros(f['classes'], f['classes'], device='cuda', dtype=torch.int64)
                                   for f in schema if f['kind'] == 'categorical'} for region in self.totals}
        self.devices = None

    def add(self, batch, out):
        self.devices = {k: str(v.device) for k, v in batch.items() if torch.is_tensor(v)}
        self.devices['prediction'] = str(out['prediction'].device)
        assert all(v.startswith('cuda') for v in self.devices.values())
        idx = torch.arange(batch['values'].shape[1], device='cuda')[None]
        regions = {'interpolation': idx < batch['cutoff'][:, None],
                   'forecast': idx >= batch['cutoff'][:, None], 'overall': batch['valid']}
        at = 0
        for j, field in enumerate(self.schema):
            width = field.get('classes', 1)
            for region, selected in regions.items():
                valid = selected & batch['mask'][..., j].bool()
                self.totals[region][0, j] += valid.sum()
                if field['kind'] == 'numerical':
                    diff = (out['prediction'][..., at]-batch['values'][..., j])[valid].double()
                    self.totals[region][1, j] += diff.square().sum()
                    self.totals[region][2, j] += diff.abs().sum()
                else:
                    pred = out['prediction'][..., at:at+width].argmax(-1)[valid]
                    target = batch['values'][..., j][valid].long()
                    self.confusions[region][field['name']] += torch.bincount(
                        target*width+pred, minlength=width*width).reshape(width, width)
            at += width

    def results(self, native):
        result = {}
        for region, totals in self.totals.items():
            fields = {}; standardized = []
            for j, field in enumerate(self.schema):
                count = int(totals[0, j])
                assert count == native[region]['per_field'][field['name']]['observations']
                if not count:
                    continue
                row = {'observations': count}
                if field['kind'] == 'numerical':
                    mse = float(totals[1, j]/count); mae = float(totals[2, j]/count)
                    row.update(normalized_mse=mse, normalized_mae=mae,
                               raw_mse=mse*self.scale[j]**2, raw_mae=mae*self.scale[j],
                               raw_rmse=mse**.5*self.scale[j],
                               training_raw_std=self.std[j]*self.scale[j])
                    if self.std[j] > 1e-12:
                        row['standardized_mse'] = mse/self.std[j]**2
                        standardized.append(row['standardized_mse'])
                else:
                    cm = self.confusions[region][field['name']].double()
                    denom = cm.sum(0)+cm.sum(1)
                    row.update(accuracy=float(cm.diag().sum()/count),
                               macro_f1=float((2*cm.diag()/denom.clamp_min(1)).mean()),
                               confusion_matrix=cm.long().cpu().tolist())
                fields[field['name']] = row
            result[region] = {'mean_standardized_numeric_mse': sum(standardized)/len(standardized),
                              'per_field': fields}
        return result


def report(result, baseline, schema, out):
    names = {'interpolation': '历史回顾', 'forecast': '未来预测', 'overall': '整体'}
    lines = ['# ADNI：去掉 Jacobian 正则，第30轮临时测试', '',
             '本组保留α=0.1残差门控、统一速度场和0.01 weak prior，仅将Jacobian损失权重改为0。模型包含44,875个参数。按验证整体误差选择的第30轮检查点已复制到本目录，后续训练不会改变这份评测快照。', '',
             '测试使用原ADNI划分、全部非空历史前缀、batch=32和训练集min-max变换。历史目标从已吸收历史的截止点向过去回顾；未来目标从同一状态自由演化生成，不吸收未来观测。全部前向计算在物理CUDA 0完成，不执行参数更新。', '',
             '## 1. 测试集汇总', '',
             '| 区域 | MSE＋CE | 数值字段MSE之和 | 类别字段CE之和 | 数值标准化MSE均值 | 有效观测字段值数 |',
             '|---|---:|---:|---:|---:|---:|']
    for region, label in names.items():
        x = result['test'][region]; e = result['clinical_test'][region]
        lines.append(f"| {label} | {x['reconstruction_error']:.8f} | {x['sum_feature_mse']:.8f} | {x['sum_feature_cce']:.8f} | {e['mean_standardized_numeric_mse']:.8f} | {x['observed_scalars']:,} |")
    prior = result['test']['prior']; dyn = result['test']['dynamics']
    lines += ['', '数值标准化MSE按每个数值字段的训练集标准差换算，再对字段取平均。逐字段MAE和RMSE恢复为原始量纲；类别字段另报告准确率、宏平均F1和混淆矩阵。', '',
              f"校正前预测（weak prior诊断）：MSE＋CE={prior['reconstruction_error']:.8f}，数值MSE={prior['sum_feature_mse']:.8f}，类别CE={prior['sum_feature_cce']:.8f}。", '',
              f"速度场Jacobian平方Frobenius范数均值={dyn['jacobian_frobenius_squared']:.8f}，统计{dyn['states']:,}个生成状态。此处为评测诊断值，训练时该损失权重为0。", '',
              '## 2. 与完整主模型比较', '',
              '完整主模型同样为第30轮最佳检查点、44,875参数，保留0.01 Jacobian损失。下表各组采用相同测试划分和评测流程。', '',
              '| 测试指标 | 完整主模型 | 去掉Jacobian，第30轮 | 差值（消融−主模型） |', '|---|---:|---:|---:|']
    for region, label in names.items():
        main = baseline['test'][region]['reconstruction_error']; val = result['test'][region]['reconstruction_error']
        lines.append(f'| {label}MSE＋CE | {main:.8f} | {val:.8f} | {val-main:+.8f} |')
    lines += ['', '## 3. 全部字段指标', '',
              'MSE/CE列沿用正式评测定义。MAE、RMSE以各字段原始单位计；准确率和宏平均F1仅用于类别字段。CSV包含全部区域、字段及额外的标准化误差。', '',
              '[完整JSON](metrics.json) · [逐字段CSV](per_field_metrics.csv)', '']
    for region, label in names.items():
        lines += [f'### {label}', '', '| 字段 | 有效数 | MSE / CE | 原量纲MAE | 原量纲RMSE | 准确率 | 宏平均F1 |', '|---|---:|---:|---:|---:|---:|---:|']
        for field in schema:
            name = field['name']; x = result['test'][region]['per_field'][name]; e = result['clinical_test'][region]['per_field'][name]
            vals = [name, str(x['observations']), f"{x['loss']:.8f}"]
            vals += [f'{e[k]:.6f}' if k in e else '—' for k in ['raw_mae', 'raw_rmse', 'accuracy', 'macro_f1']]
            lines.append('| '+' | '.join(vals)+' |')
        lines.append('')
    lines += ['## 4. 快照与核验', '',
              f"- 测试患者：{result['test_patients']}；历史截止点任务数：{result['test_prefixes']}。",
              f"- 验证重载最大误差：{result['validation_reload_max_difference']:.3g}。",
              f"- 快照SHA256：`{result['checkpoint_sha256']}`。",
              f"- ODE：Tsit5，rtol={result['rtol']}，atol={result['atol']}；时间单位为年。",
              '- 评测独立进程与CUDA 0现有训练并行运行；模型参数未变化，optimizer.step()次数为0。', '']
    (out/'report.zh-CN.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True); args = parser.parse_args()
    folder = Path(args.run_dir); out = Path(args.output)
    if Path(sys.prefix).name != 'ode' or not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise RuntimeError('Assigned CUDA and ode are required; CPU fallback disabled')
    cfg = json.loads((folder/'launch_config.json').read_text())
    assert cfg['dataset'] == 'adni'
    manifest = json.loads((folder/'manifest.json').read_text())
    assert Path(inspect.getfile(make_model)).resolve().parent == folder/'source_snapshot'
    for base, hashes in [(folder/'source_snapshot', manifest['source_sha256']),
                         (Path(manifest['native_source']), manifest['native_source_hashes'])]:
        for filename, expected in hashes.items():
            assert sha(base/filename) == expected
    raw = Path(cfg['data']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == cfg['dataset_sha256']
    payload = pickle.loads(raw); schema = payload['schema']
    torch.set_num_threads(1); torch.cuda.set_device(0); seed_all(cfg['seed'])
    checkpoint_path = out/'checkpoint.pt'; checkpoint_hash = sha(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    assert saved['epoch'] == args.epoch
    assert saved['config']['jacobian_weight'] == 0 and saved['config']['prior_weight'] == .01
    model = make_model(schema, cfg, cfg['update_alpha']); model.load_state_dict(saved['state_dict']); model.eval()
    before = [p.detach().clone() for p in model.parameters()]
    import paper_odeae.train as native
    native.jacobian_regularization = jacobian_regularization
    splits = {s: [x for x in payload['sequences'] if x['split'] == s] for s in ['train', 'validation', 'test']}
    history = json.loads((out/'history_at_selection.json').read_text())
    expected = next(row['validation'] for row in history if row['epoch'] == args.epoch)
    start = time.monotonic()
    validation = evaluate(model, splits['validation'], schema, cfg['batch_size'])
    difference = max(abs(validation[r][k]-expected[r][k]) for r in ['interpolation', 'forecast', 'overall', 'prior']
                     for k in ['reconstruction_error', 'sum_feature_mse', 'sum_feature_cce'])
    assert difference < 1e-6, difference
    print(json.dumps({'validation_reloaded': True, 'epoch': args.epoch, 'max_difference': difference}), flush=True)
    clinical = ClinicalMetrics(schema, payload['metadata'], splits['train'])
    original_forward = model.forward
    def collect(*a, **kw):
        output = original_forward(*a, **kw); clinical.add(a[0], output); return output
    model.forward = collect
    test = evaluate(model, splits['test'], schema, cfg['batch_size'])
    model.forward = original_forward
    assert all(torch.equal(a, p) for a, p in zip(before, model.parameters()))
    result = {'status': 'temporary_checkpoint_evaluated', 'dataset': 'adni', 'checkpoint_epoch': args.epoch,
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
              'source_run': str(folder), 'validation': validation, 'test': test, 'clinical_test': clinical.results(test),
              'validation_reload_max_difference': difference, 'parameters': sum(p.numel() for p in model.parameters()),
              'physical_gpu': args.gpu, 'parameter_device': str(next(model.parameters()).device),
              'actual_tensor_devices': clinical.devices, 'gpu_name': torch.cuda.get_device_name(),
              'optimizer_steps': 0, 'parameters_unchanged': True, 'concurrent_with_training': True,
              'rtol': cfg['rtol'], 'atol': cfg['atol'], 'solver': 'Tsit5', 'time_unit': 'year',
              'batch_size': cfg['batch_size'], 'test_patients': len(splits['test']),
              'test_prefixes': sum(len(s['ages']) for s in splits['test']), 'dataset_sha256': cfg['dataset_sha256'],
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-start,
              'evaluation_script_sha256': sha(Path(__file__))}
    atomic(out/'metrics.json', result)
    columns = ['region','field','kind','observations','loss','normalized_mse','normalized_mae','raw_mse','raw_mae','raw_rmse','standardized_mse','accuracy','macro_f1']
    with (out/'per_field_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for region in ['interpolation','forecast','overall']:
            for field in schema:
                name = field['name']; row = {'region':region,'field':name,'kind':field['kind'], **test[region]['per_field'][name], **result['clinical_test'][region]['per_field'][name]}
                writer.writerow({k:row.get(k,'') for k in columns})
    baseline_path = Path(manifest['reference'])/'metrics.json'
    report(result, json.loads(baseline_path.read_text()), schema, out)
    atomic(folder/'temporary_test_epoch030.json', {'metrics_path': str(out/'metrics.json'),
           'report_path': str(out/'report.zh-CN.md'), 'checkpoint_epoch': args.epoch,
           'checkpoint_sha256': checkpoint_hash, 'temporary_evaluation': True})
    print(json.dumps({'complete':str(out), 'test':{k:{x:v for x,v in test[k].items() if x!='per_field'} for k in test}}), flush=True)


if __name__ == '__main__':
    main()
