"""Wait for an existing comparison controller, then render the complete current report."""
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import time

from paper_odeae.scaling_study import identity
from .baseline_report import generate
from .report import read
from .prepare import atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output).resolve();state={'pid':os.getpid(),'start_time':identity(os.getpid()),'status':'waiting_for_training',
         'source_sha256':{str(path.name):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(__file__).with_name('baseline_report.py')]}}
    atomic(out/'final_report_queue.json',state)
    while True:
        queue=read(out/'queue.json',{})
        alive=bool(queue.get('pid') and identity(queue['pid'])==queue.get('start_time'))
        if not alive:break
        time.sleep(10)
    generate(out)
    rows=read(out/'comparison.json')['models'];complete=[row for row in rows if row['result']]
    if len(complete)==5:
        lines=['','## 9. 完成后的结果观察','',
               '下表按测试未来MSE＋交叉熵由低到高排序。数值部分与GCS交叉熵分别列出，便于查看总体误差来自哪些字段。','',
               '| 模型 | 未来总误差 | 数值MSE之和 | GCS交叉熵之和 | 数值标准化MSE |','|---|---:|---:|---:|---:|']
        for row in sorted(complete,key=lambda row:row['result']['test']['forecast']['reconstruction_error']):
            test=row['result']['test'];f=test['forecast']
            lines.append(f"| {row['name']} | {f['reconstruction_error']:.5f} | {f['sum_feature_mse']:.5f} | {f['sum_feature_cce']:.5f} | {test['mean_standardized_numeric_mse']:.5f} |")
        lines+=['','再逐字段检查哪些数值项目优于保持最近值。下表统计27个数值字段中，模型MAE更小的字段数。','',
                '| 模型 | MAE优于保持最近值的字段数 |','|---|---:|']
        for row in complete:
            test=row['result']['test'];wins=sum(value['mae']<test['carry_forward'][field]['mae'] for field,value in test['clinical_forecast'].items() if 'mae' in value)
            lines.append(f"| {row['name']} | {wins} / 27 |")
        with (out/'summary.md').open('a') as stream:stream.write('\n'.join(lines)+'\n')
    state.update(status='complete',completed_models=len(complete),finished_at=datetime.now(timezone.utc).isoformat());atomic(out/'final_report_queue.json',state)


if __name__=='__main__':main()
