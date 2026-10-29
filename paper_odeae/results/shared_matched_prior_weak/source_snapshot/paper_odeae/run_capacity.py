"""Run a single capacity-matched model across four physical CUDA GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='paper_odeae/configs/deterministic_22545.json')
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    parser.add_argument('--output');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--check-step',action='store_true')
    parser.add_argument('--no-baseline-report',action='store_true')
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or set(args.gpus)-{0,1,2,3}:
        raise ValueError('use unique physical GPUs from 0,1,2,3')
    cfg=json.loads(Path(args.config).read_text())
    if args.output:cfg['output']=args.output
    if (args.smoke or args.check_step) and not args.output:
        raise ValueError('checks require a separate output directory')
    output=Path(cfg['output']).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists() or (output/'history.json').exists():
        raise ValueError('output already contains results; choose a new directory')
    cfg['output']=str(output)
    config=output/'requested_config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    (output/'launch.json').write_text(json.dumps({'pid':os.getpid(),'physical_gpus':args.gpus,
        'world_size':len(args.gpus),'port':port,'smoke':args.smoke,'check_step':args.check_step},indent=2)+'\n')
    processes=[];logs=[]
    try:
        for rank,gpu in enumerate(args.gpus):
            command=[sys.executable,'-u','-m','paper_odeae.train_ddp','--config',str(config),
                '--rank',str(rank),'--world-size',str(len(args.gpus)),'--port',str(port)]
            if args.smoke:command.append('--smoke')
            if args.check_step:command.append('--check-step')
            stream=(output/f'rank_{rank}.log').open('w');logs.append(stream)
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT)
            processes.append(process)
            print(f'rank {rank}, physical GPU {gpu}, PID {process.pid}',flush=True)
        last_report=0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in [None,0] for p in processes):
                raise RuntimeError(f'DDP worker failed; inspect {output}/rank_*.log')
            if not args.no_baseline_report and not args.smoke and not args.check_step and time.monotonic()-last_report>=60:
                subprocess.run([sys.executable,'-m','paper_odeae.report_baselines'],cwd=ROOT,check=True)
                last_report=time.monotonic()
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('DDP worker failed')
        if not args.no_baseline_report and not args.smoke and not args.check_step:
            subprocess.run([sys.executable,'-m','paper_odeae.report_baselines'],cwd=ROOT,check=True)
        print('all DDP workers completed successfully',flush=True)
    finally:
        for process in processes:
            if process.poll() is None:process.send_signal(signal.SIGTERM)
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for stream in logs:stream.close()


if __name__=='__main__':main()
