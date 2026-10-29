"""Queue age/zero forward-encoder ablations after the four-GPU control."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        text=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return None if text[0]=='Z' else text[19]
    except FileNotFoundError:
        return None


def write_json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',nargs='+',default=[
        'paper_odeae/configs/deterministic_forward_age_from22545.json',
        'paper_odeae/configs/deterministic_forward_from22545.json'])
    parser.add_argument('--control',default='paper_odeae/results/capacity_matched/deterministic_22545/seed_0')
    args=parser.parse_args();control=Path(args.control).resolve()
    configs=[(name,json.loads(Path(name).read_text())) for name in args.configs]
    root=ROOT/'paper_odeae/results/direction_ablation';root.mkdir(parents=True,exist_ok=True)
    launch=json.loads((control/'launch.json').read_text());pid=launch['pid'];identity=process_identity(pid)
    old=json.loads((root/'queue.json').read_text()) if (root/'queue.json').exists() else {}
    if old.get('pid') and process_identity(old['pid'])==old.get('process_start_time'):
        raise RuntimeError('ablation queue already running')
    state={'pid':os.getpid(),'process_start_time':process_identity(os.getpid()),'status':'queued',
           'control':str(control),'control_pid':pid,'control_process_start_time':identity,
           'requested_at':datetime.now(timezone.utc).isoformat(),
           'jobs':[{'config':str(Path(name).resolve()),'output':cfg['output'],
                    'encoder_initialization':cfg['encoder_initialization'],
                    'target_parameters':cfg['target_parameters'],'status':'queued'} for name,cfg in configs],
           'reason':'wait for the four-GPU control to release GPUs 0,1,2,3'}
    def save(status):
        state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat())
        write_json(root/'queue.json',state)
        for job in state['jobs']:
            path=Path(job['output']);path.mkdir(parents=True,exist_ok=True)
            write_json(path/'queue.json',{**job,'pid':state['pid'],'updated_at':state['updated_at']})
    def report():
        subprocess.run([sys.executable,'-m','paper_odeae.report_baselines'],cwd=ROOT,check=True)
    def run(config,arguments):
        subprocess.run([sys.executable,'-u','-m','paper_odeae.run_capacity','--config',config,*arguments],cwd=ROOT,check=True)
    active=None
    try:
        save('queued');report()
        while identity is not None and process_identity(pid)==identity:
            time.sleep(30);report()
        if not (control/'metrics.json').exists():
            raise RuntimeError('control stopped without final metrics; ablations were not started')
        tracking=json.loads((control/'swanlab.json').read_text())
        if tracking['status']!='success':raise RuntimeError('control did not finish successfully')
        control_cfg=json.loads((control/'config.json').read_text())
        expected_hash=json.loads((control/'hardware.json').read_text())['dataset_sha256']
        if hashlib.sha256((ROOT/control_cfg['data']).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('dataset changed since the control experiment')
        settings=['latent','ode_width','mlp_width','lr','lr_decay','lr_decay_steps','mse_weight','cce_weight',
                  'kl_weight','l2_weight','max_epochs','min_epochs','patience','batch_size','input_dropout',
                  'normalization','rtol','atol','seed','model','data','scope']
        if control_cfg['distributed_world_size']!=4:raise ValueError('control must use the same four-GPU layout')
        state['matching_settings']=settings
        for index,(name,cfg) in enumerate(configs):
            active=state['jobs'][index]
            if any(control_cfg[k]!=cfg[k] for k in settings):raise ValueError('ablation settings differ from control')
            if (Path(cfg['output'])/'metrics.json').exists():
                raise ValueError('completed ablation exists; inspect results before changing the queue')
            active['status']='checking';save('running');report()
            checks=root/'checks'/cfg['encoder_initialization']
            snapshot=Path(cfg['output'])/'queued_config.json';write_json(snapshot,cfg)
            run(str(snapshot),['--check-step','--output',str(checks/'distributed_step')])
            run(str(snapshot),['--smoke','--output',str(checks/'smoke')])
            active['status']='running';save('running');report()
            run(str(snapshot),[])
            active['status']='complete';save('running');report()
        save('complete');report()
    except BaseException as error:
        state['error']=f'{type(error).__name__}: {error}'
        if active is not None:active['status']='failed'
        save('failed');report();raise


if __name__=='__main__':main()
