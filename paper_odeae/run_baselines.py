"""Four-GPU single-seed baseline comparison using the unchanged ODE-AE task."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
JOBS=[('native','latent_ode',0),('native','ode_rnn',1),
      ('small','latent_ode',2),('small','ode_rnn',3)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='paper_odeae/results/baselines_full')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/'paper_odeae/configs/local_available.json').read_text())
    dataset_sha=hashlib.sha256((ROOT/original['data']).read_bytes()).hexdigest()
    def launch(job):
        variant,model,gpu=job
        output=root/variant/model/'seed_0';output.mkdir(parents=True,exist_ok=True)
        cfg={**original,'model':model,'seed':0,'output':str(output),
             'comparison_variant':variant,'encoder_hidden':64 if variant=='native' else 16,
             'latent':32 if variant=='native' else 6,'ode_width':64 if variant=='native' else 16}
        config_path=root/'configs'/f'{variant}_{model}.json'
        config_path.parent.mkdir(exist_ok=True)
        expected={**cfg,'smoke':args.smoke}
        if args.smoke:expected.update(max_epochs=1,min_epochs=0,patience=1,batch_size=4)
        if (output/'metrics.json').exists():
            if json.loads((output/'config.json').read_text())!=expected:
                raise ValueError(f'completed configuration differs: {output}')
            if json.loads((output/'hardware.json').read_text())['dataset_sha256']!=dataset_sha:
                raise ValueError('dataset changed; use a new comparison directory')
            return
        if (output/'history.json').exists():
            raise ValueError(f'incomplete previous run exists; preserve it and choose a new output: {output}')
        config_path.write_text(json.dumps(cfg,indent=2)+'\n')
        cmd=[sys.executable,'-u','-m','paper_odeae.train','--config',str(config_path)]
        if args.smoke:cmd.append('--smoke')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 SWANLAB_MODE=os.environ.get('SWANLAB_MODE','local'))
        print(f'physical GPU {gpu}: {variant}/{model}, seed 0',flush=True)
        with (output/'train.log').open('w') as stream:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    (root/'launch.json').write_text(json.dumps({'pid':os.getpid(),'smoke':args.smoke,
        'dataset_sha256':dataset_sha,'jobs':[{'variant':v,'model':m,'physical_gpu':g} for v,m,g in JOBS]},indent=2)+'\n')
    def report():
        subprocess.run([sys.executable,'-m','paper_odeae.report_baselines','--root',str(root)],cwd=ROOT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(launch,job) for job in JOBS}
        while pending:
            done,pending=wait(pending,timeout=60)
            for future in done:future.result()
            report()
    report()


if __name__=='__main__':main()
