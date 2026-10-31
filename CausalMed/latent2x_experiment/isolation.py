"""A model namespace exposes only public data, frozen model source and its output."""
import os
from pathlib import Path

PYTHON=Path('/home/xjh/miniforge3/envs/ode/bin/python')


def command(source,public,output,swanlog,stage,physical_gpu=3,environment='ode'):
    for path in [source,public,output,swanlog]:
        if not Path(path).is_dir():raise ValueError(f'Missing directory {path}')
    python=Path('/home/xjh/miniforge3/envs')/environment/'bin/python'
    return ['bwrap','--die-with-parent','--unshare-pid','--unshare-net',
            '--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--symlink','usr/bin','/bin',
            '--ro-bind','/etc','/etc','--ro-bind','/sys','/sys','--dev-bind','/dev','/dev','--proc','/proc','--tmpfs','/tmp',
            '--ro-bind',str(python.parent.parent),str(python.parent.parent),
            '--ro-bind',str(Path(source).resolve()),'/workspace/source',
            '--ro-bind',str(Path(public).resolve()),'/workspace/public',
            '--bind',str(Path(output).resolve()),'/workspace/output','--bind',str(Path(swanlog).resolve()),'/workspace/swanlog',
            '--chdir','/workspace/source','--setenv','PYTHONPATH','/workspace/source',
            '--setenv','PYTHONNOUSERSITE','1','--setenv','PYTHONDONTWRITEBYTECODE','1',
            '--setenv','CUDA_VISIBLE_DEVICES',str(physical_gpu),'--setenv','OMP_NUM_THREADS','1','--setenv','MKL_NUM_THREADS','1',
            '--setenv','SWANLAB_MODE','local','--setenv','SWANLAB_LOG_DIR','/workspace/swanlog',
            str(python),'-u','-m','latent2x_experiment.train','--config','/workspace/output/launch_config.json','--stage',stage]
