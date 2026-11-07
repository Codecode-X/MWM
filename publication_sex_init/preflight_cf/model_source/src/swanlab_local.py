"""Serialize SwanLab 0.10 local dashboard callbacks across training processes.

The legacy dashboard creates shared project charts with a check-then-insert.
Simultaneous first logs can otherwise leave a partially registered chart.
This process-local adapter locks only dashboard writes, never model execution.
"""
from functools import wraps
from pathlib import Path
import threading
import fcntl

_LOCK = threading.RLock()


def install_local_serialization():
    from swanlab.deprecated.local import LocalCallbacker
    if getattr(LocalCallbacker, '_latentode_serialized', False):
        return
    def guard(name, original):
        @wraps(original)
        def wrapped(self, *args, **kwargs):
            run_dir = args[0] if name == 'on_run_initialized' else self._run_dir
            if run_dir is None:
                return original(self,*args,**kwargs)
            directory = Path(run_dir).parent
            directory.mkdir(parents=True,exist_ok=True)
            with _LOCK, (directory/'.dashboard-write.lock').open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX)
                try:
                    return original(self,*args,**kwargs)
                finally:
                    fcntl.flock(handle,fcntl.LOCK_UN)
        return wrapped
    for name in ['on_run_initialized','on_scalar_flush','on_media_flush','on_log_flush','on_run_finished']:
        setattr(LocalCallbacker,name,guard(name,getattr(LocalCallbacker,name)))
    LocalCallbacker._latentode_serialized = True
