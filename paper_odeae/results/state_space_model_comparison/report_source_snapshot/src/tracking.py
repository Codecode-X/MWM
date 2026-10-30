"""Shared SwanLab monitoring for every maintained training entry point.

Default local mode needs no login. Configure via SWANLAB_* environment variables.
Only scalar metrics and explicit run configuration are passed to the SDK.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import warnings


DETAIL_KEYS = {'per_feature', 'per_field', 'patient_ids', 'subject_ids', 'predictions'}
SECRET = re.compile(r'(api.?key|token|password|secret|credential)', re.IGNORECASE)


def scalar_metrics(values, prefix=''):
    """Flatten metric dictionaries while leaving patient records out of logs."""
    result = {}
    for key, value in values.items():
        if key in DETAIL_KEYS or SECRET.search(str(key)):
            continue
        name = f'{prefix}/{key}' if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, Real):
            if math.isfinite(float(value)):
                result[name] = float(value)
        elif hasattr(value, 'numel') and value.numel() == 1:
            number = float(value.detach())
            if math.isfinite(number):
                result[name] = number
    return result


def public_config(config):
    return {str(k): public_config(v) if isinstance(v, Mapping) else v
            for k,v in config.items() if not SECRET.search(str(k)) and k not in DETAIL_KEYS}


def project_name():
    value = os.environ.get('SWANLAB_PROJECT', os.environ.get('SWANLAB_PROJECT__NAME', 'LatentODE'))
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    return parsed.get('name', 'LatentODE') if isinstance(parsed, dict) else str(parsed)


@contextmanager
def sdk_project_environment():
    # SwanLab 0.10 Settings treats SWANLAB_PROJECT as a nested JSON object.
    # Accept the customary plain-name variable at our training entry points.
    original = os.environ.get('SWANLAB_PROJECT')
    if original is not None:
        os.environ['SWANLAB_PROJECT'] = json.dumps({'name': project_name()})
    try:
        yield
    finally:
        if original is not None:
            os.environ['SWANLAB_PROJECT'] = original


class SwanLabTracker:
    def __init__(self, output, config, family, model=None, seed=None, runtime=None):
        self.output = Path(output)
        self.config = public_config(dict(config))
        self.family, self.model, self.seed = family, model, seed
        self.runtime = runtime or {}
        self.mode = os.environ.get('SWANLAB_MODE', 'local').lower()
        if self.mode not in {'local', 'offline', 'online', 'disabled'}:
            raise ValueError('SWANLAB_MODE must be local, offline, online, or disabled')
        self.every = int(os.environ.get('SWANLAB_LOG_EVERY', '20'))
        if self.every < 1:
            raise ValueError('SWANLAB_LOG_EVERY must be positive')
        self.step, self.run = 0, None
        self.logdir = Path(os.environ.get('SWANLAB_LOG_DIR', Path(__file__).resolve().parents[1]/'swanlog')).resolve()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.info = {'mode':self.mode, 'status':'running', 'family':self.family,
                     'training_seed':self.seed, 'log_dir':str(self.logdir),
                     'physical_gpu':self.runtime.get('physical_gpu', os.environ.get('CUDA_VISIBLE_DEVICES'))}
        if self.mode != 'disabled':
            try:
                import swanlab
            except ImportError as error:
                raise RuntimeError('SwanLab is required: install requirements.txt, or explicitly set SWANLAB_MODE=disabled') from error
            model_info = dict(self.runtime)
            if self.model is not None:
                parameter = next(self.model.parameters())
                device_name = str(parameter.device)
                if device_name.startswith('cuda'):
                    index = int(device_name.partition(':')[2] or '0')
                    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                    self.info['physical_gpu'] = visible.split(',')[index].strip() if visible else str(index)
                model_info.update(parameter_device=str(parameter.device),
                                  model_parameters=sum(p.numel() for p in self.model.parameters()))
            config = {**self.config, 'tracking_family':self.family,
                      'training_seed':self.seed, 'physical_gpu':self.info['physical_gpu'],
                      **model_info}
            self.info.update(parameter_device=model_info.get('parameter_device'),
                             model_parameters=model_info.get('model_parameters', model_info.get('parameters')))
            name = os.environ.get('SWANLAB_EXPERIMENT_NAME') or '-'.join(
                [self.family, str(self.seed), self.output.parent.name, self.output.name])
            if self.mode == 'local' and swanlab.__version__.startswith('0.10.'):
                from .swanlab_local import install_local_serialization
                install_local_serialization()
            project = project_name()
            with sdk_project_environment():
                self.run = swanlab.init(
                    project=project,
                    workspace=os.environ.get('SWANLAB_WORKSPACE'),
                    name=name, group=self.family, job_type='train', config=config,
                    log_dir=str(self.logdir), mode=self.mode,
                    settings=swanlab.Settings(interactive=False,
                        terminal=swanlab.Settings.Terminal(proxy_type='none')))
            self.info.update(run_id=self.run.id, experiment_name=name, sdk_version=swanlab.__version__)
            if self.mode == 'online':
                self.info['url'] = self.run.url
        self._save()
        return self

    def _save(self):
        (self.output/'swanlab.json').write_text(json.dumps(self.info, indent=2))

    def log(self, metrics):
        if self.run is not None:
            values = scalar_metrics(metrics)
            if values:
                self.run.log(values, step=self.step)

    def batch(self, loss, lr, **components):
        self.step += 1
        if self.step == 1 or self.step % self.every == 0:
            self.log({'batch':{'loss':loss, 'learning_rate':lr, **components}, 'optimizer_step':self.step})

    def epoch(self, epoch, train, validation, lr, **extra):
        self.log({'epoch':epoch, 'optimizer_step':self.step,
                  'train':{**train, 'learning_rate':lr}, 'validation':validation, **extra})

    def final(self, results):
        self.log({'final':results})

    def __exit__(self, exc_type, exc, traceback):
        state = 'success' if exc_type is None else 'aborted' if issubclass(exc_type, KeyboardInterrupt) else 'crashed'
        self.info.update(status=state, optimizer_steps=self.step)
        try:
            if self.run is not None:
                self.run.finish(state=state, error=None if exc is None else f'{exc_type.__name__}: {exc}')
        except Exception as error:
            if exc_type is None:
                self.info['status'] = 'tracking_error'
                raise
            warnings.warn(f'SwanLab finish failed while preserving training error: {error}', RuntimeWarning)
        finally:
            self._save()
        return False
