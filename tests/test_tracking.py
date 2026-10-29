"""Monitoring lifecycle checks use fake SDK objects, not CPU model experiments."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from src.tracking import SwanLabTracker,scalar_metrics,public_config
from src.track_history import read_json,follow


def sdk():
    run=Mock();run.id='fake-id';run.url='https://example.invalid/run'
    settings=Mock();settings.Terminal=Mock()
    return SimpleNamespace(init=Mock(return_value=run),Settings=settings,__version__='test'),run


class TrackingTests(unittest.TestCase):
    def test_nested_scalars_and_record_exclusion(self):
        got=scalar_metrics({'loss':1.,'validation':{'mse':.2,'per_feature':{'patient':{'mse':99}}},
                            'patient_ids':[1,2],'nan':float('nan'),'api_key':'secret'})
        self.assertEqual(got,{'loss':1.,'validation/mse':.2})
        self.assertEqual(public_config({'lr':.1,'api_key':'x','nested':{'password':'x','width':16}}),
                         {'lr':.1,'nested':{'width':16}})

    def test_sampling_epoch_and_finish(self):
        fake,run=sdk()
        with TemporaryDirectory() as path,patch.dict(os.environ,{'SWANLAB_MODE':'local','SWANLAB_LOG_EVERY':'3'}),patch.dict('sys.modules',{'swanlab':fake}):
            with SwanLabTracker(path,{'seed':1},'test',runtime={'parameter_device':'cuda:0'}) as tracker:
                for n in range(1,5):tracker.batch(n,.01)
                tracker.epoch(1,{'loss':2.},{'mse':.3},.01)
                tracker.final({'test':{'mse':.2}})
            self.assertEqual(run.log.call_count,4)
            self.assertEqual([c.kwargs['step'] for c in run.log.call_args_list],[1,3,4,4])
            run.finish.assert_called_once_with(state='success',error=None)
            self.assertEqual(json.loads((Path(path)/'swanlab.json').read_text())['optimizer_steps'],4)

    def test_exception_is_preserved_and_run_is_crashed(self):
        fake,run=sdk()
        with TemporaryDirectory() as path,patch.dict(os.environ,{'SWANLAB_MODE':'local'}),patch.dict('sys.modules',{'swanlab':fake}):
            with self.assertRaisesRegex(ValueError,'training failure'):
                with SwanLabTracker(path,{},'test'):
                    raise ValueError('training failure')
            run.finish.assert_called_once_with(state='crashed',error='ValueError: training failure')
            self.assertEqual(json.loads((Path(path)/'swanlab.json').read_text())['status'],'crashed')

    def test_disabled_requires_no_sdk(self):
        with TemporaryDirectory() as path,patch.dict(os.environ,{'SWANLAB_MODE':'disabled'}),patch.dict('sys.modules',{'swanlab':None}):
            with SwanLabTracker(path,{},'test') as tracker:
                tracker.batch(1.,.1);tracker.epoch(1,{}, {},.1)
            self.assertEqual(json.loads((Path(path)/'swanlab.json').read_text())['mode'],'disabled')

    def test_plain_project_environment_is_supported_and_restored(self):
        fake,run=sdk()
        with TemporaryDirectory() as path,patch.dict(os.environ,{'SWANLAB_MODE':'local','SWANLAB_PROJECT':'My-ADNI'}),patch.dict('sys.modules',{'swanlab':fake}):
            with SwanLabTracker(path,{},'test'):pass
            self.assertEqual(fake.init.call_args.kwargs['project'],'My-ADNI')
            self.assertEqual(os.environ['SWANLAB_PROJECT'],'My-ADNI')

    def test_physical_gpu_mapping_uses_actual_model_device(self):
        fake,run=sdk();parameter=Mock();parameter.device='cuda:1';parameter.numel.return_value=7
        model=Mock();model.parameters.side_effect=lambda:iter([parameter])
        with TemporaryDirectory() as path,patch.dict(os.environ,{'SWANLAB_MODE':'local','CUDA_VISIBLE_DEVICES':'3,2'}),patch.dict('sys.modules',{'swanlab':fake}):
            with SwanLabTracker(path,{},'test',model):pass
            info=json.loads((Path(path)/'swanlab.json').read_text())
            self.assertEqual(info['physical_gpu'],'2')
            self.assertEqual(info['parameter_device'],'cuda:1')

    def test_completed_history_import_and_partial_json(self):
        from unittest.mock import MagicMock
        with TemporaryDirectory() as path:
            directory=Path(path)
            (directory/'config.json').write_text(json.dumps({'seed':2,'lr':.001}))
            (directory/'history.json').write_text(json.dumps([
                {'epoch':1,'optimizer_steps':5,'training_loss':1.,'validation':{'mse':.4}},
                {'epoch':2,'optimizer_steps':10,'training_loss':.8,'validation':{'mse':.3}}]))
            (directory/'metrics.json').write_text(json.dumps({'test':{'mse':.2}}))
            tracker=Mock();tracker.step=0
            context=MagicMock();context.__enter__.return_value=tracker
            with patch('src.track_history.SwanLabTracker',return_value=context):
                follow(directory)
            self.assertEqual(tracker.epoch.call_count,2)
            tracker.final.assert_called_once_with({'test':{'mse':.2}})
            (directory/'history.json').write_text('{')
            self.assertIsNone(read_json(directory/'history.json'))

    def test_history_owner_failure_is_reported(self):
        from unittest.mock import MagicMock
        with TemporaryDirectory() as path:
            directory=Path(path)
            (directory/'config.json').write_text('{}')
            (directory/'history.json').write_text('[]')
            tracker=Mock();tracker.step=0
            context=MagicMock();context.__enter__.return_value=tracker
            with patch('src.track_history.SwanLabTracker',return_value=context),patch('src.track_history.process_start',side_effect=['start',None]):
                with self.assertRaisesRegex(RuntimeError,'ended without completed'):
                    follow(directory,pid=123)

    def test_sequential_runs_finish_separately(self):
        fake,run=sdk()
        with TemporaryDirectory() as path,patch.dict(os.environ,{'SWANLAB_MODE':'local'}),patch.dict('sys.modules',{'swanlab':fake}):
            for seed in [0,1]:
                with SwanLabTracker(Path(path)/str(seed),{},'test',seed=seed):pass
            self.assertEqual(fake.init.call_count,2)
            self.assertEqual(run.finish.call_count,2)


if __name__=='__main__':unittest.main()
