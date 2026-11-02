"""Frozen MedicalNet MRI embeddings on CUDA; CPU only decodes image files.

One earliest T1 acquisition per patient, selected without diagnoses or outcomes.
The embedding is inserted at its acquisition date, never broadcast backwards.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .adni_data import DEFAULT_ROOT, CACHE
from .vendor.medicalnet_resnet import resnet10


def choose_scans(root, output):
    index = pd.read_csv(root / 'indexes/image_series.csv')
    index = index[(index.modality == 'mri') & (index.integrity_status == 'complete')]
    selected, excluded = [], []
    for subject, rows in index.sort_values(['date', 'asset_id']).groupby('subject_id', sort=True):
        for row in rows.to_dict('records'):
            manifest = pd.read_csv(root / row['directory'] / 'files.csv')
            description = str(manifest.archive_entry.iloc[0]).lower()
            # Restrict to structural T1, excluding localizers and derived maps.
            if not any(k in description for k in ['mprage', 'mp-rage', 'spgr', 't1w', 't1_weighted']):
                continue
            row['files'] = manifest.stored_file.tolist()
            row['description'] = description.split('/')[2] if '/' in description else description
            selected.append(row)
            break
        else:
            excluded.append(subject)
    output.write_text(json.dumps({'scans': selected, 'no_recognized_T1': excluded}, indent=2))
    print(json.dumps({'selected': len(selected), 'excluded': len(excluded)}), flush=True)


class Volumes(Dataset):
    def __init__(self, root, scans):
        self.root, self.scans = root, scans

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        row = self.scans[i]
        try:
            directory = self.root / row['directory']
            paths = [directory / f for f in row['files']]
            nii = [p for p in paths if str(p).endswith(('.nii', '.nii.gz'))]
            if nii:
                img = nib.as_closest_canonical(nib.load(nii[0]))
                data = img.get_fdata(dtype=np.float32)
                if data.ndim != 3:
                    raise ValueError('not a 3D T1 volume')
                volume = data.transpose(2, 1, 0)
            else:
                tags = ['PixelData', 'ImageOrientationPatient', 'ImagePositionPatient',
                        'PixelSpacing', 'RescaleSlope', 'RescaleIntercept', 'Rows', 'Columns',
                        'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                        'SamplesPerPixel', 'PhotometricInterpretation', 'PlanarConfiguration',
                        'NumberOfFrames']
                dcms = [pydicom.dcmread(p, specific_tags=tags) for p in paths if p.suffix.lower() == '.dcm']
                dcms = [d for d in dcms if hasattr(d, 'PixelData')]
                if len(dcms) < 32:
                    raise ValueError('too few DICOM slices')
                orient = np.array(dcms[0].ImageOrientationPatient, float)
                normal = np.cross(orient[:3], orient[3:])
                positions = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in dcms])
                order = np.argsort(positions)
                if np.diff(positions[order]).min() < 1e-4:
                    raise ValueError('duplicate slice positions / multiple volumes')
                volume = np.stack([dcms[j].pixel_array.astype(np.float32) *
                                   float(getattr(dcms[j], 'RescaleSlope', 1)) +
                                   float(getattr(dcms[j], 'RescaleIntercept', 0)) for j in order])
                # Build LPS affine from DICOM axes; canonical RAS orientation
                # removes sagittal/axial and left/right acquisition differences.
                first = dcms[order[0]]
                spacing = np.array(first.PixelSpacing, float)
                affine = np.eye(4)
                affine[:3, 0] = orient[:3] * spacing[1]
                affine[:3, 1] = orient[3:] * spacing[0]
                affine[:3, 2] = normal * np.median(np.diff(positions[order]))
                affine[:3, 3] = np.array(first.ImagePositionPatient, float)
                affine = np.diag([-1., -1., 1., 1.]) @ affine
                img = nib.as_closest_canonical(nib.Nifti1Image(volume.transpose(2, 1, 0), affine))
                volume = img.get_fdata(dtype=np.float32).transpose(2, 1, 0)
            if not np.isfinite(volume).all():
                raise ValueError('nonfinite image')
            # Foreground bounding box, nonzero normalization. Resizing is a
            # deterministic adaptation of the MedicalNet evaluation pipeline.
            foreground = volume > 0
            coords = np.where(foreground)
            if len(coords[0]) < 100:
                raise ValueError('empty image')
            box = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
            volume = volume[box].copy()
            foreground = volume > 0
            mean, std = volume[foreground].mean(), volume[foreground].std()
            if std < 1e-6:
                raise ValueError('constant image')
            volume = np.where(foreground, (volume - mean) / std, 0).astype(np.float32)
            # CPU interpolation is data preparation, not encoder inference.
            from scipy.ndimage import zoom
            volume = zoom(volume, np.array([96, 96, 96]) / volume.shape, order=1)
            return i, torch.from_numpy(volume[None].copy()), ''
        except Exception as e:
            return i, torch.zeros(1, 96, 96, 96), f'{type(e).__name__}: {e}'


def extract(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU encoder fallback is forbidden')
    torch.set_num_threads(1)
    root, cache = Path(args.root), Path(args.cache)
    scans = json.loads((cache / 'mri_manifest.json').read_text())['scans'][args.shard::4]
    with (cache / 'prepared.pkl').open('rb') as f:
        eligible = {s['patient_id'] for s in pickle.load(f)['sequences']}
    scans = [s for s in scans if s['subject_id'] in eligible]
    output = cache / 'embeddings'
    output.mkdir(parents=True, exist_ok=True)
    scans = [s for s in scans if not (output / f"{s['subject_id']}.npz").exists()]
    model = resnet10(sample_input_D=96, sample_input_H=96, sample_input_W=96,
                     num_seg_classes=2, shortcut_type='B', no_cuda=False).cuda()
    checkpoint = torch.load(args.weights, map_location='cuda', weights_only=True)
    weights = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    result = model.load_state_dict(weights, strict=False)
    if any(not k.startswith('conv_seg.') for k in result.missing_keys + result.unexpected_keys):
        raise RuntimeError(f'backbone weight mismatch: {result}')
    backbone = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                             model.layer1, model.layer2, model.layer3, model.layer4,
                             nn.AdaptiveAvgPool3d(1), nn.Flatten()).eval()
    backbone.requires_grad_(False)
    hardware = {'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'parameter_device': str(next(backbone.parameters()).device),
                'gpu_name': torch.cuda.get_device_name(), 'shard': args.shard,
                'weights_sha256': hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                'image_shape': [1, 96, 96, 96], 'pooling': 'global average, 512 dims',
                'missing_checkpoint_keys': result.missing_keys,
                'unexpected_checkpoint_keys': result.unexpected_keys}
    failures = []
    loader = DataLoader(Volumes(root, scans), batch_size=4, num_workers=args.workers)
    with torch.inference_mode():
        for step, (indices, volumes, errors) in enumerate(loader):
            good = [j for j, e in enumerate(errors) if not e]
            if good:
                x = volumes[good].cuda()
                embedding = backbone(x)
                if not torch.isfinite(embedding).all():
                    raise FloatingPointError('nonfinite frozen embeddings')
                hardware.update(input_device=str(x.device), output_device=str(embedding.device))
                for j, v in zip(good, embedding.cpu().numpy()):
                    row = scans[int(indices[j])]
                    np.savez_compressed(output / f"{row['subject_id']}.npz", embedding=v,
                                        date=row['date'], asset_id=row['asset_id'])
            for j, e in enumerate(errors):
                if e:
                    failures.append({'subject_id': scans[int(indices[j])]['subject_id'], 'error': e})
            if step % 10 == 0:
                print(json.dumps({'batches': step+1, 'scans': len(scans), 'failures': len(failures)}), flush=True)
                (cache / f'encoder_hardware_{args.shard}.json').write_text(json.dumps(hardware, indent=2))
                (cache / f'encoder_failures_{args.shard}.json').write_text(json.dumps(failures, indent=2))
    (cache / f'encoder_hardware_{args.shard}.json').write_text(json.dumps(hardware, indent=2))
    (cache / f'encoder_failures_{args.shard}.json').write_text(json.dumps(failures, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=DEFAULT_ROOT)
    p.add_argument('--cache', default=CACHE)
    p.add_argument('--select', action='store_true')
    p.add_argument('--shard', type=int, choices=range(4), default=0)
    p.add_argument('--weights', default='models/pretrained/resnet_10.pth')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.select:
        choose_scans(Path(args.root), Path(args.cache) / 'mri_manifest.json')
    else:
        extract(args)
