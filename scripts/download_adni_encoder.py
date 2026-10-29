#!/usr/bin/env python3
"""Download the pinned MedicalNet checkpoint and verify its SHA256."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((root/'results/adni_cuda/research/encoder_provenance.json').read_text())
    destination=root/'models/pretrained'/provenance['weight']
    expected=provenance['weight_sha256']
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==expected:
        print(f'Already verified: {destination}')
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/{provenance['weight_repo']}/resolve/{provenance['weight_revision']}/{provenance['weight']}"
    temporary=destination.with_suffix('.download')
    with urlopen(url,timeout=120) as source, temporary.open('wb') as output:
        while chunk:=source.read(1024*1024):
            output.write(chunk)
    actual=hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError(f'Checkpoint checksum mismatch: {actual}')
    temporary.replace(destination)
    print(f'Verified {destination}: {actual}')


if __name__=='__main__':
    main()
