"""Validate completed artifacts and report references without executing models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from .captions import enrich


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'publication_sex_init'
DOC=ROOT/'reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.md'


def read(path):
    return json.loads(Path(path).read_text())


def validate():
    queue=read(OUT/'queue.json');manifest=read(OUT/'manifest.json')
    assert len(manifest['jobs'])==89 and queue['status']=='complete'
    assert all(queue['jobs'][j['id']]['status']=='complete' for j in manifest['jobs'])
    training=[]
    for job in manifest['jobs']:
        cfg=read(job['config']);assert cfg['physical_gpu'] in range(4)
        result=Path(job['output'])/job.get('completion_file','metrics.json')
        assert result.exists(),job['id']
        if job['module']=='publication_sex_init.train':
            metrics=read(result)
            assert metrics['epochs_run']<=150 and cfg['max_epochs']<=150
            assert cfg['gradient_accumulation_steps']==1
            training.append(job['id'])
    for name in ['adni/jacobian_utility','mimic/jacobian_utility','mimic/stability']:
        folder=OUT/'analysis'/name;done=read(folder/'complete.json')
        assert done['parameter_updates']==0 and done['device']=='cuda:0'
        if 'jacobian' in name:
            for item in read(folder/'metrics.json').values():
                assert max(item['convergence'].values())<1e-4 and item['autograd_relative_error']<1e-4
                assert hashlib.sha256(Path(item['checkpoint']).read_bytes()).hexdigest()==item['checkpoint_sha256']
    snapshots={};excluded=[]
    for version in ['v1','v5','v6','v7']:
        folder=OUT/'source_snapshot'/version;hashes=read(folder/'source_hashes.json')
        source={k:v for k,v in hashes.items() if '__pycache__' not in Path(k).parts and not k.endswith('.pyc')}
        excluded.extend(f'{version}/{k}' for k in hashes if k not in source)
        bad=[name for name,value in source.items() if not (folder/name).exists() or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=value]
        assert not bad,(version,bad)
        snapshots[version]=len(source)
    text=DOC.read_text();links=[]
    for _,target in re.findall(r'!?\[([^\]]*)\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'):
            continue
        assert (DOC.parent/target.strip('<>').split('#')[0]).exists(),target
        links.append(target)
    figures=re.findall(r'^!\[图 (\d+)：',text,re.M)
    captions=re.findall(r'^\*\*图 (\d+)｜',text,re.M)
    assert figures==captions==list(map(str,range(1,61)))
    lines=text.splitlines();count=0;i=0
    while i<len(lines):
        if lines[i].startswith('|') and i+1<len(lines) and re.match(r'^\|[ :\-|]+\|$',lines[i+1]):
            width=len(lines[i].split('|'));count+=1;i+=2
            while i<len(lines) and lines[i].startswith('|'):
                assert len(lines[i].split('|'))==width
                i+=1
            assert '<!-- publication-caption -->' in '\n'.join(lines[i:i+5])
        else:
            i+=1
    assert count==34

    def table(number):
        found=re.search(rf'表 {number}：[^\n]*\n\n(\|[^\n]*\n(?:\|[^\n]*\n)*)',text)
        return [[s.strip() for s in line.split('|')[1:-1]] for line in found.group(1).strip().splitlines()[2:]]

    for row in table(26):
        dataset,name=row[:2]
        name='M0（性别 INIT）' if name=='M0' else 'HealthFlux-KL（KL=1）' if dataset=='ADNI' and name=='HealthFlux-KL' else 'HealthFlux-KL（KL=0.01）' if name=='HealthFlux-KL' else name
        matches=[r for r in table(1 if dataset=='ADNI' else 3) if r[0]==name]
        assert len(matches)==1 and row[7]==matches[0][5],row
    first=enrich(text,DOC,OUT);second=enrich(first,DOC,OUT)
    assert first==second,'Caption rendering must be idempotent'
    record={'status':'passed','updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'complete_jobs':89,'failed_jobs':0,'new_clinical_training_jobs_within_150_epochs':len(training),
            'repaired_GPU_analyses':3,'diagnostic_parameter_updates':0,'figures_with_captions':60,
            'tables_with_captions':34,'valid_local_links':len(links),
            'verified_snapshot_source_files':snapshots,'excluded_runtime_bytecode':excluded,
            'cost_table_matches_primary_test_scores':True,'caption_render_idempotent':True,
            'model_execution_performed':False}
    (OUT/'validation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    validate()
