"""Audit current ADNI/MIMIC caches and export aggregate dataset documentation; CPU data I/O only."""
import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/assets/datasets'
ADNI_SOURCE=Path('/data/SAMSUNG_backup/ADNI_data')
MIMIC_SOURCE=Path('/data/MIMIC/derived/latentode_iv31')
GROUPS={
 'adni':{
  '人口学':['sex','age','marital_status','education_years','handedness'],
  '诊断':['diagnosis'],
  '认知与神经心理':['mmse','cdrsb','avtot1']+[f'adas_q{i}' for i in range(1,14)]+['clock','digit_forward','digit_forward_length','digit_backward','digit_backward_length','category_animals','category_animals_perseveration','category_animals_intrusion','category_vegetables','category_vegetables_perseveration','category_vegetables_intrusion'],
  'MRI海马体积变化':['hippocampal_volume_change_percent'],
  '淀粉样PET':['amyloid_cortical_suvr'],
  'tau PET':['tau_cortical_suvr'],
  'APOE遗传':['apoe4_count'],
  '脑脊液CSF':['csf_abeta42','csf_abeta40','csf_ptau']},
 'mimic':{
  '生命体征与体重':['heart_rate','sbp','dbp','map','respiratory_rate','spo2','temperature','fio2','weight'],
  '化验':['glucose','urea_nitrogen','creatinine','sodium','potassium','chloride','bicarbonate','calcium','magnesium','phosphate','wbc','hemoglobin','hematocrit','platelets','lactate','blood_gas_ph','blood_gas_pco2','blood_gas_po2'],
  'GCS意识状态':['gcs_eye','gcs_motor','gcs_verbal'],
  '性别':['sex']}}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={'adni':ROOT/'paper_odeae/cache/local_available.pkl','mimic':MIMIC_SOURCE/'dataset.pkl'}
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'scope':'current cached common-task datasets',
             'documentation_builder_sha256':sha(__file__),'datasets':{}}
    cohort_rows=[];coverage_rows=[];observation_rows=[]
    original=pd.read_csv(ADNI_SOURCE/'ADNI_multimodal_timeline/indexes/patient_splits.csv')
    assert not original.subject_id.duplicated().any();adni_assignment=dict(zip(original.subject_id,original.split))
    for dataset,path in paths.items():
        payload=pickle.loads(path.read_bytes());schema=payload['schema'];seq=payload['sequences'];meta=payload['metadata'];columns={field['name']:j for j,field in enumerate(schema)}
        patients=[s['subject_id'] for s in seq];assert len(patients)==len(set(patients))
        groups=GROUPS[dataset];flat=[field for fields in groups.values() for field in fields]
        assert set(flat)==set(columns) and len(flat)==len(columns)
        if dataset=='adni':assert all(adni_assignment[s['subject_id']]==s['split'] for s in seq)
        else:
            for s in seq:
                u=int(hashlib.sha256(f'mimiciv31/seed0/{int(s["subject_id"])}'.encode()).hexdigest()[:16],16)/2**64
                assert s['split']==('train' if u<.7 else 'validation' if u<.85 else 'test')
        partitions={split:{s['subject_id'] for s in seq if s['split']==split} for split in ['train','validation','test']}
        assert not (partitions['train']&partitions['validation'] or partitions['train']&partitions['test'] or partitions['validation']&partitions['test'])
        dataset_summary={'cache':str(path),'cache_sha256':sha(path),'patients':len(seq),'fields':len(schema),
                         'numeric_fields':sum(f['kind']=='numerical' for f in schema),'categorical_fields':sum(f['kind']=='categorical' for f in schema),
                         'expanded_dimensions':sum(f.get('classes',1) for f in schema),'normalization':meta['normalization'],
                         'splits':{},'modalities':[],'normalization_fit':'train only','patient_partition_check':'passed'}
        hardware=ROOT/('paper_odeae/results/latent2x_loss_ablation/both/seed_0/hardware.json' if dataset=='adni' else 'mimic_experiment/results/latent2x_both_cuda/seed_0/hardware.json')
        assert json.loads(hardware.read_text())['dataset_sha256']==dataset_summary['cache_sha256']
        dataset_summary['training_artifact_cache_hash_verified']=True
        field_rows=[]
        for split in ['train','validation','test','all']:
            selected=[s for s in seq if split=='all' or s['split']==split]
            n=np.array([len(s['ages']) for s in selected]);span=np.array([float(s['ages'][-1]-s['ages'][0]) for s in selected])
            row={'dataset':dataset,'split':split,'patients':len(selected),'observation_rows':int(n.sum()),'visits_min':int(n.min()),'visits_median':float(np.median(n)),'visits_max':int(n.max()),'visits_mean':float(n.mean()),
                 'timestamp_span_median':float(np.median(span)),'timestamp_span_min':float(span.min()),'timestamp_span_max':float(span.max()),'time_unit':'year' if dataset=='adni' else 'hour'}
            cohort_rows.append(row);dataset_summary['splits'][split]=row
            observation_rows.extend({'dataset':dataset,'split':split,'observed_timepoints':int(count),'patients':int(value)} for count,value in sorted(Counter(n).items()))
            for name,fields in groups.items():
                ids=[columns[field] for field in fields];counts=np.array([int(s['mask'][:,ids].any(1).sum()) for s in selected])
                coverage={'dataset':dataset,'split':split,'modality':name,'fields':len(fields),'patients':len(selected),
                          'patients_with_any_observation':int((counts>0).sum()),'patients_with_at_least_two_timepoints':int((counts>=2).sum()),'timepoints_with_any_observation':int(counts.sum())}
                coverage_rows.append(coverage)
                if split=='all':dataset_summary['modalities'].append(coverage)
        for j,field in enumerate(schema):
            row={'field':field['name'],'kind':field['kind'],'classes':field.get('classes',''),'group':next(name for name,fields in groups.items() if field['name'] in fields),
                 'unit':field.get('unit',''),'offset':meta['offset'][j],'scale':meta['scale'][j]}
            for split in ['train','validation','test']:
                selected=[s for s in seq if s['split']==split];n=sum(int(s['mask'][:,j].sum()) for s in selected);row[split+'_observations']=n
                row[split+'_retained_row_coverage']=n/sum(len(s['ages']) for s in selected)
            field_rows.append(row)
        write_csv(OUT/f'{dataset}_features.csv',field_rows)
        (OUT/f'{dataset}_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
        if dataset=='adni':
            dataset_summary.update(original_split_counts=original.split.value_counts().to_dict(),split_file=str(ADNI_SOURCE/'ADNI_multimodal_timeline/indexes/patient_splits.csv'),
                                   split_file_sha256=sha(ADNI_SOURCE/'ADNI_multimodal_timeline/indexes/patient_splits.csv'),excluded=meta['excluded'],source_files=meta['source_files'])
        else:dataset_summary.update(revision=meta['dataset_revision'],selection_audit=json.loads((MIMIC_SOURCE/'cohort_selection.json').read_text()),
                                  excluded_observation_support=meta['excluded_insufficient_observation_support'],excluded_chronology=meta['excluded_invalid_48h_chronology_after_observation_filter'])
        summary['datasets'][dataset]=dataset_summary
    shutil.copy2(MIMIC_SOURCE/'verified_item_mapping.csv',OUT/'mimic_item_mapping.csv')
    write_csv(OUT/'cohort_summary.csv',cohort_rows);write_csv(OUT/'modality_coverage.csv',coverage_rows);write_csv(OUT/'observation_counts.csv',observation_rows)
    (OUT/'datasets_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:{'patients':value['patients'],'fields':value['fields'],'rows':value['splits']['all']['observation_rows'],'modalities':value['modalities']} for key,value in summary['datasets'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
