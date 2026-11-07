"""Calendar-quarter data contract for ODE-AE, independent of the old ADNI cache."""
import argparse
from collections import defaultdict,Counter
from datetime import date,datetime
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parent
SCHEMA=json.loads((ROOT/'reference/features46.json').read_text())
DEFAULT_SOURCE='/data/SAMSUNG_backup/ADNI_data'


def num(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else None
    except (ValueError,TypeError):
        return None


def parse_date(v):
    if pd.isna(v):return None
    try:return date.fromisoformat(str(v)[:10])
    except ValueError:return None


def birth_date(row):
    v=str(row.get('PTDOB',''))
    for fmt in ['%m/%Y','%Y-%m-%d','%m-%d-%Y']:
        try:return datetime.strptime(v,fmt).date()
        except ValueError:pass
    v=str(row.get('PTDOBYY',''))[:4]
    return date(int(v),7,1) if v.isdigit() and 1850<int(v)<2020 else None


def raw_to_frame(source):
    source=Path(source)
    split=pd.read_csv(source/'ADNI_multimodal_timeline/indexes/patient_splits.csv')
    ids=set(split.subject_id)
    patient=pd.read_csv(source/'ADNI_multimodal_timeline/indexes/patients.csv')
    rid_to_id={int(r.rid):r.subject_id for r in patient.itertuples()}
    visits=defaultdict(lambda:{'dates':[],'values':defaultdict(list)})
    source_files=[]
    def read(pattern):
        paths=sorted(source.glob(pattern))
        if not paths:return pd.DataFrame()
        path=paths[0];source_files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return pd.read_csv(path,low_memory=False)
    def subject(r):
        if str(r.get('PTID')) in ids:return str(r['PTID'])
        rid=num(r.get('RID'))
        return rid_to_id.get(int(rid)) if rid is not None else None
    def add(row,values):
        pid=subject(row)
        if pid not in ids:return
        dated=[(i,d) for i,k in enumerate(['EXAMDATE','VISDATE','SCANDATE']) if (d:=parse_date(row.get(k))) is not None]
        if not dated:return
        code=row.get('VISCODE2')
        if pd.isna(code) or not str(code).strip():code=row.get('VISCODE')
        if pd.isna(code):code='date:'+str(min(dated)[1])
        phase=str(row.get('PHASE','unknown'))
        if phase in ['unknown','nan','None']:
            code=str(code)+':date:'+str(min(dated)[1])
        key=(pid,phase,str(code))
        visit=visits[key];visit['dates'].extend(dated)
        for k,v in values.items():
            if v is not None:visit['values'][k].append(v)
    def selected(r,mapping):
        out={}
        for target,field in mapping.items():
            v=num(r.get(field))
            if v is not None and v>=0:out[target]=v
        return out
    demo=read('Enrollment/PTDEMOG_*.csv')
    static={};births={}
    if not demo.empty:
        for pid,rows in demo.groupby('PTID'):
            if pid not in ids:continue
            r=rows.sort_values('VISDATE',na_position='last').iloc[0].to_dict()
            births[pid]=birth_date(r)
            st={}
            for k,field,maximum in [('sex','PTGENDER',2),('marital_status','PTMARRY',5),('handedness','PTHAND',3)]:
                v=num(r.get(field))
                if v is not None and 1<=v<=maximum:st[k]=int(v)-1
            v=num(r.get('PTEDUCAT'))
            if v is not None and 0<=v<=40:st['education_years']=v
            static[pid]=st
    genetics=read('Biospecimen/APOERES_*.csv')
    for r in genetics.to_dict('records'):
        pid=subject(r)
        alleles=re.findall(r'[234]',str(r.get('GENOTYPE','')))
        if pid in ids and len(alleles)==2:
            static.setdefault(pid,{})['apoe4_count']=alleles.count('4')
    tables=[('Neuropsychological/MMSE_*.csv',{'mmse':'MMSCORE'}),
            ('Neuropsychological/CDR_*.csv',{'cdrsb':'CDRSB'}),
            ('Neuropsychological/ADAS_*.csv',{f'adas_q{i}':f'Q{i}SCORE' for i in range(1,14)}),
            ('Neuropsychological/NEUROBAT_*.csv',dict(zip(
                ['avtot1','clock','digit_forward','digit_forward_length','digit_backward','digit_backward_length','category_animals','category_animals_perseveration','category_animals_intrusion','category_vegetables','category_vegetables_perseveration','category_vegetables_intrusion'],
                ['AVTOT1','CLOCKSCOR','DSPANFOR','DSPANFLTH','DSPANBAC','DSPANBLTH','CATANIMSC','CATANPERS','CATANINTR','CATVEGESC','CATVGPERS','CATVGINTR']))),
            ('Neuropsychological/*ECOG*.csv',{'ecog_self':'EcogPtTotal','ecog_partner':'EcogSPTotal'}),
            ('Neuropsychological/NPI_*.csv',{'npi':'NPITOTAL'}),
            ('MRI_PET_csv/PET/*UPENNBIOMK_ROCHE_ELECSYS*.csv',{'csf_abeta42':'ABETA42','csf_abeta40':'ABETA40','csf_ptau':'PTAU'})]
    for pattern,mapping in tables:
        for r in read(pattern).to_dict('records'):add(r,selected(r,mapping))
    dx=read('Enrollment/DXSUM_*.csv')
    for r in dx.to_dict('records'):
        d=num(r.get('DIAGNOSIS'))
        if d in [1,2,3]:d=int(d)-1
        else:
            codes=[i for i,k in enumerate(['DXNORM','DXMCI','DXAD']) if num(r.get(k))==1]
            d=codes[0] if len(codes)==1 else None
        add(r,{'diagnosis':d})
    for r in read('MRI_PET_csv/*UCSFFSX7*.csv').to_dict('records'):
        if any('fail' in str(r.get(k)).lower() for k in ['OVERALLQC','HIPPOQC']):continue
        left,right=num(r.get('ST29SV')),num(r.get('ST88SV'))
        if left is not None and right is not None and min(left,right)>0:
            add(r,{'hippocampal_volume_raw':left+right})
    for pattern,target in [('MRI_PET_csv/PET/*UCBERKELEY_AMY_6MM*.csv','amyloid_cortical_suvr'),('MRI_PET_csv/PET/*UCBERKELEY_TAU_6MM*.csv','tau_cortical_suvr')]:
        for r in read(pattern).to_dict('records'):
            if num(r.get('qc_flag'))!=2:continue
            weighted=[]
            for k in r:
                if k.startswith('CTX_') and not k.startswith(('CTX_LH_','CTX_RH_')) and k.endswith('_SUVR'):
                    a,w=num(r[k]),num(r.get(k[:-5]+'_VOLUME'))
                    if a is not None and w is not None and a>=0 and w>0:weighted.append((a,w))
            if weighted:add(r,{target:sum(a*w for a,w in weighted)/sum(w for _,w in weighted)})
    categories={f['name'] for f in SCHEMA if f['kind']=='categorical'}
    rows=[]
    for (pid,phase,code),visit in visits.items():
        d=min(visit['dates'])[1]
        values={k:(Counter(v).most_common(1)[0][0] if k in categories else float(np.mean(v))) for k,v in visit['values'].items() if v}
        if not values:continue
        values.update(subject_id=pid,date=d.isoformat())
        rows.append(values)
    return pd.DataFrame(rows),split,static,births,source_files


def prepare(source=DEFAULT_SOURCE,scope='local_available',canonical=None,split_file=None,normalization='train_minmax'):
    if canonical:
        frame=pd.read_csv(canonical)
        if not split_file:raise ValueError('canonical import requires an explicit patient split CSV')
        split=pd.read_csv(split_file);static={};births={}
        sources=[{'path':str(canonical),'sha256':hashlib.sha256(Path(canonical).read_bytes()).hexdigest()}]
    else:
        frame,split,static,births,sources=raw_to_frame(source)
    if split.subject_id.duplicated().any():raise ValueError('duplicate patient split')
    assignments=dict(zip(split.subject_id,split.split.replace({'val':'validation'})))
    if set(assignments.values())-{'train','validation','test'}:raise ValueError('unknown split names')
    categories={f['name']:f['classes'] for f in SCHEMA if f['kind']=='categorical'}
    sequences=[];excluded=Counter()
    for pid,records in frame.groupby('subject_id',sort=True):
        if pid not in assignments:continue
        quarter=defaultdict(lambda:defaultdict(list))
        for row in records.to_dict('records'):
            d=parse_date(row.get('date'))
            if d is None:continue
            q=(d.year,(d.month-1)//3)
            for name in [f['name'] for f in SCHEMA]+['hippocampal_volume_raw']:
                v=num(row.get(name))
                if v is None:continue
                if v<0 and name not in ['hippocampal_volume_change_percent','cortical_volume_change_percent']:continue
                if name in categories and (v!=int(v) or not 0<=v<categories[name]):continue
                quarter[q][name].append(v)
        values=[];ages=[];dates=[];first_volume=None
        for (year,q),fields in sorted(quarter.items()):
            d=date(year+1,1,1) if q==3 else date(year,3*q+4,1)
            ordinal=d.toordinal()-1
            vals={k:(Counter(v).most_common(1)[0][0] if k in categories else float(np.mean(v))) for k,v in fields.items()}
            vals.update(static.get(pid,{}))
            if births.get(pid):vals['age']=(ordinal-births[pid].toordinal())/365.25
            age=vals.get('age')
            if age is None or not 35<=age<=120:continue
            volume=vals.pop('hippocampal_volume_raw',None)
            if volume is not None:
                if first_volume is None:first_volume=volume
                vals['hippocampal_volume_change_percent']=100*(volume/first_volume-1)
            values.append([vals.get(f['name'],np.nan) for f in SCHEMA]);ages.append(age);dates.append(date.fromordinal(ordinal).isoformat())
        if len(ages)<2 or not np.all(np.diff(ages)>0):
            excluded['fewer_than_two_dated_quarters_or_invalid_age']+=1;continue
        sequences.append({'subject_id':pid,'split':assignments[pid],'ages':np.array(ages,np.float32),'values':np.array(values,np.float32),'dates':dates})
    if not sequences:raise ValueError('no usable patient sequences')
    train=np.concatenate([s['values'] for s in sequences if s['split']=='train'])
    counts=np.isfinite(train).sum(0)
    missing=[f['name'] for f,n in zip(SCHEMA,counts) if n==0]
    selected=[i for i,n in enumerate(counts) if n>0]
    schema=[SCHEMA[i] for i in selected]
    t=train[:,selected];minimum=np.nanmin(t,0);maximum=np.nanmax(t,0)
    if normalization=='train_zscore':offset=np.nanmean(t,0);scale=np.nanstd(t,0)
    elif normalization=='train_minmax':offset=minimum;scale=maximum-minimum
    else:raise ValueError('normalization must be an explicit train-only transform')
    scale=np.maximum(scale,1e-6)
    for j,f in enumerate(schema):
        if f['kind']=='categorical':offset[j]=0;scale[j]=1
    for s in sequences:
        raw=s['values'][:,selected];s['mask']=np.isfinite(raw).astype(np.float32)
        s['values']=np.where(s['mask'],(raw-offset)/scale,0).astype(np.float32)
    audit={'scope':scope,'features_in_paper_contract':46,'features_with_training_observations':len(schema),
           'missing_training_features':missing,'patient_counts':dict(Counter(s['split'] for s in sequences)),
           'excluded':dict(excluded),'normalization':normalization,'offset':offset.tolist(),'scale':scale.tolist(),
           'source_files':sources,'training_feature_counts':{f['name']:int(counts[i]) for i,f in enumerate(SCHEMA)},
           'quarter_definition':'calendar quarter, timestamp at right edge',
           'visit_merge':'PHASE + VISCODE2 (fallback VISCODE); earliest date from EXAMDATE, then VISDATE, then SCANDATE',
           'exact_paper_results_reproduced':False,
           'implementation_assumptions':['train-only minmax/zscore is explicit; original normalization unspecified',
             'category coding and static earliest demographic selection reconstructed',
             'quarter timestamp right edge; birth year-only uses July 1',
             'unphased source visits keep their dates to avoid cross-phase VISCODE collisions',
             'Roche CSF assay substitution for unavailable UPENN MASTER sources',
             'cortical PET SUVR reconstructed as volume-weighted cortical regions; full visual QC only']}
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    (cache/f'{scope}_audit.json').write_text(json.dumps(audit,indent=2))
    if scope=='paper46' and missing:
        raise ValueError('paper46 preparation stopped; missing training features: '+', '.join(missing))
    payload={'schema':schema,'sequences':sequences,'metadata':audit}
    with (cache/f'{scope}.pkl').open('wb') as f:pickle.dump(payload,f)
    print(json.dumps(audit,indent=2),flush=True)
    return payload


class Prefixes(Dataset):
    def __init__(self,sequences):
        self.sequences=sequences
        self.indices=[(i,c) for i,s in enumerate(sequences) for c in range(1,len(s['ages'])+1)]
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i,c=self.indices[index]
        return self.sequences[i],c


def collate(items,dropout=0.):
    b=len(items);n=max(len(s['ages']) for s,_ in items);f=items[0][0]['values'].shape[-1]
    values=np.zeros((b,n,f),np.float32);mask=np.zeros_like(values);ages=np.zeros((b,n),np.float32)
    valid=np.zeros((b,n),bool);context=np.zeros_like(values);cuts=[];ids=[]
    for j,(s,c) in enumerate(items):
        t=len(s['ages']);values[j,:t]=s['values'];mask[j,:t]=s['mask'];ages[j,:t]=s['ages'];ages[j,t:]=s['ages'][-1]
        valid[j,:t]=True;context[j,:c]=s['mask'][:c];cuts.append(c);ids.append(s['subject_id'])
    if dropout:context*=np.random.random(context.shape)>=dropout
    return {'values':torch.from_numpy(values),'mask':torch.from_numpy(mask),'ages':torch.from_numpy(ages),
            'context_mask':torch.from_numpy(context),'valid':torch.from_numpy(valid),'cutoff':torch.tensor(cuts),
            'max_context':max(cuts),'subject_ids':ids}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=DEFAULT_SOURCE)
    p.add_argument('--scope',choices=['paper46','local_available'],default='local_available')
    p.add_argument('--canonical');p.add_argument('--split-file')
    p.add_argument('--normalization',choices=['train_minmax','train_zscore'],default='train_minmax')
    a=p.parse_args();prepare(a.source,a.scope,a.canonical,a.split_file,a.normalization)
