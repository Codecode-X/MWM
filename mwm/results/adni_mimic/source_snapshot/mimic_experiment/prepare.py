"""Stream MIMIC-IV 3.1 into masked first-48-hour ICU trajectories (CPU I/O)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd

DEFAULT=Path('/data/MIMIC/derived/latentode_iv31')
ARCHIVE=Path('/data/MIMIC/mimic-iv-3.1.zip')
# Broad QC bounds; vital limits follow MIT-LCP mimic-code vitalsign.sql.
# Lab limits are explicit engineering QC bounds, not clinical reference intervals.
NUMERIC=[
 ('heart_rate','bpm',0,300,[220045],[]),
 ('sbp','mmHg',0,400,[220050,220179,225309],[]),
 ('dbp','mmHg',0,300,[220051,220180,225310],[]),
 ('map','mmHg',0,300,[220052,220181,225312],[]),
 ('respiratory_rate','breaths/min',0,70,[220210,224690],[]),
 ('spo2','%',0,100,[220277],[]),
 ('temperature','degC',10,50,[223761,223762],[]),
 ('fio2','fraction',.20,1.,[223835],[]),
 ('weight','kg',1,500,[224639,226512],[]),
 ('glucose','mg/dL',0,2000,[],[50809,50931]),
 ('urea_nitrogen','mg/dL',0,300,[],[51006]),
 ('creatinine','mg/dL',0,40,[],[50912]),
 ('sodium','mEq/L',80,200,[],[50983]),
 ('potassium','mEq/L',.5,15,[],[50971]),
 ('chloride','mEq/L',40,200,[],[50902]),
 ('bicarbonate','mEq/L',0,80,[],[50882]),
 ('calcium','mg/dL',0,25,[],[50893]),
 ('magnesium','mg/dL',0,20,[],[50960]),
 ('phosphate','mg/dL',0,30,[],[50970]),
 ('wbc','K/uL',0,1000,[],[51300,51301]),
 ('hemoglobin','g/dL',0,30,[],[51222]),
 ('hematocrit','%',0,80,[],[51221]),
 ('platelets','K/uL',0,3000,[],[51265]),
 ('lactate','mmol/L',0,50,[],[50813]),
 ('blood_gas_ph','pH',6.3,8.1,[],[50820]),
 ('blood_gas_pco2','mmHg',0,200,[],[50818]),
 ('blood_gas_po2','mmHg',0,1000,[],[50821]),
]
CATEGORICAL=[('gcs_eye',4,[220739]),('gcs_motor',6,[223901]),('gcs_verbal',5,[223900])]


def atomic(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def schema():
    return ([{'name':n,'kind':'numerical','unit':unit,'qc_lower_exclusive':lo,'qc_upper_inclusive':hi,
              'chart_itemids':chart,'lab_itemids':lab} for n,unit,lo,hi,chart,lab in NUMERIC]+
            [{'name':n,'kind':'categorical','classes':k,'codes':list(range(1,k+1)),'chart_itemids':ids,'lab_itemids':[]} for n,k,ids in CATEGORICAL]+
            [{'name':'sex','kind':'categorical','classes':2,'codes':['F','M'],'availability':'first observed hour only','chart_itemids':[],'lab_itemids':[]}])


def split(subject):
    value=int(hashlib.sha256(f'mimiciv31/seed0/{int(subject)}'.encode()).hexdigest()[:16],16)/2**64
    return 'train' if value<.7 else 'validation' if value<.85 else 'test'


def chronology_support(frame):
    """Require a complete living, in-hospital first-48-hour observation window."""
    start=pd.to_datetime(frame.intime);end=start+pd.Timedelta(hours=48)
    admitted=pd.to_datetime(frame.admittime);discharged=pd.to_datetime(frame.dischtime)
    death=pd.to_datetime(frame.deathtime)
    return (start>=admitted)&(end<=discharged)&(death.isna()|(death>=end))


def sources(root,archive):
    dest=root/'source';dest.mkdir(exist_ok=True);previous=json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else {}
    record={}
    with ZipFile(archive) as z:
        for table in ['patients','admissions','icustays','d_items','d_labitems','chartevents','labevents']:
            name=next(n for n in z.namelist() if n.endswith('/'+table+'.csv.gz'));info=z.getinfo(name);target=dest/(table+'.csv.gz')
            if table in previous and target.exists() and target.stat().st_size==info.file_size and previous[table]['crc32']==info.CRC:
                record[table]=previous[table];continue
            print(f'Extract selected gzip member: {table}, {info.file_size/2**30:.2f} GiB',flush=True)
            digest=hashlib.sha256();tmp=target.with_suffix('.gz.partial')
            with z.open(name) as stream,tmp.open('wb') as output:
                while block:=stream.read(8*1024*1024):digest.update(block);output.write(block)
            tmp.replace(target)
            record[table]={'archive_member':name,'bytes':info.file_size,'crc32':info.CRC,'sha256':digest.hexdigest()}
            atomic(dest/'manifest.json',{**previous,**record})
    atomic(dest/'manifest.json',record)


def cohort(root):
    source=root/'source';stays=pd.read_csv(source/'icustays.csv.gz');patients=pd.read_csv(source/'patients.csv.gz');admissions=pd.read_csv(source/'admissions.csv.gz',usecols=['subject_id','hadm_id','hospital_expire_flag'])
    stays['intime']=pd.to_datetime(stays.intime);stays['outtime']=pd.to_datetime(stays.outtime)
    first=stays.sort_values(['subject_id','intime','stay_id']).drop_duplicates('subject_id').merge(patients[['subject_id','gender','anchor_age','anchor_year']],on='subject_id',validate='one_to_one')
    first['admission_age']=first.anchor_age+first.intime.dt.year-first.anchor_year
    first['duration_hours']=(first.outtime-first.intime).dt.total_seconds()/3600
    good=first[(first.admission_age>=18)&(first.duration_hours>=48)].copy().sort_values('subject_id').reset_index(drop=True)
    good=good.merge(admissions,on=['subject_id','hadm_id'],how='left',validate='one_to_one');good['row_index']=np.arange(len(good));good['split']=good.subject_id.map(split)
    good.to_csv(root/'candidate_cohort.csv',index=False)
    audit={'all_icu_stays':len(stays),'unique_first_icu_patients':len(first),'adult_first_icu':int((first.admission_age>=18).sum()),
           'first_icu_at_least48h':len(good),'candidate_by_split':good.split.value_counts().to_dict(),
           'first_stay_rule':'earliest ICU stay per subject before LOS filtering; never select a later eligible stay'}
    atomic(root/'cohort_selection.json',audit);atomic(root/'schema.json',schema())
    items=pd.read_csv(source/'d_items.csv.gz');labs=pd.read_csv(source/'d_labitems.csv.gz');mapping=[]
    for feature in schema():
        for table,ids,dictionary in [('chartevents',feature['chart_itemids'],items),('labevents',feature['lab_itemids'],labs)]:
            for item in ids:
                match=dictionary[dictionary.itemid==item]
                if len(match)!=1:raise ValueError(f'Item dictionary mismatch: {item}')
                if table=='chartevents' and match.iloc[0].linksto!='chartevents':raise ValueError('wrong event table')
                mapping.append({'feature':feature['name'],'table':table,'itemid':item,'label':str(match.iloc[0]['label'])})
    pd.DataFrame(mapping).to_csv(root/'verified_item_mapping.csv',index=False)
    return good


def filter_table(root,table):
    frame=pd.read_csv(root/'candidate_cohort.csv');n=len(frame);fields=schema();f=len(fields)
    sums=np.zeros((n,48,f),np.float64);counts=np.zeros((n,48,f),np.uint32)
    categories={j:np.zeros((n,48,s['classes']),np.uint32) for j,s in enumerate(fields) if s['kind']=='categorical' and s['name']!='sex'}
    key='stay_id' if table=='chartevents' else 'subject_id'
    row_lookup=pd.Series(frame.row_index.to_numpy(),index=frame[key]);intimes=pd.to_datetime(frame.intime).astype('int64').to_numpy();hadms=frame.hadm_id.to_numpy()
    mapping={item:j for j,s in enumerate(fields) for item in s['chart_itemids' if table=='chartevents' else 'lab_itemids']}
    usecols=['subject_id','hadm_id','charttime','storetime','itemid','valuenum','valueuom']+(['stay_id'] if table=='chartevents' else [])
    audit=Counter();unit_census={s['name']:Counter() for s in fields};started=time.monotonic()
    reader=pd.read_csv(root/'source'/(table+'.csv.gz'),usecols=usecols,chunksize=1_000_000,
                       dtype={'subject_id':'int64','hadm_id':'float64','itemid':'int64','valuenum':'float64',**({'stay_id':'int64'} if table=='chartevents' else {})})
    for number,chunk in enumerate(reader,1):
        audit['source_rows']+=len(chunk)
        chunk=chunk[chunk.itemid.isin(mapping)&chunk[key].isin(row_lookup.index)&chunk.valuenum.notna()].copy()
        if not len(chunk):continue
        rows=chunk[key].map(row_lookup).to_numpy(dtype=np.int64);chunk['r']=rows
        stay_match=chunk.hadm_id.isna()|(chunk.hadm_id.to_numpy()==hadms[rows])
        audit['admission_mismatch']+=int((~stay_match).sum());chunk=chunk[stay_match].copy()
        if not len(chunk):continue
        rows=chunk.r.to_numpy(dtype=np.int64);measured=pd.to_datetime(chunk.charttime,errors='coerce');stored=pd.to_datetime(chunk.storetime,errors='coerce')
        available=measured.where(stored.isna()|(measured>=stored),stored)
        elapsed=(available.astype('int64').to_numpy()-intimes[rows])/3.6e12
        within=measured.notna().to_numpy()&np.isfinite(elapsed)&(elapsed>=0)&(elapsed<48)
        audit['outside_available_48h']+=int((~within).sum());chunk=chunk[within].copy();elapsed=elapsed[within]
        if not len(chunk):continue
        rows=chunk.r.to_numpy(dtype=np.int64);bins=np.floor(elapsed).astype(np.int64);feature=chunk.itemid.map(mapping).to_numpy(dtype=np.int64);values=chunk.valuenum.to_numpy()
        fahrenheit=chunk.itemid.to_numpy()==223761
        fahrenheit_valid=(~fahrenheit)|((values>70)&(values<120));values=np.where(fahrenheit,(values-32)/1.8,values)
        oxygen=feature==next(j for j,s in enumerate(fields) if s['name']=='fio2');values=np.where(oxygen&(values>1),values/100,values)
        valid=np.isfinite(values)&fahrenheit_valid
        for j,s in enumerate(fields):
            select=feature==j
            if not select.any():continue
            unit_census[s['name']].update(chunk.loc[select,'valueuom'].fillna('(missing)').astype(str).value_counts().to_dict())
            if s['kind']=='numerical':valid[select]&=(values[select]>s['qc_lower_exclusive'])&(values[select]<=s['qc_upper_inclusive'])
            else:valid[select]&=(values[select]>=1)&(values[select]<=s['classes'])&np.isclose(values[select],np.rint(values[select]))
        audit['invalid_value_or_unit_conversion']+=int((~valid).sum())
        rows=rows[valid];bins=bins[valid];feature=feature[valid];values=values[valid]
        for j,s in enumerate(fields):
            select=feature==j
            if not select.any():continue
            if s['kind']=='categorical':np.add.at(categories[j],(rows[select],bins[select],np.rint(values[select]).astype(int)-1),1)
            else:
                flat=(rows[select]*48+bins[select])*f+j
                np.add.at(sums.ravel(),flat,values[select]);np.add.at(counts.ravel(),flat,1)
        audit['retained_event_values']+=len(values)
        if number%5==0:
            progress={**dict(audit),'table':table,'chunks':number,'elapsed_seconds':time.monotonic()-started}
            atomic(root/(table+'_progress.json'),progress);print(json.dumps(progress),flush=True)
    target=root/(table+'_aggregates.npz');tmp=target.with_suffix('.partial.npz')
    np.savez_compressed(tmp,sums=sums,counts=counts,**{f'category_{j}':v for j,v in categories.items()});tmp.replace(target)
    atomic(root/(table+'_audit.json'),{**dict(audit),'elapsed_seconds':time.monotonic()-started,'unit_census':{k:dict(v) for k,v in unit_census.items() if v}})
    print(f'{table}: aggregation complete',flush=True)


def finalize(root):
    fields=schema();cohort_=pd.read_csv(root/'candidate_cohort.csv');tables=[np.load(root/(table+'_aggregates.npz')) for table in ['chartevents','labevents']]
    admissions=pd.read_csv(root/'source/admissions.csv.gz',usecols=['hadm_id','admittime','dischtime','deathtime'])
    cohort_=cohort_.merge(admissions,on='hadm_id',how='left',validate='one_to_one',sort=False)
    assert np.array_equal(cohort_.row_index.to_numpy(),np.arange(len(cohort_)))
    counts=tables[0]['counts']+tables[1]['counts'];sums=tables[0]['sums']+tables[1]['sums'];mask=counts>0
    values=np.divide(sums,counts,out=np.zeros_like(sums),where=mask).astype(np.float32)
    for j,s in enumerate(fields):
        if s['kind']=='categorical' and s['name']!='sex':
            cat=tables[0][f'category_{j}']+tables[1][f'category_{j}'];mask[:,:,j]=cat.sum(-1)>0;values[:,:,j]=cat.argmax(-1)
    dynamic=mask.any(-1);observed=(dynamic[:,:24].sum(1)>=6)&(dynamic[:,24:].sum(1)>=6)
    chronology=chronology_support(cohort_).to_numpy();eligible=observed&chronology
    excluded=int((~observed).sum());chronology_excluded=int((observed&~chronology).sum())
    cohort_=cohort_[eligible].copy();values=values[eligible];mask=mask[eligible];dynamic=dynamic[eligible]
    sex=next(j for j,s in enumerate(fields) if s['name']=='sex')
    for row,gender in enumerate(cohort_.gender):
        if gender in ['F','M']:
            first=np.flatnonzero(dynamic[row])[0];values[row,first,sex]=int(gender=='M');mask[row,first,sex]=True
    train=cohort_.split.to_numpy()=='train';offset=[];scale=[];std=[];profile=[]
    for j,s in enumerate(fields):
        observed=values[:,:,j][mask[:,:,j]&train[:,None]].astype(np.float64)
        if not len(observed):raise ValueError('Feature has no training data: '+s['name'])
        if s['kind']=='numerical':lo=float(observed.min());size=max(float(observed.max()-observed.min()),1e-6);sd=float(observed.std())
        else:lo=0.;size=1.;sd=float(observed.std())
        offset.append(lo);scale.append(size);std.append(sd)
        item={'name':s['name'],'kind':s['kind'],'unit':s.get('unit'),'train_observations':len(observed),'train_min':float(observed.min()),'train_max':float(observed.max()),'train_mean':float(observed.mean()),'train_std':sd}
        for split_ in ['train','validation','test']:
            selected=cohort_.split.to_numpy()==split_;item[split_+'_observed_fraction']=float(mask[selected,:,j].sum()/(selected.sum()*48))
        profile.append(item)
        if s['kind']=='numerical':values[:,:,j]=np.where(mask[:,:,j],(values[:,:,j]-lo)/size,0.)
    sequences=[]
    for row,patient in enumerate(cohort_.itertuples()):
        keep=np.flatnonzero(dynamic[row]);times=(keep+1).astype(np.float32)
        seq={'subject_id':str(patient.subject_id),'hadm_id':int(patient.hadm_id),'stay_id':int(patient.stay_id),
             'split':patient.split,'admission_age':float(patient.admission_age),'ages':times,
             'values':values[row,keep].astype(np.float32),'mask':mask[row,keep].astype(np.float32),
             'dates':[f'ICU+{int(t)}h' for t in times], 'hospital_expire_flag':int(patient.hospital_expire_flag)}
        assert len(seq['ages'])>=12 and np.diff(seq['ages']).min()>0
        assert np.searchsorted(times,24,side='right')>=6 and (times>24).sum()>=6
        sequences.append(seq)
    report={'dataset':'MIMIC-IV v3.1','scope':'mimic_iv31_first48h','normalization':'train_minmax','time_unit':'hour since ICU admission; hourly window endpoints',
            'age_initialization':'anchor_age + ICU admission year - anchor_year; supplied separately from ODE hours',
            'availability_time':'max(charttime,storetime), missing storetime falls back to charttime',
            'aggregation':'numeric mean per available-time hour; GCS category mode, ties smallest code; empty hours removed',
            'normalization_fit':'training patients only; no validation/test clipping','offset':offset,'scale':scale,'raw_train_std':std,
            'patient_counts':{s:sum(r['split']==s for r in sequences) for s in ['train','validation','test']},
            'patients':len(sequences),'features':len(fields),'expanded_features':sum(s.get('classes',1) for s in fields),
            'excluded_insufficient_observation_support':excluded,'excluded_invalid_48h_chronology_after_observation_filter':chronology_excluded,
            'cohort_chronology':'ICU start >= admission; ICU start + 48h <= discharge; recorded death absent or >= ICU start + 48h',
            'dataset_revision':'chronology_v2','primary_context_hours':24,'training_context_hours':[6,12,18,24],
            'forecast_hours':[6,12,24],'source_manifest':str(root/'source/manifest.json'),'source_version_separate':'MIMIC-III retained separately; not pooled',
            'feature_profile':profile,'preparation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for s in ['train','validation','test']:
        assert report['patient_counts'][s]>0
    tmp=root/'dataset.partial.pkl'
    with tmp.open('wb') as stream:pickle.dump({'schema':fields,'sequences':sequences,'metadata':report},stream,pickle.HIGHEST_PROTOCOL)
    tmp.replace(root/'dataset.pkl');atomic(root/'dataset_summary.json',report);pd.DataFrame(profile).to_csv(root/'feature_profile.csv',index=False)
    cohort_.to_csv(root/'cohort.csv',index=False)
    print(json.dumps({'prepared':str(root/'dataset.pkl'),'patients':len(sequences),'splits':report['patient_counts'],'fields':len(fields)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(DEFAULT));parser.add_argument('--archive',default=str(ARCHIVE));parser.add_argument('--table',choices=['chartevents','labevents'])
    args=parser.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    if args.table:filter_table(root,args.table);return
    if (root/'dataset.pkl').exists():print('Prepared dataset already exists; preserved',flush=True);return
    sources(root,Path(args.archive));cohort(root)
    def launch(table):
        if (root/(table+'_aggregates.npz')).exists():return
        with (root/(table+'_prepare.log')).open('w') as output:
            subprocess.run([sys.executable,'-u','-m','mimic_experiment.prepare','--root',str(root),'--table',table],stdout=output,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(launch,['chartevents','labevents']))
    finalize(root)


if __name__=='__main__':main()
