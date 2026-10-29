"""Read-only treatment-table audit; saves aggregates, never patient records."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import re

import pandas as pd


def precision(value):
    text=str(value).strip()
    if text.lower() in ['nan','','none','-4','-4.0']:return 'missing_or_redacted'
    for fmt in ['%Y-%m-%d','%m/%d/%Y']:
        try:datetime.strptime(text,fmt);return 'day'
        except ValueError:pass
    if re.fullmatch(r'\d{4}-\d{2}-(?:xx|--)',text,re.I) or re.fullmatch(r'\d{1,2}/--/\d{4}',text):return 'month'
    if re.fullmatch(r'\d{4}-(?:xx|--)-(?:xx|--)',text,re.I) or re.fullmatch(r'--/--/\d{4}',text):return 'year'
    return 'other'


def day(value):
    if precision(value)!='day':return None
    for fmt in ['%Y-%m-%d','%m/%d/%Y']:
        try:return datetime.strptime(str(value),fmt)
        except ValueError:pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/data/SAMSUNG_backup/ADNI_data')
    parser.add_argument('--output',default='results/adni_treatment_feasibility/audit.json')
    args=parser.parse_args();root=Path(args.root);tables={};report={}
    payload=pickle.loads(Path('paper_odeae/cache/local_available.pkl').read_bytes())
    cohort={s['subject_id'] for s in payload['sequences']}
    for path in sorted((root/'interventions').glob('*.csv')):
        name=path.name.split('_18Oct')[0];df=pd.read_csv(path,low_memory=False);tables[name]=df
        report[name]={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows':len(df),'patients':int(df.RID.nunique()),'columns':list(df.columns)}
        if 'PTID' in df:report[name]['patients_in_current_2900_cohort']=int(df.loc[df.PTID.isin(cohort),'RID'].nunique())
    meds=tables['RECCMEDS'];name=meds.CMMED.fillna('').astype(str).str.strip()
    known=~name.str.lower().isin(['','-4','-4.0','nan'])
    report['RECCMEDS']['start_precision']=dict(Counter(meds.CMBGN.map(precision)))
    report['RECCMEDS']['end_precision']=dict(Counter(meds.CMEND.map(precision)))
    report['RECCMEDS']['usable_medication_names']=int(known.sum())
    report['RECCMEDS']['by_phase']={}
    for phase,g in meds.groupby('PHASE'):
        report['RECCMEDS']['by_phase'][str(phase)]={'rows':len(g),'patients':int(g.RID.nunique()),
            'start_precision':dict(Counter(g.CMBGN.map(precision))),
            'usable_names':int(known.loc[g.index].sum())}
    patterns={'cholinesterase_keyword':r'donepezil|aricept|rivastigmine|exelon|galantamine|razadyne|reminyl|tacrine|cognex|namzaric',
              'memantine_keyword':r'memantine|namenda|namzaric'}
    report['RECCMEDS']['exploratory_keywords']={}
    for group,pattern in patterns.items():
        chosen=meds.loc[name.str.contains(pattern,case=False,regex=True)&known]
        report['RECCMEDS']['exploratory_keywords'][group]={'rows':len(chosen),'patients':int(chosen.RID.nunique()),
            'current_cohort_patients':int(chosen.loc[chosen.PTID.isin(cohort),'RID'].nunique()),
            'start_precision':dict(Counter(chosen.CMBGN.map(precision))),
            'note':'keyword screen, not validated drug-class or incident-user labels; groups overlap'}
    anti=tables['ANTIAMYTX'];positive=anti[anti.TXREC==1]
    report['ANTIAMYTX'].update(txrec_distribution={str(k):int(v) for k,v in anti.TXREC.value_counts().items()},
        txrec1_rows=len(positive),txrec1_patients=int(positive.RID.nunique()),
        txrec1_current_cohort_patients=int(positive.loc[positive.PTID.isin(cohort),'RID'].nunique()),
        txname_codes={str(k):int(v) for k,v in positive.TXNAME.value_counts().items()},
        original_start_precision=dict(Counter(positive.STARTDATE.map(precision))),
        derived_start_precision=dict(Counter(positive.STARTDATE_DRVD.map(precision))))
    precise=positive.assign(parsed_start=positive.STARTDATE.map(day)).dropna(subset=['parsed_start'])
    starts=precise.groupby('RID').parsed_start.min().to_dict()
    report['ANTIAMYTX']['patients_with_some_exact_original_start']=len(starts)
    followup={}
    for table,field,maximum in [('MMSE','MMSCORE',30),('CDR','CDRSB',18)]:
        data=pd.read_csv(sorted((root/'Neuropsychological').glob(table+'_*.csv'))[0],low_memory=False)
        date_field='EXAMDATE' if 'EXAMDATE' in data else 'VISDATE'
        dates=pd.to_datetime(data[date_field],errors='coerce');v=pd.to_numeric(data[field],errors='coerce')
        count=Counter()
        for rid,start in starts.items():
            delta=(dates[(data.RID==rid)&v.between(0,maximum)]-start).dt.days.dropna()
            before=bool(delta.between(-365,-1).any())
            if before:count['with_prior_365d']+=1
            if before and delta.between(90,270).any():count['prior_and_6m_window']+=1
            if before and delta.between(270,450).any():count['prior_and_12m_window']+=1
        followup[table]={'date_field':date_field,**dict(count)}
    report['ANTIAMYTX']['rough_followup_screen']=followup
    report['ANTIAMYTX']['followup_note']='Earliest exact original start among available records; not confirmed first initiation. Counts use raw clinical EXAMDATE or VISDATE as recorded, not quarterly interpolation. Not an eligible causal cohort.'
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'dataset_root':str(root),
        'current_model_cohort_patients':len(cohort),'tables':report,
        'limits':['CSV rows may be repeated medication reviews, not unique prescriptions.',
                  'Date precision classification is separate from verifying event validity.',
                  'Drug code meanings must be checked against the phase-specific dictionary.',
                  'Absence of records is not untreated status; no causal cohort has been built.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
