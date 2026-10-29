"""Population statistics from source tables; no patient identifiers exported."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

font_path=Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams['axes.unicode_minus']=False
ZH={'Male':'男','Female':'女','Unknown':'未知','CN':'认知正常','MCI':'轻度认知障碍',
    'Dementia':'痴呆','Conflicting labels':'同日诊断冲突','Other code 10 (unverified)':'其他编码10（待确认）',
    'Cholinesterase inhibitors':'胆碱酯酶抑制剂','Memantine':'美金刚',
    'Antidepressants (screen)':'抗抑郁药（初筛）','Lipid lowering (screen)':'降脂药（初筛）',
    'Antihypertensives (screen)':'降压相关药物（初筛）','Glucose lowering (screen)':'降糖药（初筛）',
    'Anti-amyloid (TXREC=1)':'抗淀粉样治疗记录（TXREC=1）'}


DRUGS={
 'Cholinesterase inhibitors':r'donepezil|aricept|rivastigmine|exelon|galantamine|razadyne|reminyl|tacrine|cognex|namzaric',
 'Memantine':r'memantine|namenda|namzaric',
 'Antidepressants (screen)':r'sertraline|zoloft|citalopram|celexa|escitalopram|lexapro|fluoxetine|prozac|paroxetine|paxil|bupropion|wellbutrin|venlafaxine|effexor|duloxetine|cymbalta|mirtazapine|remeron|trazodone',
 'Lipid lowering (screen)':r'atorvastatin|lipitor|simvastatin|zocor|rosuvastatin|crestor|pravastatin|pravachol|lovastatin|ezetimibe|zetia',
 'Antihypertensives (screen)':r'lisinopril|losartan|amlodipine|norvasc|valsartan|irbesartan|enalapril|ramipril|atenolol|metoprolol|carvedilol|hydrochlorothiazide',
 'Glucose lowering (screen)':r'metformin|glucophage|insulin|glipizide|glyburide|glimepiride|sitagliptin|januvia|empagliflozin|jardiance|semaglutide|ozempic',
}


def read(root,pattern):
    return pd.read_csv(sorted(root.glob(pattern))[-1],low_memory=False)


def birth(row):
    value=str(row.get('PTDOB',''))
    for fmt in ['%m/%Y','%Y-%m-%d','%m-%d-%Y']:
        try:return pd.Timestamp(datetime.strptime(value,fmt))
        except ValueError:pass
    year=re.match(r'^(\d{4})',str(row.get('PTDOBYY','')))
    if year and 1850<int(year[1])<2020:return pd.Timestamp(int(year[1]),7,1)
    return pd.NaT


def categorical(series):
    return {str(k):int(v) for k,v in series.value_counts(dropna=False).items()}


def table(counts,path,total):
    pd.DataFrame([{'category':k,'patients':v,'percent':round(100*v/total,4)} for k,v in counts.items()]).to_csv(path,index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/data/SAMSUNG_backup/ADNI_data')
    parser.add_argument('--output',default='results/adni_population');args=parser.parse_args()
    root,out=Path(args.root),Path(args.output);out.mkdir(parents=True,exist_ok=True)
    indexed=pd.read_csv(root/'ADNI_multimodal_timeline/indexes/patients.csv')
    times=pd.read_csv(root/'ADNI_multimodal_timeline/indexes/timepoints.csv')
    roster=read(root,'Enrollment/ROSTER_*.csv')
    population=sorted(set(roster.RID.dropna().astype(int)))
    rid_by_ptid=dict(zip(indexed.subject_id,indexed.rid.astype(int)))
    indexed_rids=sorted(indexed.rid.astype(int).unique())
    frame=pd.DataFrame(index=pd.Index(sorted(set(population)|set(indexed_rids)),name='RID'))
    demo=read(root,'Enrollment/PTDEMOG_*.csv');demo['date']=pd.to_datetime(demo.VISDATE,errors='coerce')
    sex={};dob={}
    for rid,g in demo.sort_values('date').groupby('RID'):
        g=g[g.PTGENDER.isin([1,2])]
        if len(g):sex[int(rid)]={1:'Male',2:'Female'}[int(g.iloc[0].PTGENDER)]
    for rid,g in demo.sort_values('date').groupby('RID'):
        dates=[birth(row) for _,row in g.iterrows()];dates=[x for x in dates if pd.notna(x)]
        if dates:dob[int(rid)]=dates[0]
    frame['sex']=pd.Series(sex).reindex(frame.index).fillna('Unknown')
    dx=read(root,'outcome/DXSUM_*.csv');dx['date']=pd.to_datetime(dx.EXAMDATE,errors='coerce')
    def diagnosis(row):
        code=row['DIAGNOSIS']
        if pd.notna(code) and code in [1,2,3]:return {1:'CN',2:'MCI',3:'Dementia'}[int(code)]
        if pd.notna(code) and code>=0:return f'Other code {int(code)} (unverified)'
        flags=[name for col,name in [('DXNORM','CN'),('DXMCI','MCI'),('DXAD','Dementia')] if row[col]==1]
        return flags[0] if len(flags)==1 else None
    dx['label']=dx.apply(diagnosis,axis=1)
    valid=dx.dropna(subset=['date','label']).copy()
    # Treat conflicting labels on the same day explicitly, rather than arbitrarily choosing one row.
    daily=valid.groupby(['RID','date']).label.agg(lambda x:next(iter(set(x))) if len(set(x))==1 else 'Conflicting labels').reset_index()
    first=daily.sort_values('date').groupby('RID').first();last=daily.sort_values('date').groupby('RID').last()
    frame['diagnosis_first']=first.label.reindex(frame.index).fillna('Unknown')
    frame['diagnosis_latest']=last.label.reindex(frame.index).fillna('Unknown')
    # Age refers to earliest available dated cognitive/diagnostic assessment, not medication-onset dates.
    clinical=[dx[['RID','date']]]
    for pattern in ['Neuropsychological/MMSE_*.csv','Neuropsychological/CDR_*.csv']:
        data=read(root,pattern);clinical.append(pd.DataFrame({'RID':data.RID,'date':pd.to_datetime(data.VISDATE,errors='coerce')}))
    first_clinical=pd.concat(clinical).dropna().groupby('RID').date.min()
    age={}
    for rid,date in first_clinical.items():
        if int(rid) in dob:
            number=(date-dob[int(rid)]).days/365.25
            if 18<=number<=120:age[int(rid)]=number
    frame['age']=pd.Series(age).reindex(frame.index)
    counts_all=times.groupby('subject_id').date.nunique().rename(index=rid_by_ptid)
    exams=times[times.examinations_status=='present']
    counts_exams=exams.groupby('subject_id').date.nunique().rename(index=rid_by_ptid)
    quarters=pd.to_datetime(exams.date).dt.to_period('Q')
    quarter_counts=exams.assign(quarter=quarters).groupby('subject_id').quarter.nunique().rename(index=rid_by_ptid)
    for key,counts in [('all_dates',counts_all),('exam_dates',counts_exams),('exam_quarters',quarter_counts)]:
        frame[key]=counts.reindex(frame.index).fillna(0).astype(int)
    observed=indexed.set_index('rid').timepoint_count.sort_index()
    assert (counts_all.reindex(observed.index).fillna(0).astype(int)==observed).all()
    reg=read(root,'Enrollment/REGISTRY_*.csv');reg['date']=pd.to_datetime(reg.EXAMDATE,errors='coerce')
    frame['registry_dates']=reg.dropna(subset=['date']).groupby('RID').date.nunique().reindex(frame.index).fillna(0).astype(int)
    meds=read(root,'interventions/RECCMEDS_*.csv')
    name=meds.CMMED.fillna('').astype(str).str.strip()
    fallback=meds.CMMEDO.fillna('').astype(str).str.strip()
    invalid=name.str.lower().isin(['','nan','-4','-4.0'])
    name=name.where(~invalid,fallback)
    known=~name.str.lower().isin(['','nan','-4','-4.0'])
    has_record=set(meds.RID.dropna().astype(int));has_name=set(meds.loc[known,'RID'].astype(int))
    for label,pattern in DRUGS.items():
        ids=set(meds.loc[known&name.str.contains(pattern,case=False,regex=True),'RID'].astype(int))
        frame[label]=frame.index.isin(ids)
    anti=read(root,'interventions/ANTIAMYTX_*.csv')
    positive=set(anti.loc[anti.TXREC==1,'RID'].astype(int))
    frame['Anti-amyloid (TXREC=1)']=frame.index.isin(positive)
    exclusive=[]
    for rid,row in frame.iterrows():
        if row['Cholinesterase inhibitors'] and row['Memantine']:label='Both classes ever recorded (not necessarily concurrent)'
        elif row['Cholinesterase inhibitors']:label='Cholinesterase class only ever recorded'
        elif row['Memantine']:label='Memantine class only ever recorded'
        elif rid in has_name:label='Other readable medication, no target keyword'
        elif rid in has_record:label='Medication rows, names unknown'
        else:label='No RECCMEDS record (unknown exposure)'
        exclusive.append(label)
    frame['medication_history_category']=exclusive
    results={}
    for cohort,ids in [('longitudinal',indexed_rids),('raw_roster',population)]:
        target=out/cohort;target.mkdir(exist_ok=True);f=frame.loc[ids];n=len(f)
        modes=['exam_dates','all_dates','exam_quarters'] if cohort=='longitudinal' else ['registry_dates']
        for mode in modes:
            dist=f[mode].value_counts().reindex(range(int(f[mode].max())+1),fill_value=0).sort_index();dist.rename_axis('observations').reset_index(name='patients').to_csv(target/f'{mode}_counts.csv',index=False)
        pd.crosstab(f.diagnosis_first,f.diagnosis_latest).to_csv(target/'diagnosis_transition_counts.csv')
        for col in ['sex','diagnosis_first','diagnosis_latest','medication_history_category']:
            table(categorical(f[col]),target/f'{col}.csv',n)
        bins=list(range(15,126,5));age_counts=pd.cut(f.age,bins=bins,right=False).value_counts().sort_index()
        table({str(k):int(v) for k,v in age_counts.items()}|{'Unknown':int(f.age.isna().sum())},target/'age_distribution.csv',n)
        drugs={k:int(f[k].sum()) for k in list(DRUGS)+['Anti-amyloid (TXREC=1)']}
        table(drugs,target/'drug_keyword_counts.csv',n)
        ages=f.age.dropna();summary={'patients':n,'age_available':len(ages),'age_unknown':int(f.age.isna().sum()),
          'age_mean':float(ages.mean()),'age_sd':float(ages.std()),'age_median':float(ages.median()),'age_min':float(ages.min()),'age_max':float(ages.max()),
          'sex':categorical(f.sex),'diagnosis_first':categorical(f.diagnosis_first),'diagnosis_latest':categorical(f.diagnosis_latest),
          'medication_history_category':categorical(f.medication_history_category),'drug_keywords':drugs,
          'observations':{mode:{'sum':int(f[mode].sum()),'mean':float(f[mode].mean()),'median':float(f[mode].median()),'min':int(f[mode].min()),'max':int(f[mode].max())} for mode in modes}}
        results[cohort]=summary
        fig,axes=plt.subplots(2,3,figsize=(18,10))
        mode=modes[0];counts=f[mode].value_counts().sort_index()
        axes[0,0].bar(counts.index,counts.values,width=.9);axes[0,0].set(xlabel='有检查记录的日期数' if cohort=='longitudinal' else '登记访视日期数',ylabel='患者人数',title='观测次数分布（每个整数单独计数）')
        axes[0,1].hist(ages,bins=bins,edgecolor='white');axes[0,1].set(xlabel='首次临床评估时年龄（岁）',ylabel='患者人数',title=f'年龄分布（未知 {n-len(ages)} 人）',xlim=(45,100))
        for ax,col,title in [(axes[0,2],'sex','性别分布'),(axes[1,0],'diagnosis_first','首次有效诊断'),(axes[1,1],'diagnosis_latest','最近一次有效诊断')]:
            values=f[col].value_counts();bars=ax.bar(range(len(values)),values.values);ax.set_xticks(range(len(values)),[ZH.get(x,x) for x in values.index],rotation=20,ha='right');ax.set(title=title,ylabel='患者人数');ax.bar_label(bars,padding=3)
        axes[1,2].barh([ZH.get(x,x) for x in drugs],list(drugs.values()));axes[1,2].set(title='随访期间曾记录的药物类别（允许重叠）',xlabel='患者人数')
        fig.suptitle(f'ADNI {"正式纵向队列" if cohort=="longitudinal" else "原始总名册"}：{n} 位患者',fontsize=17);fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(target/f'population_dashboard.{ext}',dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,5));ax.bar(counts.index,counts.values,width=.9);ax.set(xlabel='有检查记录的不同日期数' if cohort=='longitudinal' else '登记访视日期数',ylabel='患者人数',title=f'ADNI 观测次数直方图（N={n}，每个整数独立计数）');fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(target/f'observation_histogram.{ext}',dpi=180)
        plt.close(fig)
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'source_root':str(root),'cohorts':results,
     'drug_keyword_patterns':DRUGS,'diagnosis_codes':{'1':'CN','2':'MCI','3':'Dementia','10':'Other unverified code'},
     'definitions':{'observations':'Main: distinct dates with examinations_status=present, not visit episodes or modality counts.',
      'age':'Earliest dated MMSE/CDR/diagnostic assessment; month birth date uses first day, year only uses July 1; unavailable or outside 18..120 is unknown.',
      'drugs':'Ever matched text in CMMED, CMMEDO fallback; not validated prescription episodes, not simultaneous regimens; absent records are unknown.'}}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
