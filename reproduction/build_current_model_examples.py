"""Build method-document examples from data and saved CUDA predictions; CPU I/O only."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/assets/current_model'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sequence(payload,subject):
    sequence=next(s for s in payload['sequences'] if s['subject_id']==subject)
    raw=sequence['values']*np.asarray(payload['metadata']['scale'])+np.asarray(payload['metadata']['offset'])
    raw=np.where(sequence['mask'].astype(bool),raw,np.nan)
    return sequence,raw,{field['name']:j for j,field in enumerate(payload['schema'])}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    adni_data=ROOT/'paper_odeae/cache/local_available.pkl'
    cases_path=ROOT/'paper_odeae/results/state_space_model_comparison/latent2x_both/forecast_cases.json'
    statistics=json.loads(cases_path.with_name('statistics.json').read_text())
    assert sha(statistics['checkpoint'])==statistics['checkpoint_sha256']
    assert sha(adni_data)==statistics['dataset_sha256']
    case=next(c for c in json.loads(cases_path.read_text()) if c['subject_id']=='002_S_4219')
    payload=pickle.loads(adni_data.read_bytes());s,raw,index=raw_sequence(payload,case['subject_id'])
    assert s['split']=='test' and case['context_visits']==3 and len(s['ages'])==4
    adni_rows=[]
    for i in range(len(s['ages'])):
        values={field:float(raw[i,index[field]]) if np.isfinite(raw[i,index[field]]) else None
                for field in ['diagnosis','mmse','cdrsb','amyloid_cortical_suvr','hippocampal_volume_change_percent']}
        if values['mmse'] is None:assert case['actual_visit_mmse'][i] is None
        else:assert abs(values['mmse']-case['actual_visit_mmse'][i])<1e-4
        adni_rows.append({'date':s['dates'][i],'age':float(s['ages'][i]),'years_from_cutoff':case['visit_years'][i],
                          'role':'history' if i<3 else 'future_target','observations':values,
                          'generated_mmse':case['native_visit_mmse'][i]})
    del payload
    mimic_data=Path('/data/MIMIC/derived/latentode_iv31/dataset.pkl')
    payload=pickle.loads(mimic_data.read_bytes());m,mraw,mi=raw_sequence(payload,'10004235')
    assert m['split']=='test' and 24 in m['ages'] and 48 in m['ages']
    mimic_rows=[]
    for i,hour in enumerate(m['ages']):
        mimic_rows.append({'hour':int(hour),'role':'history' if hour<=24 else 'future_target',
                           'observations':{field:float(mraw[i,mi[field]]) if np.isfinite(mraw[i,mi[field]]) else None
                                           for field in ['heart_rate','sbp','respiratory_rate','spo2','glucose','creatinine']}})
    result={'purpose':'Real held-out patient timelines for a model-method explanation; cached ADNI CUDA predictions reused.',
            'model_execution_performed':False,
            'adni':{'subject_id':s['subject_id'],'split':s['split'],'context_visits':3,'source_data':str(adni_data),
                    'source_data_sha256':sha(adni_data),'prediction_source':str(cases_path),'prediction_source_sha256':sha(cases_path),
                    'checkpoint':statistics['checkpoint'],'checkpoint_sha256':statistics['checkpoint_sha256'],
                    'checkpoint_epoch':statistics['checkpoint_epoch'],'rows':adni_rows},
            'mimic':{'subject_id':m['subject_id'],'split':m['split'],'context_hours':24,'admission_age_proxy':m['admission_age'],
                     'source_data':str(mimic_data),'source_data_sha256':sha(mimic_data),
                     'historical_observed_hours':int((m['ages']<=24).sum()),'future_observed_hours':int((m['ages']>24).sum()),
                     'rows':mimic_rows}}
    (OUT/'case_data.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.7))
    x=np.asarray(case['visit_years'])*12;truth=np.asarray([np.nan if v is None else v for v in case['actual_visit_mmse']])
    pred=np.asarray(case['native_visit_mmse']);ax.axvspan(x[0]-.35,0,color='#4477AA',alpha=.07);ax.axvspan(0,x[-1]+.5,color='#EEAA33',alpha=.10)
    ax.axvline(0,color='gray',linestyle='--',linewidth=1)
    ax.plot(x[:3],pred[:3],color='#4477AA',marker='x',label='Reconstructed MMSE')
    curve_x=np.asarray(case['years'])*12;curve_y=np.asarray(case['mmse_curve']);keep=curve_x<=x[-1]+1e-4
    ax.plot(curve_x[keep],curve_y[keep],color='#CC6677',label='ODE forecast from cutoff')
    ax.scatter(x[:3],truth[:3],color='#228833',s=65,zorder=4,label='Observed historical MMSE')
    ax.scatter(x[3:],truth[3:],color='#EE7733',s=65,zorder=4,label='Held-out future MMSE')
    ax.annotate('PET/MRI record\nMMSE missing',xy=(x[1],pred[1]),xytext=(x[1]-.3,29.45),ha='center',fontsize=10,
                arrowprops={'arrowstyle':'->','color':'gray'})
    ax.set(xlabel='Months relative to the history cutoff',ylabel='MMSE (points)',title='ADNI: history assimilation, reconstruction and future forecast',
           xlim=(x[0]-.35,x[-1]+.5),ylim=(27.8,30.7));ax.legend(loc='upper right',fontsize=9)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'adni_case.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8.2),sharex=True)
    for ax,field,title,unit in zip(axes,['heart_rate','glucose','creatinine'],['Heart rate','Glucose','Creatinine'],['bpm','mg/dL','mg/dL']):
        values=mraw[:,mi[field]];past=m['ages']<=24;future=~past
        ax.axvspan(0,24,color='#4477AA',alpha=.07);ax.axvspan(24,48,color='#EEAA33',alpha=.10);ax.axvline(24,color='gray',linestyle='--',linewidth=1)
        ax.scatter(m['ages'][past],values[past],color='#4477AA',s=27,label='Available history')
        ax.scatter(m['ages'][future],values[future],color='#EE7733',s=27,label='Held-out future observations')
        ax.set(ylabel=f'{title}\n({unit})',xlim=(0,49));ax.grid(axis='y',alpha=.18)
    axes[0].set_title('MIMIC-IV: the same patient observed before and after a 24h cutoff')
    axes[0].legend(loc='upper right',fontsize=9);axes[-1].set_xlabel('Hours since ICU admission');fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'mimic_case.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({'adni_rows':len(adni_rows),'mimic_rows':len(mimic_rows),'output':str(OUT),'model_execution_performed':False}))


if __name__=='__main__':main()
