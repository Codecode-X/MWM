"""Summarize the prepared cohort using CPU data statistics; never loads a model."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .prepare import DEFAULT,atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default=str(DEFAULT))
    parser.add_argument('--output',default='mimic_experiment/results/dataset_profile')
    args=parser.parse_args();root=Path(args.data_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=pickle.loads((root/'dataset.pkl').read_bytes());meta=data['metadata'];cohort=pd.read_csv(root/'cohort.csv')
    hours=np.array([len(s['ages']) for s in data['sequences']]);counts=pd.Series(hours).value_counts().sort_index()
    counts.rename_axis('observed_hours').rename('patients').to_csv(out/'observation_counts.csv')
    record={'dataset_revision':meta['dataset_revision'],'dataset_sha256':hashlib.sha256((root/'dataset.pkl').read_bytes()).hexdigest(),
            'patients':len(cohort),'patient_counts':meta['patient_counts'],'observed_hour_rows':int(hours.sum()),
            'observed_hours_per_patient':{'min':int(hours.min()),'median':float(np.median(hours)),'max':int(hours.max())},
            'age_proxy':cohort.admission_age.describe().to_dict(),'gender':cohort.gender.value_counts().to_dict(),
            'hospital_expire_flag_metadata_only':{str(k):int(v) for k,v in cohort.hospital_expire_flag.value_counts().items()},
            'care_unit':cohort.first_careunit.value_counts().to_dict()}
    atomic(out/'statistics.json',record)
    pd.DataFrame(meta['feature_profile']).to_csv(out/'feature_profile.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    axes[0].hist(cohort.admission_age,bins=np.arange(15,106,5),color='#4477AA',edgecolor='white')
    axes[0].set(xlabel='Anchor-derived age proxy (years)',ylabel='Patients',title='Age at ICU admission')
    gender=cohort.gender.value_counts().reindex(['F','M']);axes[1].bar(gender.index,gender.values,color=['#CC6677','#4477AA'])
    for i,n in enumerate(gender):axes[1].text(i,n,f'{n:,}',ha='center',va='bottom')
    axes[1].set(xlabel='Recorded sex',ylabel='Patients',title='Sex distribution',ylim=(0,gender.max()*1.13))
    axes[2].bar(counts.index,counts.values,color='#228833');axes[2].set(xlabel='Nonempty hours in the first 48h',ylabel='Patients',title='Observation support',xlim=(11,49))
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'cohort_profile.{ext}',dpi=180)
    plt.close(fig)
    profile=pd.DataFrame(meta['feature_profile']);fig,ax=plt.subplots(figsize=(10,11));y=np.arange(len(profile))
    for offset,key,color in [(-.24,'train','#4477AA'),(0,'validation','#EEAA33'),(.24,'test','#228833')]:
        ax.barh(y+offset,100*profile[key+'_observed_fraction'],height=.23,label=key,color=color)
    ax.set_yticks(y,profile.name);ax.invert_yaxis();ax.set(xlabel='Observed windows / all patient × 48h windows (%)',title='Feature coverage by patient split')
    ax.legend();fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'feature_coverage.{ext}',dpi=180)
    plt.close(fig)
    lines=['# MIMIC-IV 整理后的人群与观测覆盖','',
           '每位患者只取首次ICU住院的前48小时。完整筛选规则见[实验说明](../../README.zh-CN.md)。','',
           '| 项目 | 数量 |','|---|---:|',f'| 患者 | {len(cohort):,} |']
    lines += [f'| {key} | {value:,} |' for key,value in meta['patient_counts'].items()]
    lines += [f'| 有观测的小时行 | {int(hours.sum()):,} |',f'| 每人小时行数：最少 / 中位数 / 最多 | {hours.min()} / {np.median(hours):.0f} / {hours.max()} |',
              f'| 年龄估计值：均值 / 标准差 | {cohort.admission_age.mean():.2f} / {cohort.admission_age.std():.2f} 年 |',
              f"| 女性 / 男性 | {int(gender['F']):,} / {int(gender['M']):,} |",'',
              '![年龄、性别与观测窗口数量](cohort_profile.png)','',
              '下图的分母是每组患者数乘以48小时。一个字段在某小时至少有一条合格记录，就计为该小时可观测。性别只在第一条观测提供，因此占比固定为1/48。','',
              '![各字段观测覆盖](feature_coverage.png)','',
              '生命体征通常每小时有记录，化验项目和体重更稀疏。训练保留这种缺失差异，通过掩码决定哪些观测参与损失。','',
              '[逐观测次数统计](observation_counts.csv) · [字段覆盖与训练集分布](feature_profile.csv) · [汇总数据](statistics.json)','']
    (out/'summary.zh-CN.md').write_text('\n'.join(lines))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
