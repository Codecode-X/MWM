"""Statistical graphics from CUDA artifacts; no torch/model imports."""
import csv
import json
from pathlib import Path

import numpy as np


def graphics(out,records,measured,precision,patients):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr,rankdata
    out=Path(out);links=[];summaries={}
    if measured:
        byid={r['id']:r for r in records};rows=[r for r in measured if r['step']['id'] in byid]
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for name,title in [('A_reencoded','Re-encoded history'),('B_fixed_state','Fixed cutoff state')]:
            x=[byid[r['step']['id']]['relative_field_update'] for r in rows]
            y=[r['flags'][name]['after_before_ratio'] for r in rows]
            axes[0].scatter(x,y,s=10,alpha=.45,label=title)
            sx=[(r['step']['epoch']-1)*169+r['step']['batch'] for r in rows]
            axes[1].plot(sx,[r['before'][name]['past']['gain']['median'] for r in rows],ls='--',lw=.8,alpha=.7,label=title+' before')
            axes[1].plot(sx,[r['after'][name]['past']['gain']['median'] for r in rows],lw=.9,label=title+' after')
        axes[0].set(xscale='log',yscale='log',xlabel='Relative ODE parameter update norm',ylabel='After / before sensitivity',title='Update amplitude and sensitivity change')
        axes[1].set(yscale='log',xlabel='Optimizer step',ylabel='Median amplification',title='Before and after each update')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'update_effects.png',dpi=170);plt.close(fig);links.append(('更新幅度与敏感性变化','update_effects.png'))
        with (out/'update_sensitivity.csv').open('w',newline='') as f:
            fields=['epoch','batch','optimizer_step','history_gradient','future_gradient','cosine','relative_field_update','A_before','A_after','B_before','B_after']
            writer=csv.writer(f);writer.writerow(fields)
            for r in rows:
                s=r['step'];t=byid[s['id']]
                writer.writerow([s['epoch'],s['batch'],t['optimizer_step'],t['before']['history_gradient_norm'],t['before']['future_gradient_norm'],t['before']['history_future_cosine'],t['relative_field_update']]+[r[when][name]['past']['gain']['median'] for name in ['A_reencoded','B_fixed_state'] for when in ['before','after']])
    if precision:
        fig,axes=plt.subplots(1,3,figsize=(16,4.8));names=['train32','diagnostic32','tight32','tightest32','diagnostic64','tight64','tightest64']
        for item in precision:
            rows={r['setting']:r for r in item['settings'] if r['anchor']=='snapshot_reencoded_state' and r['status']=='complete'}
            for ax,key,title in zip(axes,['past_endpoint','past_singular_value','history_field_gradient'],['Past endpoint','Past largest singular value','History direct field gradient']):
                xs=[i for i,n in enumerate(names) if n in rows and 'relative_differences_to_reference' in rows[n]]
                ys=[max(rows[names[i]]['relative_differences_to_reference'][key],1e-16) for i in xs]
                if xs:ax.plot(xs,ys,'o-',ms=3,label=item['snapshot']['id'])
                ax.set(yscale='log',title=title,ylabel='Relative difference to refined RK4');ax.set_xticks(range(len(names)),names,rotation=60,ha='right');ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'solver_convergence.png',dpi=170);plt.close(fig);links.append(('求解器状态与梯度精度对照','solver_convergence.png'))
    for item in patients:
        case=item['case']['id'];rows=item['patients'];folder=out/'experiment3'/case
        t=np.array([r['history_span_hours'] for r in rows]);a=np.array([r['past_state_max_singular_value'] for r in rows]);g=np.array([r['history_field_gradient_norm'] for r in rows])
        actual=np.array([r['earliest_actual_direction_gain'] for r in rows]);bound=np.array([r['native_past_state_max_singular_value'] for r in rows]);share=np.array([r['projected_share_of_batch_history_gradient'] for r in rows])
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        axes[0,0].scatter(t,a,s=18,alpha=.65);axes[0,0].set(xlabel='History span (hours)',ylabel='Past amplification',yscale='log')
        axes[0,1].scatter(t,g,s=18,alpha=.65);axes[0,1].set(xlabel='History span (hours)',ylabel='Patient history field gradient',yscale='log')
        axes[1,0].scatter(a,g,s=18,alpha=.65);axes[1,0].set(xlabel='Past amplification',ylabel='Patient history field gradient',xscale='log',yscale='log')
        axes[1,1].scatter(bound,actual,s=18,alpha=.65);lo=max(min(bound.min(),actual.min()),1e-8);hi=max(bound.max(),actual.max());axes[1,1].plot([lo,hi],[lo,hi],ls='--',color='gray')
        axes[1,1].set(xlabel='Largest singular value: bound',ylabel='Actual earliest-loss gradient gain',xscale='log',yscale='log')
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle(case);fig.tight_layout();fig.savefig(folder/'patient_mechanisms.png',dpi=160);plt.close(fig)
        links.append((case+'：跨度、状态与实际梯度放大','experiment3/'+case+'/patient_mechanisms.png'))
        order=np.argsort(g)[::-1];top=order[:20]
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        axes[0].hist(np.log10(np.maximum(g,1e-20)),bins=20);axes[0].set(xlabel='log10 patient field gradient norm',ylabel='Patients',title='Patient contribution distribution')
        axes[1].bar(np.arange(len(top)),share[top]);axes[1].set(xlabel='Rank by gradient norm',ylabel='Projected share of batch history gradient',title='Top 20 gradient contributors')
        selected=np.argsort(a)[::-1][:8]
        for j,i in enumerate(selected):axes[2].scatter(rows[i]['history_times_hours'],np.full(len(rows[i]['history_times_hours']),j),s=8)
        axes[2].set(xlabel='Hours from ICU admission',ylabel='Patient index',title='Observation times: highest sensitivity');axes[2].set_yticks(range(len(selected)),[str(i) for i in selected])
        fig.tight_layout();fig.savefig(folder/'patient_contributions.png',dpi=160);plt.close(fig);links.append((case+'：患者贡献及高敏感性患者的观测时间','experiment3/'+case+'/patient_contributions.png'))
        covariates=['history_span_hours','history_observations','missing_fraction','cutoff_state_norm','decoder_history_gradient_stacked_norm','cutoff_history_gradient_norm']
        corr={}
        for key in covariates:
            x=np.array([r[key] for r in rows])
            for yname,y in [('sensitivity',a),('history_field_gradient',g)]:
                rho,p=spearmanr(x,y);corr[key+'__'+yname]={'spearman':float(rho) if np.isfinite(rho) else None,'pvalue':float(p) if np.isfinite(p) else None}
        one=int(np.argmax(g));summaries[case]={'correlations':corr,'largest_gradient_patient':one,'largest_gradient_projected_share':float(share[one]),
                                             'top5_norm_ranked_projected_share':float(share[order[:5]].sum()),'median_actual_gain_over_bound':float(np.median(actual/bound)),
                                             'maximum_cutoff_state_gradient':max(r['cutoff_history_gradient_norm'] for r in rows)}
        controls=np.column_stack([np.ones(len(rows))]+[rankdata([r[k] for r in rows]) for k in ['history_observations','missing_fraction','cutoff_state_norm']]+[rankdata([r['observed_numeric_scalars']+r['observed_categorical_scalars'] for r in rows])])
        tx=rankdata(t);tx=tx-controls@np.linalg.lstsq(controls,tx,rcond=None)[0]
        partial={}
        for name,y in [('sensitivity',a),('history_field_gradient',g)]:
            yy=rankdata(y);yy=yy-controls@np.linalg.lstsq(controls,yy,rcond=None)[0]
            value=np.corrcoef(tx,yy)[0,1] if np.std(tx)>1e-10 and np.std(yy)>1e-10 else np.nan
            partial[name]=float(value) if np.isfinite(value) else None
        summaries[case]['history_span_partial_rank_correlation']=partial
        (folder/'statistical_summary.json').write_text(json.dumps(summaries[case],indent=2)+'\n')
        with (folder/'patients.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if len(patients)>=2:
        stable=next((p for p in patients if p['case']['id']=='stable_matched_same_patients'),patients[0])
        states=np.array([r['cutoff_state'] for r in stable['patients']]);center=states.mean(0);_,_,vh=np.linalg.svd(states-center,full_matrices=False)
        all_colors=np.concatenate([np.log10(np.maximum([r['history_field_gradient_norm'] for r in item['patients']],1e-20)) for item in patients])
        fig,axes=plt.subplots(1,len(patients),figsize=(5*len(patients),4.5))
        for ax,item in zip(np.atleast_1d(axes),patients):
            s=np.array([r['cutoff_state'] for r in item['patients']]);xy=(s-center)@vh[:2].T;c=np.log10(np.maximum([r['history_field_gradient_norm'] for r in item['patients']],1e-20))
            plot=ax.scatter(xy[:,0],xy[:,1],c=c,cmap='magma',s=20,vmin=all_colors.min(),vmax=all_colors.max());ax.set(title=item['case']['id'],xlabel='Matched stable batch PC1',ylabel='Matched stable batch PC2');fig.colorbar(plot,ax=ax,label='log10 patient gradient norm')
        fig.tight_layout();fig.savefig(out/'patient_state_regions.png',dpi=160);plt.close(fig);links.append(('同一稳定batch拟合的PCA坐标中的患者梯度分布','patient_state_regions.png'))
        abnormal=next((p for p in patients if p['case']['id']=='abnormal_real_e46b121'),None)
        matched=next((p for p in patients if p['case']['id']=='stable_matched_same_patients'),None)
        if abnormal and matched:
            stable_by_id={r['subject_id']:r for r in matched['patients']};paired=[]
            for r in abnormal['patients']:
                b=stable_by_id[r['subject_id']];safe=lambda a,b:a/max(b,1e-30)
                decoder=safe(r['decoder_history_gradient_stacked_norm'],b['decoder_history_gradient_stacked_norm'])
                history=safe(safe(r['cutoff_history_gradient_norm'],r['decoder_history_gradient_stacked_norm']),safe(b['cutoff_history_gradient_norm'],b['decoder_history_gradient_stacked_norm']))
                encoder=safe(safe(r['encoder_field_gradient_norm'],r['cutoff_history_gradient_norm']),safe(b['encoder_field_gradient_norm'],b['cutoff_history_gradient_norm']))
                total=safe(r['encoder_field_gradient_norm'],b['encoder_field_gradient_norm'])
                paired.append({'patient_index':r['index'],'decoder_error_factor':decoder,'history_state_pullback_factor':history,
                               'encoder_parameter_vjp_factor':encoder,'encoder_gradient_growth':total,
                               'product_relative_error':abs(decoder*history*encoder-total)/max(total,1e-30),
                               'abnormal_history_gradient_norm':r['history_field_gradient_norm']})
            summaries['paired_path_growth']=paired
            (out/'paired_path_growth.json').write_text(json.dumps(paired,indent=2)+'\n')
    statefile=out/'encoder_path_followup/state_trace.json';jacfile=out/'encoder_path_followup/analysis.json'
    if statefile.exists():
        data=json.loads(statefile.read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for label,model in data['models'].items():
            rows=model['visits'];t=[r['time_hours'] for r in rows]
            axes[0].plot(t,[r['cutoff_wrt_postupdate_state_gain'] for r in rows],'o-',ms=3,label=label)
            axes[1].plot(t,[r['local_update_state_jacobian_largest_singular_value'] for r in rows],'o-',ms=3,label=label)
        axes[0].set(yscale='log',xlabel='Observation time (hours)',ylabel='Largest singular value',title='Final-state sensitivity to post-update state')
        axes[1].set(xlabel='Observation time (hours)',ylabel='Largest singular value',title='Local UPDATE state Jacobian')
        for ax in axes:ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(out/'encoder_recursive_sensitivity.png',dpi=180);fig.savefig(out/'encoder_recursive_sensitivity.pdf');plt.close(fig)
        links.append(('主导患者：向未来吸收历史的编码递归增益及每次UPDATE的状态Jacobian','encoder_recursive_sensitivity.png'))
    if jacfile.exists():
        d=json.loads(jacfile.read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.5));names=['INIT','UPDATE','ODE'];x=np.arange(3)
        for offset,label in [(-.18,'stable'),(.18,'abnormal')]:axes[0].bar(x+offset,[d['models'][label]['groups'][k]['jacobian_largest_singular_value'] for k in names],width=.36,label=label)
        axes[0].set(yscale='log',title='Encoder parameter Jacobians',ylabel='Largest singular value');axes[0].set_xticks(x,names);axes[0].legend()
        values=np.array([[d['crossed_unit_vjp_gains'][a][b] for b in ['stable','abnormal']] for a in ['stable','abnormal']]);im=axes[1].imshow(np.log10(values),cmap='viridis')
        for i in range(2):
            for j in range(2):axes[1].text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<100 else 'black')
        axes[1].set_xticks([0,1],['stable gradient','abnormal gradient']);axes[1].set_yticks([0,1],['stable Jacobian','abnormal Jacobian']);axes[1].set(title='Crossed unit-gradient VJP gain');fig.colorbar(im,ax=axes[1],label='log10 gain')
        fig.tight_layout();fig.savefig(out/'encoder_parameter_jacobians.png',dpi=180);plt.close(fig);links.append(('编码参数Jacobian与梯度方向交叉对照','encoder_parameter_jacobians.png'))
    return links,summaries
