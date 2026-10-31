"""Public-query policy rollouts; future decisions use each branch's own predicted X2."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import collate,cuda_batch,atomic


def policy_action(policy,relative_time,risk,specs):
    action=torch.zeros_like(policy)
    action=torch.where(policy==1,torch.ones_like(action),action)
    action=torch.where(policy==2,(relative_time>=specs['delay_years']).long(),action)
    for p in [3,4]:action=torch.where(policy==p,(risk.double()>specs['thresholds'][str(p)]).long(),action)
    return action


def event_plan(query):
    fork=float(query.fork_time);qt=np.asarray(query.trajectory_times,np.float64)
    visits=np.asarray(query.future_visit_times,np.float64)-fork
    if abs(visits[0])>1e-9 or abs(qt[0])>1e-12:raise ValueError('Fork must be an initial public decision and query')
    visits[0]=0.;times=sorted(set(qt.tolist()+visits.tolist()))
    qindex={float(t):i for i,t in enumerate(qt)};decisions=set(visits.tolist())
    return {'times':np.asarray(times,np.float64),'query_indices':np.asarray([qindex.get(t,-1) for t in times],np.int64),
            'decisions':np.asarray([t in decisions for t in times],bool),'query_times':qt}


@torch.no_grad()
def predict(model,sequences,queries,specs,normalization,output,disabled=False,patient_batch=32,policies=(0,1,2,3,4)):
    model.eval();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    by_id={s['patient_id']:s for s in sequences};grouped=dict(tuple(queries.groupby('patient_id',sort=True)))
    if set(by_id)!=set(grouped):raise ValueError('Public history and query coverage differ')
    offset=torch.tensor(normalization['offset'],device='cuda');scale=torch.tensor(normalization['scale'],device='cuda')
    ids=sorted(by_id);predicted=[];intervals=[];device=str(next(model.parameters()).device)
    for start in range(0,len(ids),patient_batch):
        chunk=ids[start:start+patient_batch];items=[(by_id[pid],len(by_id[pid]['times'])) for pid in chunk]
        batch=cuda_batch(collate(items));anchor,held,_=model.encode(batch)
        count=len(policies);state=anchor.repeat_interleave(count,0);current=held.repeat_interleave(count)
        policy=torch.tensor(policies,device='cuda').repeat(len(chunk))
        plans=[]
        for pid in chunk:
            rows=grouped[pid]
            if set(rows.policy_id)!={0,1,2,3,4}:raise ValueError('Five public policy queries required')
            first=rows.iloc[0]
            for _,row in rows.iterrows():
                if not np.array_equal(row.trajectory_times,first.trajectory_times) or not np.array_equal(row.future_visit_times,first.future_visit_times):raise ValueError('Paired policies must share exogenous schedules')
            if abs(by_id[pid]['times'][-1]-float(first.fork_time))>1e-9:raise ValueError('History must end at the public fork')
            plans.append(event_plan(first))
        length=max(len(plan['times']) for plan in plans);b=len(state)
        times=np.zeros((len(chunk),length),np.float64);valid=np.zeros_like(times,bool);decision=np.zeros_like(times,bool)
        for i,plan in enumerate(plans):
            n=len(plan['times']);times[i,:n]=plan['times'];times[i,n:]=plan['times'][-1];valid[i,:n]=True;decision[i,:n]=plan['decisions']
        clock=torch.tensor(np.repeat(times,count,0),device='cuda',dtype=torch.float64)
        active=torch.tensor(np.repeat(valid,count,0),device='cuda');dec=torch.tensor(np.repeat(decision,count,0),device='cuda')
        fork_risk=torch.tensor([by_id[pid]['raw_values'][-1,2] for pid in chunk],device='cuda',dtype=torch.float64).repeat_interleave(count)
        values=state.new_zeros(b,length,3);actions=torch.zeros(b,length,device='cuda',dtype=torch.long);previous=clock[:,0]
        for i in range(length):
            dt=torch.where(active[:,i],clock[:,i]-previous,torch.zeros_like(previous)).to(state.dtype)
            state=model.solve(state,dt);raw=model.decoder(state)*scale+offset
            values[:,i]=raw
            risk=torch.where(clock[:,i].abs()<1e-12,fork_risk,raw[:,2].double())
            proposed=policy_action(policy,clock[:,i],risk,specs);choosing=active[:,i]&dec[:,i]
            state=model.change_action(state,current,proposed,choosing,disabled)
            current=torch.where(choosing,proposed,current);actions[:,i]=current;previous=torch.where(active[:,i],clock[:,i],previous)
        raw=values.cpu().numpy();a=actions.cpu().numpy()
        for j,pid in enumerate(chunk):
            plan=plans[j];query_events=np.flatnonzero(plan['query_indices']>=0)
            for number,p in enumerate(policies):
                sample=j*count+number
                for event in query_events:
                    qi=int(plan['query_indices'][event]);v=raw[sample,event]
                    predicted.append((pid,int(p),float(plan['query_times'][qi]),float(v[0]),float(v[1]),float(v[2])))
                decision_events=np.flatnonzero(plan['decisions']);begin=0.;treatment=int(a[sample,decision_events[0]])
                for event in decision_events[1:]:
                    time=float(plan['times'][event]);new=int(a[sample,event])
                    if new!=treatment:
                        intervals.append((pid,int(p),begin,time,treatment));begin=time;treatment=new
                end=float(plan['query_times'][-1]);intervals.append((pid,int(p),begin,end,treatment))
        atomic(output/'progress.json',{'patients_completed':min(start+len(chunk),len(ids)),'patients_total':len(ids),'device':device,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'action_updates_disabled':disabled})
    frame=pd.DataFrame(predicted,columns=['patient_id','policy_id','relative_time','Y_pred','X1_pred','X2_pred'])
    treatments=pd.DataFrame(intervals,columns=['patient_id','policy_id','t_start','t_end','treatment'])
    expected=sum(len(grouped[pid].iloc[0].trajectory_times)*len(policies) for pid in ids)
    if len(frame)!=expected or frame.duplicated(['patient_id','policy_id','relative_time']).any():raise ValueError('Incomplete or duplicate predictions')
    if not np.isfinite(frame[['relative_time','Y_pred','X1_pred','X2_pred']]).all().all():raise FloatingPointError('Nonfinite counterfactual output')
    for name,data in [('predictions',frame),('predicted_treatments',treatments)]:
        temporary=output/f'{name}.partial.parquet';data.to_parquet(temporary,index=False);temporary.replace(output/f'{name}.parquet')
    hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['predictions.parquet','predicted_treatments.parquet']}
    atomic(output/'sealed.json',{'patients':len(ids),'policies':list(policies),'prediction_rows':len(frame),'treatment_intervals':len(treatments),
                                'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'device':device,'action_updates_disabled':disabled,'sha256':hashes,
                                'future_policy_risk_source':'observed public X2 at fork; own predicted X2 at later decisions'})
    return frame,treatments
