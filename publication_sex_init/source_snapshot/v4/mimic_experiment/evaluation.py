"""Common single-GPU MIMIC scoring, matching the completed four-GPU reference batches."""
import torch
from torch.utils.data import DataLoader,Subset
from paper_odeae.models import feature_errors,prior_errors
from .data import ICUContexts,collate


def carry_forward(batch,schema,category_modes,numeric_defaults):
    values=batch['values'];b,n,_=values.shape
    last=values.new_tensor(numeric_defaults)[None].expand(b,-1).clone()
    for j,field in enumerate(schema):
        if field['kind']=='categorical':last[:,j]=category_modes[j]
    for i in range(batch['max_context']):last=torch.where(batch['context_mask'][:,i].bool(),values[:,i],last)
    return last[:,None].expand(-1,n,-1)


@torch.no_grad()
def evaluate(model,sequences,schema,cfg,raw_std,category_modes,horizon=24):
    if next(model.parameters()).device.type!='cuda':raise RuntimeError('CUDA evaluation required')
    model.eval();dataset=ICUContexts(sequences,horizon=horizon)
    # Serial execution of the reference's four fixed patient shards on this GPU.
    # Thus every model sees identical query batches, including adaptive-solver grouping.
    keys=['interpolation','forecast','overall','prior','forecast_6h','forecast_12h','forecast_24h']
    totals=torch.zeros(len(keys),2,len(schema),device='cuda',dtype=torch.float64)
    absolute=torch.zeros(len(schema),device='cuda',dtype=torch.float64)
    naive_square=torch.zeros_like(absolute);naive_abs=torch.zeros_like(absolute)
    accuracy=torch.zeros_like(absolute);naive_accuracy=torch.zeros_like(absolute)
    jacobian=torch.zeros(2,device='cuda',dtype=torch.float64)
    for shard in range(4):
        loader=DataLoader(Subset(dataset,range(shard,len(dataset),4)),batch_size=32,collate_fn=collate)
        for cpu in loader:
            batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in cpu.items()};out=model(batch,sample=False)
            errors=feature_errors(out['prediction'],batch['values'],batch['mask'],schema)
            index=torch.arange(batch['values'].shape[1],device='cuda')[None]
            future=(index>=batch['cutoff'][:,None])&batch['valid']
            regions=[index<batch['cutoff'][:,None],future,batch['valid'],None,
                     future&(batch['ages']<=horizon+6),future&(batch['ages']<=horizon+12),future&(batch['ages']<=horizon+24)]
            for i,region in enumerate(regions):
                if region is None:
                    if 'prior_prediction' in out:
                        pe,mask=prior_errors(out,batch,schema);totals[i,0]+=pe.sum((0,1)).double();totals[i,1]+=mask.sum((0,1)).double()
                else:
                    totals[i,0]+=(errors*region[...,None]).sum((0,1)).double();totals[i,1]+=(batch['mask']*region[...,None]).sum((0,1)).double()
            if 'jacobian_states' in out:
                from paper_odeae.jacobian import jacobian_regularization
                _,js,jc=jacobian_regularization(model,out,batch);jacobian[0]+=js.double();jacobian[1]+=jc
            naive=carry_forward(batch,schema,category_modes,cfg['normalized_train_means']);at=0;observed=batch['mask']*future[...,None]
            for j,field in enumerate(schema):
                width=field.get('classes',1)
                if field['kind']=='numerical':
                    diff=out['prediction'][...,at]-batch['values'][...,j];nd=naive[...,j]-batch['values'][...,j]
                    absolute[j]+=(diff.abs()*observed[...,j]).sum().double()
                    naive_square[j]+=(nd.square()*observed[...,j]).sum().double();naive_abs[j]+=(nd.abs()*observed[...,j]).sum().double()
                else:
                    guessed=out['prediction'][...,at:at+width].argmax(-1)
                    accuracy[j]+=((guessed==batch['values'][...,j])*observed[...,j]).sum().double()
                    naive_accuracy[j]+=((naive[...,j]==batch['values'][...,j])*observed[...,j]).sum().double()
                at+=width
    numeric=torch.tensor([f['kind']=='numerical' for f in schema],device='cuda');result={}
    for key,total in zip(keys,totals):
        valid=total[1]>0;means=total[0]/total[1].clamp_min(1)
        mse=means[numeric&valid].sum();cce=means[(~numeric)&valid].sum()
        result[key]={'reconstruction_error':float(mse+cce),'sum_feature_mse':float(mse),'sum_feature_cce':float(cce),
                     'observed_scalars':int(total[1].sum()),'per_field':{f['name']:{'loss':float(means[j]),'observations':int(total[1,j])} for j,f in enumerate(schema) if valid[j]}}
    clinical={};reference={};standardized=[];naive_standardized=[]
    for j,field in enumerate(schema):
        count=totals[1,1,j]
        if not count:continue
        if field['kind']=='numerical':
            scale=cfg['raw_scale'][j];sd=max(raw_std[j],1e-6)
            rmse=float((totals[1,0,j]/count).sqrt())*scale;nrmse=float((naive_square[j]/count).sqrt())*scale
            clinical[field['name']]={'unit':field['unit'],'observations':int(count),'mae':float(absolute[j]/count)*scale,'rmse':rmse,'standardized_rmse':rmse/sd}
            reference[field['name']]={'unit':field['unit'],'mae':float(naive_abs[j]/count)*scale,'rmse':nrmse,'standardized_rmse':nrmse/sd}
            standardized.append((rmse/sd)**2);naive_standardized.append((nrmse/sd)**2)
        else:
            clinical[field['name']]={'observations':int(count),'accuracy':float(accuracy[j]/count)}
            reference[field['name']]={'accuracy':float(naive_accuracy[j]/count)}
    result.update(patients=len(dataset),context_hours=horizon,clinical_forecast=clinical,carry_forward=reference,
                  mean_standardized_numeric_mse=sum(standardized)/len(standardized) if standardized else None,
                  carry_forward_mean_standardized_numeric_mse=sum(naive_standardized)/len(naive_standardized) if naive_standardized else None)
    if jacobian[1]:result['dynamics']={'jacobian_frobenius_squared':float(jacobian[0]/jacobian[1]),'states':int(jacobian[1])}
    return result
