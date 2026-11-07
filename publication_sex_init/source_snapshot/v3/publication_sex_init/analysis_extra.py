"""Clinical/local/robustness/cost diagnostics that share the publication adapter."""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from paper_odeae.data import Prefixes,collate as adni_collate
from paper_odeae.train import device_batch
from paper_odeae.models import feature_errors
from paper_odeae.solver import solver_statistics,integrate
from paper_odeae.jacobian import squared_jacobian_norm
from .analysis import load,encode,write_csv,finish
from .queue import OUT,atomic,read


def summaries(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return {"n":len(x),"mean":float(x.mean()) if len(x) else None,
            "median":float(np.median(x)) if len(x) else None,
            "p95":float(np.quantile(x,.95)) if len(x) else None}


def dataset_batches(payload,dataset,split="test"):
    seqs=[s for s in payload["sequences"] if s["split"]==split]
    if dataset=="adni":
        yield from DataLoader(Prefixes(seqs),batch_size=32,shuffle=False,collate_fn=adni_collate)
    else:
        from mimic_experiment.data import ICUContexts,collate
        context=ICUContexts(seqs,horizon=24)
        for shard in range(4):
            yield from DataLoader(Subset(context,range(shard,len(context),4)),batch_size=32,collate_fn=collate)


def numeric_std(payload):
    return np.array([max(float(np.concatenate([s["values"][:,j][s["mask"][:,j].astype(bool)]
                    for s in payload["sequences"] if s["split"]=="train"]).std()),1e-6)
                    if f["kind"]=="numerical" else 1. for j,f in enumerate(payload["schema"])])


@torch.no_grad()
def noise_forecast(model,payload,dataset,sigma,out,label):
    schema=payload["schema"];std=numeric_std(payload);static={"sex","age","marital_status","education_years","handedness","apoe4_count"}
    selected=torch.tensor([f["kind"]=="numerical" and f["name"] not in static for f in schema],device="cuda")
    clinical="mmse" if dataset=="adni" else "heart_rate"
    j=next(i for i,f in enumerate(schema) if f["name"]==clinical)
    head=sum(f.get("classes",1) for f in schema[:j]);scale=payload["metadata"]["scale"][j]
    counts=torch.zeros(len(schema),device="cuda",dtype=torch.float64);errors=torch.zeros_like(counts)
    patient_rows=[];horizon_rows=[]
    for bidx,cpu in enumerate(dataset_batches(payload,dataset)):
        batch=device_batch(cpu);clean=model(batch,sample=False)
        index=torch.arange(batch["values"].shape[1],device="cuda")[None]
        future=(index>=batch["cutoff"][:,None])&batch["valid"]
        if sigma:
            seed=int(hashlib.sha256(("noise/"+str(cpu["subject_ids"][0])+"/"+str(sigma)).encode()).hexdigest()[:12],16)
            generator=torch.Generator(device="cuda").manual_seed(seed)
            perturb=torch.randn(batch["values"].shape,device="cuda",generator=generator)*torch.tensor(std,device="cuda")*sigma
            inputmask=batch["context_mask"]*selected[None,None,:]
            noisy={**batch,"values":batch["values"]+perturb*inputmask}
            pred=model(noisy,sample=False)["prediction"]
        else:
            pred=clean["prediction"]
        error=feature_errors(pred,batch["values"],batch["mask"],schema)
        mask=batch["mask"]*future[...,None]
        errors+=(error*future[...,None]).sum((0,1)).double();counts+=mask.sum((0,1)).double()
        difference=(pred[...,head]-batch["values"][...,j])*scale
        shift=(pred[...,head]-clean["prediction"][...,head])*scale
        age0=batch["ages"].gather(1,(batch["cutoff"]-1)[:,None])
        spans=batch["ages"]-age0
        bins=[(0,1),(1,2),(2,3),(3,5),(5,float("inf"))] if dataset=="adni" else [(0,6),(6,12),(12,24)]
        for i,pid in enumerate(cpu["subject_ids"]):
            valid=mask[i,:,j].bool()
            patient_rows.append({"subject_id":str(pid),"prefix":int(batch["cutoff"][i]),
                "count":int(valid.sum()),"square_error_sum":float(difference[i,valid].square().sum()),
                "output_shift_square_sum":float(shift[i,valid].square().sum())})
            if sigma==0:
                for lo,hi in bins:
                    chosen=valid&(spans[i]>lo)&(spans[i]<=hi)
                    if bool(chosen.any()):
                        horizon_rows.append({"subject_id":str(pid),"lo":lo,"hi":hi if np.isfinite(hi) else "inf",
                            "count":int(chosen.sum()),"square_error_sum":float(difference[i,chosen].square().sum())})
    means=errors/counts.clamp_min(1);total_count=sum(r["count"] for r in patient_rows)
    result={"sigma":sigma,"clinical":clinical,"future_error":float(means.sum()),
            "clinical_rmse":(sum(r["square_error_sum"] for r in patient_rows)/max(1,total_count))**.5,
            "output_shift_rmse":(sum(r["output_shift_square_sum"] for r in patient_rows)/max(1,total_count))**.5,
            "fields":{f["name"]:{"loss":float(means[i]),"count":int(counts[i])} for i,f in enumerate(schema)}}
    write_csv(out/f"noise_{label}_{sigma}_patients.csv",patient_rows)
    if horizon_rows:
        write_csv(out/f"clinical_horizons_{label}.csv",horizon_rows)
    atomic(out/f"noise_{label}_{sigma}.json",result)
    return result


def jacobian_utility(cfg):
    from mimic_experiment.jacobian_utility_adni import tangent,field_jacobian
    out=Path(cfg["output"]);dataset=cfg["dataset"];results={}
    for label,checkpoint in cfg["models"].items():
        model,saved=load({"checkpoint":checkpoint})
        payload=pickle.loads(Path(saved["config"]["data"]).read_bytes());schema=payload["schema"]
        j=next(i for i,f in enumerate(schema) if f["name"]==("mmse" if dataset=="adni" else "heart_rate"))
        head=sum(f.get("classes",1) for f in schema[:j]);scale=payload["metadata"]["scale"][j]
        states=[];pids=[]
        if dataset=="adni":
            seqs=[s for s in payload["sequences"] if s["split"]=="test"]
            for start in range(0,len(seqs),32):
                group=seqs[start:start+32];batch=device_batch(adni_collate([(s,min(3,len(s["ages"]))) for s in group]))
                with torch.no_grad():states.append(encode(model,batch))
                pids += [str(s["subject_id"]) for s in group]
        else:
            for cpu in dataset_batches(payload,dataset):
                batch=device_batch(cpu)
                with torch.no_grad():states.append(encode(model,batch))
                pids += list(map(str,cpu["subject_ids"]))
        z=torch.cat(states).double();precise=copy.deepcopy(model).double().eval();field=precise.forward_ode
        generator=torch.Generator(device="cuda").manual_seed(71683)
        dirs=torch.randn(len(z),8,28,device="cuda",dtype=torch.float64,generator=generator)
        dirs/=dirs.norm(dim=-1,keepdim=True)
        perturb=.01*z.norm(dim=-1).clamp_min(1e-8)[:,None,None]*dirs
        horizons=[0,1,2,3,5] if dataset=="adni" else [0,6,12,24]
        flows=[];shifts=[];stats=[]
        with torch.no_grad():
            local=field_jacobian(field,z).square().sum((1,2))
            for horizon in horizons:
                endpoints=[];gains=[]
                for zz in z.split(32):
                    end,jac,calls=tangent(field,zz,horizon)
                    endpoints.append(end);gains.append(torch.linalg.svdvals(jac)[:,0]);stats.extend(calls)
                end=torch.cat(endpoints);flows.append(torch.cat(gains))
                flat=(z[:,None]+perturb).reshape(-1,28);noisy=[]
                for zz in flat.split(128):
                    noisy.append(integrate(field,zz,zz.new_full((len(zz),),horizon),1e-8,1e-10))
                noisy=torch.cat(noisy).reshape(len(z),8,28)
                shift=(precise.decoder(noisy)[...,head]-precise.decoder(end)[:,None,head])*scale
                shifts.append(shift.square().mean(1).sqrt())
            a,ja,_=tangent(field,z[:8],horizons[-1])
            b,jb,_=tangent(field,z[:8],horizons[-1],1e-10,1e-12)
        convergence={"endpoint_relative_error":float((a-b).norm()/b.norm().clamp_min(1e-30)),
                     "jacobian_relative_error":float((ja-jb).norm()/jb.norm().clamp_min(1e-30))}
        assert max(convergence.values())<1e-4,convergence
        with torch.enable_grad():
            x=z[:3].detach().requires_grad_(True)
            y=integrate(field,x,x.new_full((len(x),),horizons[-1]),1e-9,1e-11)
            autograd=torch.stack([torch.autograd.grad(y[:,i].sum(),x,retain_graph=True)[0] for i in range(28)],1)
            _,ref,_=tangent(field,x.detach(),horizons[-1],1e-10,1e-12)
            relative=float((autograd-ref).norm()/ref.norm().clamp_min(1e-30))
        assert relative<1e-4,relative
        flow=torch.stack(flows,1);shift=torch.stack(shifts,1)
        np.savez_compressed(out/f"flow_{label}.npz",gain=flow.cpu().numpy(),clinical_shift=shift.cpu().numpy(),
                            local_jacobian_squared=local.cpu().numpy(),horizons=horizons,patient_ids=np.array(pids))
        noises={str(s):noise_forecast(model,payload,dataset,s,out,label) for s in [0.,.01,.05]}
        results[label]={"local_jacobian_squared":summaries(local.cpu().numpy()),
                        "longest_flow_gain":summaries(flow[:,-1].cpu().numpy()),"noise":noises,
                        "convergence":convergence,"autograd_relative_error":relative,
                        "checkpoint":checkpoint,"checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        "solver_nfe":sum(r["nfe"] for r in stats),"physical_gpu":cfg["physical_gpu"]}
    atomic(out/"metrics.json",results)
    # Paired patients bootstrap of the principal clinical endpoint/noise shift.
    import pandas as pd
    boot={}
    for sigma in [0.,.01,.05]:
        frames=[]
        for label in ["main","no_jacobian"]:
            frame=pd.read_csv(out/f"noise_{label}_{sigma}_patients.csv").groupby("subject_id")[["square_error_sum","output_shift_square_sum","count"]].sum()
            frames.append(frame)
        ids=sorted(set(frames[0].index)&set(frames[1].index));rng=np.random.default_rng(0)
        selected=rng.integers(len(ids),size=(1000,len(ids)));vals=[]
        for f in frames:
            a=f.loc[ids];mean=a.square_error_sum.to_numpy()/a["count"].to_numpy().clip(1)
            vals.append(np.sqrt(mean[selected].mean(1)))
        boot[str(sigma)]={"main_minus_no_jacobian_clinical_rmse_95ci":np.quantile(vals[0]-vals[1],[.025,.975]).tolist(),
                          "patients":len(ids)}
    atomic(out/"paired_bootstrap.json",boot)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label in results:
        a=np.load(out/f"flow_{label}.npz")
        axes[0].plot(a["horizons"],np.median(a["gain"],axis=0),"o-",label=label)
        axes[1].plot([0,.01,.05],[results[label]["noise"][str(s)]["clinical_rmse"] for s in [0.,.01,.05]],"o-",label=label)
    axes[0].set(xlabel="Years" if dataset=="adni" else "Hours",ylabel="Median flow spectral gain")
    axes[1].set(xlabel="Input noise / training SD",ylabel="Clinical forecast RMSE")
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/"jacobian_utility.png",dpi=180);plt.close(fig);finish(cfg)


def mimic_clinical(cfg):
    from paper_odeae.diagnostics.clinical_velocity import clinical_values_and_rates,numeric_metrics
    out=Path(cfg["output"]);model,saved=load(cfg);payload=pickle.loads(Path(saved["config"]["data"]).read_bytes())
    source=OUT/"analysis/mimic/state";cache=np.load(source/"states.npz");meta=read(source/"metadata.json")
    schema=payload["schema"];fields={f["name"]:i for i,f in enumerate(schema)}
    markers=["heart_rate","map"];heads=[sum(f.get("classes",1) for f in schema[:fields[m]]) for m in markers]
    scale=torch.tensor([payload["metadata"]["scale"][fields[m]] for m in markers],device="cuda")
    offset=torch.tensor([payload["metadata"]["offset"][fields[m]] for m in markers],device="cuda")
    z=torch.tensor(cache["states"],device="cuda");head=torch.tensor(heads,device="cuda")
    with torch.no_grad():
        values,rates,v=clinical_values_and_rates(model,z,head,scale,offset)
    lookup={str(s["subject_id"]):s for s in payload["sequences"]};queries=[];seen=set()
    for i,r in enumerate(meta):
        pid=r["subject_id"];cut=r["prefix"]
        if (pid,cut) in seen:continue
        seen.add((pid,cut));seq=lookup[pid]
        for m in markers:
            j=fields[m]
            if not seq["mask"][cut-1,j]:continue
            future=np.flatnonzero(seq["mask"][cut:,j])
            if not len(future):continue
            target=cut+int(future[0]);dt=float(seq["ages"][target]-seq["ages"][cut-1])
            queries.append({"subject_id":pid,"split":r["split"],"marker":m,"state_index":i,"prefix":cut,
                            "target":target,"delta":dt,"real_rate":float((seq["values"][target,j]-seq["values"][cut-1,j])*payload["metadata"]["scale"][j]/dt)})
    for begin in range(0,len(queries),64):
        q=queries[begin:begin+64];indices=torch.tensor([r["state_index"] for r in q],device="cuda")
        dt=torch.tensor([r["delta"] for r in q],device="cuda")
        with torch.no_grad():
            end=model.solve(model.forward_ode,z[indices],dt)
            y=model.decoder(end)[:,head]*scale+offset
            pred=(y-values[indices])/dt[:,None]
        for k,r in enumerate(q):
            col=markers.index(r["marker"]);r.update(predicted_rate=float(pred[k,col]),instantaneous_rate=float(rates[r["state_index"],col]))
    write_csv(out/"interval_rates.csv",queries);metrics={}
    for marker in markers:
        rows=[r for r in queries if r["split"]=="test" and r["marker"]==marker]
        metrics[marker]={**numeric_metrics([r["real_rate"] for r in rows],[r["predicted_rate"] for r in rows]),
                         "patients":len({r["subject_id"] for r in rows})}
    atomic(out/"metrics.json",metrics);finish(cfg)


def mimic_local(cfg):
    from paper_odeae.diagnostics.manifold_core import knn
    from paper_odeae.diagnostics.local_dynamics import partial_rank,correlation
    out=Path(cfg["output"]);model,saved=load(cfg);source=OUT/"analysis/mimic/state"
    a=np.load(source/"states.npz");meta=read(source/"metadata.json")
    z=torch.tensor(a["states"],device="cuda",dtype=torch.float64);v=torch.tensor(a["velocities"],device="cuda",dtype=torch.float64)
    people={p:i for i,p in enumerate(sorted({r["subject_id"] for r in meta}))}
    pid=torch.tensor([people[r["subject_id"]] for r in meta],device="cuda")
    train=torch.tensor([r["split"]=="train" for r in meta],device="cuda");test=torch.tensor([r["split"]=="test" for r in meta],device="cuda")
    distances,neighbors=knn(z[test],z[train],40,pid[test],pid[train],distinct=True)
    tv=v[train][neighbors];tz=z[train][neighbors];qv=v[test]
    C=(qv[:,None]*tv).sum(-1)/(qv.norm(dim=-1)[:,None]*tv.norm(dim=-1)).clamp_min(1e-12)
    D=(qv[:,None]-tv).norm(dim=-1)/(z[test,None]-tz).norm(dim=-1).clamp_min(1e-8)
    with torch.no_grad():J=squared_jacobian_norm(model.forward_ode,z[test].float()).sqrt()
    indices=test.nonzero().flatten().tolist();rows=[]
    for i,at in enumerate(indices):
        r=meta[at];rows.append({"subject_id":r["subject_id"],"prefix":r["prefix"],"state_index":at,
                                "C20":float(C[i,:20].mean()),"D20":float(D[i,:20].mean()),"J":float(J[i])})
    write_csv(out/"state_local_metrics.csv",rows)
    payload=pickle.loads(Path(saved["config"]["data"]).read_bytes());seqs={str(s["subject_id"]):s for s in payload["sequences"]}
    forecasts=[]
    for begin in range(0,len(rows),32):
        q=rows[begin:begin+32]
        for horizon in [12,24]:
            chunk=[]
            for r in q:
                seq=seqs[r["subject_id"]];delta=seq["ages"]-seq["ages"][r["prefix"]-1]
                candidates=np.flatnonzero((np.arange(len(delta))>=r["prefix"])&(abs(delta-horizon)<=2))
                if len(candidates):
                    target=int(candidates[np.argmin(abs(delta[candidates]-horizon))]);chunk.append((r,seq,target))
            if not chunk:continue
            starts=torch.tensor([r["state_index"] for r,s,t in chunk],device="cuda")
            dt=torch.tensor([s["ages"][t]-s["ages"][r["prefix"]-1] for r,s,t in chunk],device="cuda")
            with torch.no_grad():
                end=model.solve(model.forward_ode,z[starts].float(),dt)
                pred=model.decoder(end)[:,None]
                values=torch.tensor(np.stack([s["values"][t] for r,s,t in chunk]),device="cuda")[:,None]
                mask=torch.tensor(np.stack([s["mask"][t] for r,s,t in chunk]),device="cuda")[:,None]
                errors=feature_errors(pred,values,mask,payload["schema"]).sum((1,2))/mask.sum((1,2)).clamp_min(1)
            for i,(r,s,t) in enumerate(chunk):
                forecasts.append({**r,"horizon":horizon,"error":float(errors[i]),"history":r["prefix"],"coverage":float(s["mask"][:r["prefix"]].mean())})
    write_csv(out/"forecast_errors.csv",forecasts)
    result={"C20":summaries(C[:,:20].mean(1).cpu().numpy()),"D20":summaries(D[:,:20].mean(1).cpu().numpy()),"J":summaries(J.cpu().numpy()),"horizons":{}}
    for h in [12,24]:
        q=[r for r in forecasts if r["horizon"]==h]
        result["horizons"][str(h)]={"patients":len(q),"D_error_spearman":correlation([r["D20"] for r in q],[r["error"] for r in q]),
                                   "partial_rank":partial_rank([r["D20"] for r in q],[r["error"] for r in q],
                                                               np.array([[r["history"],r["coverage"]] for r in q]))}
    atomic(out/"metrics.json",result);finish(cfg)


def cost(cfg):
    out=Path(cfg["output"]);model,saved=load(cfg);payload=pickle.loads(Path(saved["config"]["data"]).read_bytes())
    rows=[];torch.cuda.reset_peak_memory_stats();current={"stats":None,"phase":"generation","native_nfe":0}
    # Mark actual solver calls without changing arithmetic or solver tolerances.
    if hasattr(model,"encode"):
        original_encode=model.encode
        def wrapped_encode(*args,**kwargs):
            if current["stats"] is not None:current["stats"]["phase"]="encoding"
            value=original_encode(*args,**kwargs)
            if current["stats"] is not None:current["stats"]["phase"]="generation"
            return value
        model.encode=wrapped_encode
    hooks=[]
    if hasattr(model,"core"):
        for label,module in [("generation",model.core.ode),("encoding",getattr(getattr(model.core,"encoder",None),"ode",None))]:
            if module is not None:
                def mark(module,args,label=label):
                    st=current["stats"]
                    if st is not None and st["calls"]:st["calls"][-1]["phase"]=label
                hooks.append(module.register_forward_pre_hook(mark))
    elif hasattr(model,"backward_ode") and model.backward_ode is not model.forward_ode:
        for label,module in [("encoding",model.backward_ode),("generation",model.forward_ode)]:
            def mark(module,args,label=label):
                st=current["stats"]
                if st is not None and st["calls"]:st["calls"][-1]["phase"]=label
            hooks.append(module.register_forward_pre_hook(mark))
    elif hasattr(model,"dynamics"):
        def mark_native(module,args):
            if current["stats"] is not None:current["native_nfe"]+=1
        hooks.append(model.dynamics.register_forward_pre_hook(mark_native))
    for bi,cpu in enumerate(dataset_batches(payload,cfg["dataset"])):
        batch=device_batch(cpu)
        with torch.no_grad():
            if bi==0:
                model(batch,sample=False)
            torch.cuda.synchronize();start=time.perf_counter()
            with solver_statistics(trace=bi==0) as stats:
                current["stats"]=stats;current["native_nfe"]=0
                model(batch,sample=False)
                current["stats"]=None
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if bi==0:atomic(out/"step_trace.json",stats)
        rows.append({"batch":bi,"patients":len(batch["cutoff"]),"seconds":elapsed,
                     "nfe":sum(r["nfe"] for r in stats["calls"]),
                     "encoding_nfe":sum(r["nfe"] for r in stats["calls"] if r["phase"]=="encoding"),
                     "generation_nfe":sum(r["nfe"] for r in stats["calls"] if r["phase"]=="generation"),
                     "native_rk4_nfe":current["native_nfe"],
                     "accepted":sum(r["accepted"] for r in stats["calls"]),
                     "rejected":sum(r["rejected"] for r in stats["calls"])})
    for hook in hooks:hook.remove()
    write_csv(out/"batches.csv",rows)
    atomic(out/"metrics.json",{"nfe":sum(r["nfe"] for r in rows),"accepted":sum(r["accepted"] for r in rows),
           "rejected":sum(r["rejected"] for r in rows),"milliseconds_per_patient":sum(r["seconds"] for r in rows)*1000/sum(r["patients"] for r in rows),
           "encoding_nfe":sum(r["encoding_nfe"] for r in rows),"generation_nfe":sum(r["generation_nfe"] for r in rows),
           "native_rk4_nfe":sum(r["native_rk4_nfe"] for r in rows),
           "peak_memory_gib":torch.cuda.max_memory_allocated()/2**30})
    finish(cfg)


def stability(cfg):
    from .models import objective
    from .train import data_context
    from mimic_experiment.jacobian_utility_adni import tangent
    out=Path(cfg["output"]);rows=[]
    for label,checkpoint in cfg["models"].items():
        model,saved=load({"checkpoint":checkpoint})
        for p in model.parameters():p.requires_grad_(True)
        config=dict(saved["config"]);payload,schema,splits,batches,evaluate=data_context(config)
        batch=next(batches(1))
        small={k:v[:8] if torch.is_tensor(v) else v for k,v in batch.items()}
        small["max_context"]=int(small["cutoff"].max())
        before=[p.detach().clone() for p in model.parameters()]
        pred=model(small,sample=False);loss,parts=objective(model,pred,small,schema,config)
        params=tuple(model.forward_ode.parameters())
        def grad(value):
            g=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
            return torch.cat([(torch.zeros_like(p) if x is None else x).flatten() for x,p in zip(g,params)])
        errors=feature_errors(pred["prediction"],small["values"],small["mask"],schema)
        index=torch.arange(errors.shape[1],device="cuda")[None]
        denominator=small["mask"].sum((0,1)).clamp_min(1)
        history_loss=((errors*(index<small["cutoff"][:,None])[...,None]).sum((0,1))/denominator).sum()
        future_loss=((errors*(index>=small["cutoff"][:,None])[...,None]).sum((0,1))/denominator).sum()
        h=grad(history_loss);f=grad(future_loss)
        cosine=float(torch.dot(h,f)/(h.norm()*f.norm()).clamp_min(1e-30))
        double=copy.deepcopy(model).double();z=pred["mu"].detach().double()
        spans=small["ages"].gather(1,(small["cutoff"]-1)[:,None]).squeeze(1)-small["ages"][:,0]
        gains=[]
        with torch.no_grad():
            for sign in [-1,1]:
                gain=[]
                for i in range(len(z)):
                    _,jac,_=tangent(double.forward_ode,z[i:i+1],float(spans[i])*sign)
                    gain.append(float(torch.linalg.svdvals(jac)[0,0]))
                gains.append(gain)
        anchor,_,trace=model.encode(small,return_trace=True)
        recursive=[];update=[]
        probes=sorted({0,max(0,len(trace)//2),len(trace)-1})
        for ti in probes:
            item=trace[ti];node=item["after"]
            jac=torch.stack([torch.autograd.grad(anchor[:,j].sum(),node,retain_graph=True)[0]
                             for j in range(model.latent)],1)
            active=item["active"]
            recursive.extend(torch.linalg.svdvals(jac[active])[:,0].detach().cpu().tolist())
            # Exact local residual UPDATE derivative with the same observed input.
            from paper_odeae.models import encode_values
            inputs=encode_values(small["values"],small["context_mask"],schema)
            ii=item["index"];rr=torch.arange(len(ii),device="cuda")
            u=model.update;x=torch.cat([item["before"].detach(),inputs[rr,ii]],-1)
            derivative=1-u[1](u[0](x)).square()
            ju=(u[2].weight[None]*derivative[:,None])@u[0].weight[:,:model.latent]
            identity=torch.eye(model.latent,device="cuda")[None]
            actual=(1-u.alpha)*identity+u.alpha*ju
            update.extend(torch.linalg.svdvals(actual[active])[:,0].detach().cpu().tolist())
        assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
        rows.append({"model":label,"checkpoint_epoch":saved["epoch"],"history_gradient_norm":float(h.norm()),
                     "future_gradient_norm":float(f.norm()),"gradient_cosine":cosine,
                     "backward_gain":summaries(gains[0]),"forward_gain":summaries(gains[1]),
                     "recursive_state_gain":summaries(recursive),"local_update_gain":summaries(update),
                     "gradient_normalization":"historical/future contributions share original full-batch field denominators",
                     "batch_kind":"fixed eight-patient diagnostic; no parameter update",
                     "patient_ids":small.get("subject_ids",[])[:8],"physical_gpu":cfg["physical_gpu"]})
    atomic(out/"metrics.json",rows);finish(cfg)


def horizons(cfg):
    out=Path(cfg["output"]);model,saved=load(cfg);payload=pickle.loads(Path(saved["config"]["data"]).read_bytes())
    schema=payload["schema"];numeric=torch.tensor([f["kind"]=="numerical" for f in schema],device="cuda")
    std=torch.tensor(numeric_std(payload),device="cuda",dtype=torch.float64)
    bins=[(0,1),(1,2),(2,3),(3,5)] if cfg["dataset"]=="adni" else [(0,6),(0,12),(0,24)]
    sums=torch.zeros(len(bins),2,len(schema),device="cuda",dtype=torch.float64);people=[set() for _ in bins]
    for cpu in dataset_batches(payload,cfg["dataset"]):
        batch=device_batch(cpu);index=torch.arange(batch["values"].shape[1],device="cuda")[None]
        anchor=batch["ages"].gather(1,(batch["cutoff"]-1)[:,None]);delta=batch["ages"]-anchor
        future=(index>=batch["cutoff"][:,None])&batch["valid"]
        with torch.no_grad():
            pred=model(batch,sample=False)["prediction"]
            errors=feature_errors(pred,batch["values"],batch["mask"],schema)
        for i,(lo,hi) in enumerate(bins):
            valid=future&(delta>lo)&(delta<=hi);mask=batch["mask"]*valid[...,None]
            sums[i,0]+=(errors*valid[...,None]).sum((0,1)).double();sums[i,1]+=mask.sum((0,1)).double()
            for b,pid in enumerate(cpu["subject_ids"]):
                if bool(mask[b].sum()):people[i].add(str(pid))
    rows=[]
    for i,(lo,hi) in enumerate(bins):
        valid=sums[i,1]>0;mean=sums[i,0]/sums[i,1].clamp_min(1)
        rows.append({"lo":lo,"hi":hi,"patients":len(people[i]),"observed_targets":int(sums[i,1].sum()),
                     "future_error":float(mean[valid].sum()),
                     "mean_standardized_numeric_mse":float((mean[numeric&valid]/std[numeric&valid].square()).mean())})
    atomic(out/"metrics.json",rows);finish(cfg)


def run(cfg):
    {"jacobian_utility":jacobian_utility,"clinical":mimic_clinical,"local":mimic_local,
     "cost":cost,"stability":stability,"horizons":horizons}[cfg["analysis"]](cfg)
