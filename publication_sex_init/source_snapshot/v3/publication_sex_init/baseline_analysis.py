"""Matched baseline state/field/case and cost analysis on the assigned GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from paper_odeae.models import PaperModel,encode_values
from paper_odeae.data import Prefixes,collate as adni_collate
from paper_odeae.train import device_batch
from .queue import atomic,read,OUT
from .analysis import plot_state,finish


def load(cfg):
    saved=torch.load(cfg["checkpoint"],map_location="cuda",weights_only=False)
    config=dict(saved["config"]);schema=saved["schema"]
    if cfg["model_key"]=="healthflux":
        assert Path(sys.prefix).name=="healthflux"
        if cfg["dataset"]=="adni":
            from healthflux.model import HealthFlux
            model=HealthFlux(schema,config,saved.get("input_normalization")).cuda()
        else:
            from mimic_experiment.healthflux_icu import HealthFluxICU
            model=HealthFluxICU(schema,config,saved.get("input_normalization")).cuda()
    elif config.get("capacity_matched") or cfg.get("family")=="matched_baseline":
        from capacity_matched.models import make_model
        config["dataset_name"]=cfg["dataset"];model=make_model(schema,config).cuda()
    elif cfg["model_key"]=="deterministic":
        if config.get("family"):
            from .models import make_model
            model=make_model(schema,config)
        else:
            from paper_odeae.models import paper_model_from_config
            model=paper_model_from_config(schema,config).cuda()
    else:
        raise ValueError(cfg)
    model.load_state_dict(saved["state_dict"]);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,saved


@torch.no_grad()
def state_and_velocity(model,batch,cfg):
    key=cfg["model_key"];dataset=cfg["dataset"];ages=batch["ages"];cuts=batch["cutoff"]
    if key=="healthflux":
        if dataset=="adni":
            z,stochastic,context,kl=model.encode(batch,sample=False)
            age=ages.gather(1,(cuts-1)[:,None]).squeeze(1);velocity=model.vector_field(z,age,context)/10
        else:
            z,stochastic,context,age0,kl,count=model.encode_icu(batch,sample=False)
            age=batch["initialization_age"];velocity=model.vector_field(z,age,context)
        return z,velocity
    if isinstance(model,PaperModel):
        x=encode_values(batch["values"],batch["context_mask"],model.schema)
        last=ages.gather(1,(cuts-1)[:,None]).squeeze(1)
        h=model.init(batch.get("initialization_age",last)[:,None]/100);previous=last
        rows=torch.arange(len(h),device="cuda")
        for j in range(batch["max_context"]):
            at=(cuts-1-j).clamp_min(0);active=j<cuts;time=ages.gather(1,at[:,None]).squeeze(1)
            evolved=model.solve(model.backward_ode,h,torch.where(active,time-previous,torch.zeros_like(time)))
            h=torch.where(active[:,None],model.update(torch.cat([evolved,x[rows,at]],-1)),h)
            previous=torch.where(active,time,previous)
        z=h if model.kind=="deterministic" else h.chunk(2,-1)[0]
        previous=ages[:,0]
        for j in range(1,batch["max_context"]):
            active=j<cuts;time=ages[:,j]
            z=model.solve(model.forward_ode,z,torch.where(active,time-previous,torch.zeros_like(time)))
            previous=torch.where(active,time,previous)
        return z,model.forward_ode(z)
    if key=="latent_ode":
        z=model(batch,sample=False)["mu"]
        return z,model.core.ode(None,z)/(10 if dataset=="adni" else 1)
    from paper_odeae.baselines import expanded_mask,feedback
    x=encode_values(batch["values"],batch["context_mask"],model.schema)
    masks=expanded_mask(batch["context_mask"],model.schema);z=x.new_zeros(len(x),model.core.hidden)
    clock=(ages-ages[:,:1])/(10 if dataset=="adni" else 1)
    for j in range(batch["max_context"]):
        active=j<cuts
        if j:
            z=model.solve(model.core.ode,z,torch.where(active,clock[:,j]-clock[:,j-1],torch.zeros_like(clock[:,j])))
        pred=model.core.decoder(z);filled=torch.where(masks[:,j].bool(),x[:,j],feedback(pred,model.schema))
        mask=masks[:,j]*active[:,None]
        if dataset=="mimic":
            last=ages.gather(1,(cuts-1)[:,None]).squeeze(1)
            age=(batch["initialization_age"]+(ages[:,j]-last)/8766)/100
            filled=torch.cat([filled,age[:,None]],-1);mask=torch.cat([mask,active[:,None]],-1)
        proposed=model.core.cell(z,filled,mask)
        z=torch.where(active[:,None],proposed,z)
    return z,model.core.ode(None,z)/(10 if dataset=="adni" else 1)


def cases(model,payload,mean,basis,out,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seqs={str(s["subject_id"]):s for s in payload["sequences"] if s["split"]=="test"}
    ids=["003_S_4900","002_S_4219","002_S_0619"] if cfg["dataset"]=="adni" else sorted(seqs)[:3]
    fig,axes=plt.subplots(3,2,figsize=(12,12));records=[]
    collate=adni_collate
    if cfg["dataset"]=="mimic":
        from mimic_experiment.data import collate
    for row,pid in enumerate(ids):
        seq=seqs[pid];cut=min(3,len(seq["ages"])-1) if cfg["dataset"]=="adni" else int(np.searchsorted(seq["ages"],24,side="right"))
        with torch.no_grad():
            batch=device_batch(collate([(seq,cut)]));pred=model(batch,sample=False)["prediction"]
            posterior=[]
            for c in range(1,cut+1):
                z,v=state_and_velocity(model,device_batch(collate([(seq,c)])),cfg);posterior.append(z[0])
        xy=(torch.stack(posterior)-mean)@basis
        axes[row,0].plot(*xy.cpu().numpy().T,"o-");axes[row,0].set_title(pid+" / history estimates")
        marker="mmse" if cfg["dataset"]=="adni" else "heart_rate"
        j=next(i for i,f in enumerate(payload["schema"]) if f["name"]==marker);head=sum(f.get("classes",1) for f in payload["schema"][:j])
        scale=payload["metadata"]["scale"][j];offset=payload["metadata"]["offset"][j]
        valid=seq["mask"][:,j].astype(bool);real=seq["values"][:,j]*scale+offset
        estimate=pred[0,:,head].cpu().numpy()*scale+offset
        axes[row,1].plot(seq["ages"][valid],real[valid],"o",label="Observed")
        axes[row,1].plot(seq["ages"],estimate,"-",label="Native model prediction")
        axes[row,1].axvline(seq["ages"][cut-1],c="gray");axes[row,1].legend()
        records.append({"subject_id":pid,"cutoff":cut,"prediction":estimate.tolist(),"posterior_xy":xy.cpu().tolist()})
    fig.tight_layout();fig.savefig(out/"clinical_cases.png",dpi=180);plt.close(fig);atomic(out/"patient_paths.json",records)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args()
    cfg=read(args.config);out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and os.environ["CUDA_VISIBLE_DEVICES"]==str(cfg["physical_gpu"])
    torch.cuda.set_device(0);torch.set_num_threads(1)
    model,saved=load(cfg)
    if cfg["analysis"]=="cost":
        from .analysis_extra import cost
        import publication_sex_init.analysis_extra as extra
        extra.load=lambda _: (model,saved)
        cost(cfg);return
    payload=pickle.loads(Path(saved["config"]["data"]).read_bytes());meta=payload["metadata"]
    pos={f["name"]:i for i,f in enumerate(payload["schema"])};zs=[];vs=[];metadata=[]
    for split in ["train","validation","test"]:
        seqs=[s for s in payload["sequences"] if s["split"]==split]
        if cfg["dataset"]=="adni":
            loaders=[(None,DataLoader(Prefixes(seqs),batch_size=64,collate_fn=adni_collate))]
        else:
            from mimic_experiment.data import ICUContexts,collate
            loaders=[(h,DataLoader(ICUContexts(seqs,horizon=h),batch_size=64,collate_fn=collate))
                     for h in ([6,12,18,24] if split=="train" else [24])]
        for hours,loader in loaders:
            for cpu in loader:
                z,v=state_and_velocity(model,device_batch(cpu),cfg);zs.append(z);vs.append(v)
                for i,c in enumerate(cpu["cutoff"].tolist()):
                    r={"subject_id":str(cpu["subject_ids"][i]),"split":split,"prefix":c,"context_hours":hours}
                    for marker in ["diagnosis","gcs_motor"]:
                        j=pos.get(marker);r[marker]=float(cpu["values"][i,c-1,j]*meta["scale"][j]+meta["offset"][j]) if j is not None and cpu["mask"][i,c-1,j] else None
                    metadata.append(r)
    z=torch.cat(zs);v=torch.cat(vs);train=torch.tensor([r["split"]=="train" for r in metadata],device="cuda")
    mean=z[train].mean(0);x=(z[train]-mean).double()
    eig,basis=torch.linalg.eigh(x.T@x);eig=eig.flip(0).clamp_min(0);basis=basis.flip(1)[:,:2].to(z.dtype)
    ve,vb=torch.linalg.eigh(v[train].double().T@v[train].double());vb=vb.flip(1)[:,:2].to(v.dtype)
    ratio=eig/eig.sum().clamp_min(1e-30);xy=(z-mean)@basis;pv=v@basis;p=ratio[ratio>0]
    stats={"model_label":cfg["model_key"],"checkpoint":cfg["checkpoint"],"checkpoint_sha256":hashlib.sha256(Path(cfg["checkpoint"]).read_bytes()).hexdigest(),
           "checkpoint_epoch":saved["epoch"],"state_variance_retained_2d":float(ratio[:2].sum()),
           "velocity_energy_retained_2d":float(pv.square().sum()/v.square().sum().clamp_min(1e-30)),
           "velocity_plane_energy_retained_2d":float((v@vb).square().sum()/v.square().sum().clamp_min(1e-30)),
           "effective_rank":float(torch.exp(-(p*p.log()).sum())),"parameters":sum(p.numel() for p in model.parameters()),
           "state_dimension":z.shape[1],"physical_gpu":cfg["physical_gpu"],"pca_device":"cuda:0"}
    arrays={"states":z.cpu().numpy(),"velocities":v.cpu().numpy(),"coordinates":xy.cpu().numpy(),
            "projected_velocities":pv.cpu().numpy(),"pca_mean":mean.cpu().numpy(),"pca_basis":basis.cpu().numpy()}
    np.savez_compressed(out/"states.npz",**arrays);atomic(out/"metadata.json",metadata);atomic(out/"statistics.json",stats)
    plot_state(out,arrays,metadata,stats,cfg["dataset"]);cases(model,payload,mean,basis,out,cfg);finish(cfg)


if __name__=="__main__":
    main()
