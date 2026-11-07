"""Frozen-checkpoint GPU analyses with explicit publication-model adapters."""
import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from paper_odeae.data import Prefixes, collate as adni_collate
from paper_odeae.train import device_batch
from .models import make_model
from .queue import atomic, read, ROOT, OUT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@contextmanager
def argv(arguments):
    previous = sys.argv
    try:
        sys.argv = [previous[0], *map(str, arguments)]
        yield
    finally:
        sys.argv = previous


def load(cfg):
    saved = torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)
    model = make_model(saved["schema"], saved["config"])
    model.load_state_dict(saved["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, saved


@torch.no_grad()
def encode(model, batch, return_before=False):
    state, _, trace = model.encode(batch, return_trace=return_before)
    if not return_before:
        return state
    before = torch.zeros_like(state)
    for j, item in enumerate(trace):
        before = torch.where((batch["cutoff"]-1 == j)[:,None], item["before"], before)
    return state, before


def patch_factory(module):
    module.paper_model_from_config = lambda schema,cfg:make_model(schema,cfg)
    if hasattr(module, "encode"):
        module.encode = encode


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def finish(cfg, **kwargs):
    out=Path(cfg["output"])
    atomic(out/"complete.json",{"status":"complete","analysis":cfg["analysis"],
        "physical_gpu":cfg["physical_gpu"],"device":"cuda:0","checkpoint":cfg.get("checkpoint"),
        "checkpoint_sha256":sha(cfg["checkpoint"]) if cfg.get("checkpoint") else None,
        "training_performed":False,"parameter_updates":0,**kwargs})


def state_extract(cfg):
    model,saved=load(cfg);payload=pickle.loads(Path(saved["config"]["data"]).read_bytes())
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=True)
    schema=payload["schema"];positions={f["name"]:j for j,f in enumerate(schema)}
    meta=payload["metadata"];states=[];velocities=[];metadata=[]
    dataset=cfg["dataset"]
    for split in ["train","validation","test"]:
        seqs=[s for s in payload["sequences"] if s["split"]==split]
        if dataset=="adni":
            loaders=[(None,DataLoader(Prefixes(seqs),batch_size=64,shuffle=False,collate_fn=adni_collate))]
        else:
            from mimic_experiment.data import ICUContexts,collate
            hours=[6,12,18,24] if split=="train" else [24]
            loaders=[(h,DataLoader(ICUContexts(seqs,horizon=h),batch_size=64,collate_fn=collate)) for h in hours]
        for h,loader in loaders:
            for cpu in loader:
                batch=device_batch(cpu)
                with torch.no_grad():
                    z=encode(model,batch);v=model.forward_ode(z)
                states.append(z);velocities.append(v)
                for i,cut in enumerate(cpu["cutoff"].tolist()):
                    row={"subject_id":str(cpu["subject_ids"][i]),"split":split,"prefix":cut,
                         "age":float(cpu.get("initialization_age",cpu["ages"][:,0])[i]) if dataset=="mimic" else float(cpu["ages"][i,cut-1]),
                         "clock":float(cpu["ages"][i,cut-1]),
                         "context_hours":h,"coverage":float(cpu["context_mask"][i,:cut].mean())}
                    for name in ["diagnosis","mmse","cdrsb","heart_rate","map","gcs_eye","gcs_motor","gcs_verbal"]:
                        j=positions.get(name)
                        row[name]=float(cpu["values"][i,cut-1,j]*meta["scale"][j]+meta["offset"][j]) if j is not None and cpu["mask"][i,cut-1,j] else None
                    metadata.append(row)
            print(json.dumps({"extraction":split,"hours":h,"states_so_far":len(metadata)}),flush=True)
    z=torch.cat(states);v=torch.cat(velocities)
    train=torch.tensor([m["split"]=="train" for m in metadata],device="cuda")
    mean=z[train].mean(0)
    _,singular,basis_full=torch.linalg.svd((z[train]-mean).double(),full_matrices=False)
    basis=basis_full[:2].T.to(z.dtype);ratio=singular.square()/singular.square().sum()
    _,vs,vh=torch.linalg.svd(v[train].double(),full_matrices=False)
    vb=vh[:2].T.to(z.dtype);xy=(z-mean)@basis;pv=v@basis
    probs=ratio[ratio>0];rank=float(torch.exp(-(probs*probs.log()).sum()))
    statistics={"model_label":"M0 sex INIT","checkpoint":cfg["checkpoint"],"checkpoint_sha256":sha(cfg["checkpoint"]),
                "checkpoint_epoch":saved["epoch"],"dataset_sha256":sha(saved["config"]["data"]),
                "state_variance_retained_2d":float(ratio[:2].sum()),
                "velocity_energy_retained_2d":float(pv.square().sum()/v.square().sum().clamp_min(1e-30)),
                "velocity_plane_energy_retained_2d":float((v@vb).square().sum()/v.square().sum().clamp_min(1e-30)),
                "effective_rank":rank,"pca_variance_ratio":ratio.tolist(),"latent":z.shape[1],
                "physical_gpu":cfg["physical_gpu"],"state_device":str(z.device),"pca_fit":"train only",
                "counts_by_split":{s:sum(m["split"]==s for m in metadata) for s in ["train","validation","test"]}}
    arrays={"states":z.cpu().numpy(),"velocities":v.cpu().numpy(),"coordinates":xy.cpu().numpy(),
            "projected_velocities":pv.cpu().numpy(),"pca_mean":mean.cpu().numpy(),"pca_basis":basis.cpu().numpy(),
            "velocity_basis":vb.cpu().numpy()}
    np.savez_compressed(out/"states.npz",**arrays);atomic(out/"metadata.json",metadata);atomic(out/"statistics.json",statistics)
    if dataset=="mimic":
        geom=OUT/"analysis/mimic/manifold";geom.mkdir(parents=True,exist_ok=True)
        for role,allowed in [("train",{"train"}),("heldout",{"validation","test"})]:
            ids=np.array([i for i,r in enumerate(metadata) if r["split"] in allowed])
            np.savez_compressed(geom/f"mimic_states_{role}.npz",states=arrays["states"][ids],velocities=arrays["velocities"][ids])
            atomic(geom/f"mimic_metadata_{role}.json",[metadata[i] for i in ids])
    plot_state(out,arrays,metadata,statistics,dataset)
    if dataset=="adni":
        clinical_cases(model,payload,mean,basis,out)
    finish(cfg,states=len(z))


def plot_state(out,arrays,metadata,stats,dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test=np.array([r["split"]=="test" for r in metadata]);x=arrays["coordinates"][test];v=arrays["projected_velocities"][test]
    labels=np.array([r["diagnosis"] if r["diagnosis"] is not None else -1 for r in metadata])[test]
    if dataset=="mimic":
        labels=np.array([r["gcs_motor"] if r["gcs_motor"] is not None else -1 for r in metadata])[test]
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    im=axes[0].scatter(*x.T,c=labels,cmap="viridis",s=7,alpha=.5);fig.colorbar(im,ax=axes[0],label="Diagnosis" if dataset=="adni" else "GCS motor")
    pick=np.arange(0,len(x),max(1,len(x)//1200))
    q=axes[1].quiver(*x[pick].T,*v[pick].T,np.linalg.norm(v[pick],axis=1),cmap="plasma",angles="xy",scale_units="xy")
    fig.colorbar(q,ax=axes[1],label="Projected speed")
    from paper_odeae.diagnostics.state_space import local_field
    train=np.array([r["split"]=="train" for r in metadata])
    trainlabels=torch.tensor([-1 if r["diagnosis"] is None else int(r["diagnosis"]) for r in metadata],device="cuda")
    field=local_field(torch.tensor(arrays["coordinates"][train],device="cuda"),
                      torch.tensor(arrays["projected_velocities"][train],device="cuda"),trainlabels[torch.tensor(train,device="cuda")])
    mask=field["supported"];grid=field["grid"][mask].cpu().numpy();vv=field["velocity"][mask].cpu().numpy()
    axes[2].scatter(*x.T,c="#aaa",s=2,alpha=.2)
    if len(grid):
        axes[2].quiver(*grid.T,*vv.T,angles="xy",scale_units="xy")
    for ax,title in zip(axes,["Test states","Raw per-state velocities","Training-neighborhood average"]):
        ax.set(xlabel="Training state PC1",ylabel="Training state PC2",title=title)
    fig.tight_layout();fig.savefig(out/"state_and_velocity.png",dpi=180);plt.close(fig)
    np.savez_compressed(out/"projected_field.npz",**{k:x.cpu().numpy() if torch.is_tensor(x) else x for k,x in field.items()})


@torch.no_grad()
def clinical_cases(model,payload,mean,basis,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup={str(s["subject_id"]):s for s in payload["sequences"] if s["split"]=="test"}
    rows=[];fig,axes=plt.subplots(3,2,figsize=(12,13))
    for row,pid in enumerate(["003_S_4900","002_S_4219","002_S_0619"]):
        seq=lookup[pid];cut=min(3,len(seq["ages"])-1)
        batch=device_batch(adni_collate([(seq,cut)]));state,_,trace=model.encode(batch,return_trace=True)
        path=model(batch);points=torch.cat([x["after"] for x in trace]);xy=(points-mean)@basis
        latent=path["jacobian_states"][0];p=(latent-mean)@basis
        axes[row,0].plot(*p.cpu().numpy().T,"-",label="Free ODE")
        axes[row,0].plot(*xy.cpu().numpy().T,"o--",label="Observation updates")
        axes[row,0].set_title(pid);axes[row,0].legend()
        j=next(j for j,f in enumerate(model.schema) if f["name"]=="mmse")
        head=sum(f.get("classes",1) for f in model.schema[:j]);scale=payload["metadata"]["scale"][j];offset=payload["metadata"]["offset"][j]
        observed=seq["values"][:,j]*scale+offset;pred=path["prediction"][0,:,head].cpu().numpy()*scale+offset
        valid=seq["mask"][:,j].astype(bool)
        axes[row,1].plot(seq["ages"][valid],observed[valid],"o",label="Observed MMSE");axes[row,1].plot(seq["ages"],pred,"-",label="Predicted MMSE")
        axes[row,1].axvline(seq["ages"][cut-1],color="gray");axes[row,1].legend()
        rows.append({"subject_id":pid,"cutoff":cut,"projected_trajectory":p.cpu().tolist(),"observed_mmse":observed.tolist(),"predicted_mmse":pred.tolist()})
    fig.tight_layout();fig.savefig(out/"clinical_cases.png",dpi=180);plt.close(fig);atomic(out/"patient_paths.json",rows)


def adni_reuse(cfg):
    name=cfg["analysis"];out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=True)
    source=OUT/"analysis/adni/state"
    if name in ["clinical","local"]:
        module=importlib.import_module("paper_odeae.diagnostics."+("clinical_velocity" if name=="clinical" else "local_dynamics"))
        patch_factory(module)
        # Reports are regenerated from this run's actual metrics below.
        module.report=lambda *args,**kwargs:None
        with argv(["--gpu",cfg["physical_gpu"],"--source",source,"--output",out]):
            module.main()
        import pandas as pd
        if name=="clinical":
            from paper_odeae.diagnostics.clinical_velocity import numeric_metrics
            frame=pd.read_csv(out/"interval_rates.csv")
            frame=frame[frame.adjacent_visit]
            summary=read(out/"metrics.json")
            for (marker,split),group in frame.groupby(["marker","split"]):
                patient=group.groupby("subject_id")[["observed_rate","predicted_rate"]].mean()
                item=numeric_metrics(patient.observed_rate,patient.predicted_rate)
                item.update(patients=len(patient),intervals=len(group))
                summary["metrics"][marker][split]["patient_weighted"]=item
            atomic(out/"metrics.json",summary)
        else:
            frame=pd.read_csv(out/"state_local_metrics.csv")
            neighbor=np.load(out/"neighborhoods.npz")
            frame["overlap"]=neighbor["neighbor_patient_overlap"]
            summary=read(out/"metrics.json")
            for item in summary["regions"]:
                group=frame[(frame.split=="test")&(frame.region==item["region"])]
                if len(group):
                    patient=group.groupby("subject_id")[["C20","D20","J","overlap"]].mean()
                    item.update(patient_weighted_C=float(patient.C20.mean()),patient_weighted_D=float(patient.D20.mean()),
                                patient_weighted_J=float(patient.J.mean()),patient_weighted_overlap=float(patient.overlap.mean()))
            atomic(out/"metrics.json",summary)
    elif name=="stage":
        from paper_odeae.diagnostics import continuous_stage as original
        # Compile a recorded dimension-generalized copy of the existing task.
        body=Path(original.__file__).read_text()
        body=body.replace("cfg['latent']==6", "cfg['latent']==28")
        body=body.replace("'latent_dimensions':6", "'latent_dimensions':28")
        target=out/"continuous_stage_source.py";target.write_text(body)
        module=importlib.util.module_from_spec(importlib.util.spec_from_file_location("paper_odeae.diagnostics.publication_stage",target))
        sys.modules[module.__name__]=module;exec(compile(body,str(target),"exec"),module.__dict__)
        patch_factory(module)
        module.run(SimpleNamespace(gpu=str(cfg["physical_gpu"]),source=str(source),output=str(out)))
        from paper_odeae.diagnostics.continuous_stage_report import generate
        generate(out)
    elif name=="mmse":
        from paper_odeae.diagnostics import mmse_rate_mechanism as original
        body=Path(original.__file__).read_text().replace(".reshape(-1,6)",".reshape(-1,model.latent)")
        target=out/"mmse_mechanism_source.py";target.write_text(body)
        module=importlib.util.module_from_spec(importlib.util.spec_from_file_location("paper_odeae.diagnostics.publication_mmse",target))
        sys.modules[module.__name__]=module;exec(compile(body,str(target),"exec"),module.__dict__)
        patch_factory(module)
        workspace=out/"inputs";workspace.mkdir(exist_ok=True)
        for alias,path in [("shared_matched_state_space",source),("shared_matched_clinical_velocity",OUT/"analysis/adni/clinical")]:
            if not (workspace/alias).exists():
                (workspace/alias).symlink_to(path,target_is_directory=True)
        module.RESULTS=workspace
        with argv(["--variant","shared_matched","--gpu",cfg["physical_gpu"]]):
            module.main()
        produced=workspace/"shared_matched_mmse_mechanism"
        for f in produced.iterdir():
            if f.is_file():
                import shutil
                shutil.copy2(f,out/f.name)
    else:
        raise ValueError(name)
    finish(cfg)


def geometry(cfg):
    from paper_odeae.diagnostics import manifold_core as core
    core.paper_model_from_config=lambda schema,c:make_model(schema,c)
    core.encode=encode
    # The four-GPU queue owns the entire device; keep enough space for full 28D probes.
    def setup():
        assert torch.cuda.is_available()
        torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
        torch.cuda.set_per_process_memory_fraction(.85)
        torch.backends.cuda.matmul.allow_tf32=False
    core.setup=setup
    original_provenance=core.provenance
    core.provenance=lambda *args,**kwargs:{**original_provenance(*args,**kwargs),"memory_budget_fraction":.85}
    module=importlib.import_module("paper_odeae.diagnostics.manifold_"+cfg["analysis"])
    module.encode=encode;module.setup=setup
    if cfg["analysis"]=="geometry":
        # State extraction has already produced both MIMIC cache halves.
        pass
    else:
        module.load_model=core.load_model
    run_cfg={**cfg,"output":str(OUT/f'analysis/{cfg["dataset"]}/manifold'),
             "job_output":cfg["output"],"data":cfg["data"],"state_source":str(OUT/f'analysis/{cfg["dataset"]}/state')}
    path=Path(cfg["output"])/"geometry_config.json";atomic(path,run_cfg)
    with argv(["--config",path]):
        module.main()
    finish(cfg)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args()
    cfg=read(args.config);out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES")==str(cfg["physical_gpu"])
    torch.cuda.set_device(0);torch.set_num_threads(1)
    if cfg["analysis"]=="state":
        state_extract(cfg)
    elif cfg["dataset"]=="adni" and cfg["analysis"] in ["clinical","local","mmse","stage"]:
        adni_reuse(cfg)
    elif cfg["analysis"] in ["geometry","dynamics"]:
        geometry(cfg)
    else:
        from .analysis_extra import run
        run(cfg)


if __name__=="__main__":
    main()
