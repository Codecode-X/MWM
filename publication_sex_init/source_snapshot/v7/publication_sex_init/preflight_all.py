"""Small CUDA checks for every queued architecture, without optimizer updates."""
import argparse
import gc
import os
from pathlib import Path
import sys
import torch
from .queue import OUT,read,atomic
from .train import data_context,preflight
from .models import make_model
from paper_odeae.train import seed_all


def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=int,required=True);args=p.parse_args()
    assert torch.cuda.is_available() and os.environ["CUDA_VISIBLE_DEVICES"]==str(args.gpu)
    torch.cuda.set_device(0);torch.set_num_threads(1)
    samples={};results=[]
    jobs=read(OUT/"manifest.json")["jobs"]
    for job in jobs:
        cfg=read(job["config"])
        if cfg.get("family") not in {"publication","matched_baseline","deterministic_baseline","native_baseline"}:
            continue
        folder=OUT/"preflight_checks";folder.mkdir(exist_ok=True);target=folder/(job["id"]+".json")
        if target.exists():results.append(read(target));continue
        d=cfg["dataset"];cfg["physical_gpu"]=args.gpu
        if d not in samples:
            base=read(OUT/f"results/{d}/main/launch_config.json")
            payload,schema,splits,batches,evaluate=data_context(base)
            samples[d]=(schema,next(batches(1)))
            del payload,splits
        schema,batch=samples[d]
        seed_all(cfg["seed"]);model=make_model(schema,cfg)
        try:
            info=preflight(model,batch,schema,cfg)
            info.update(job=job["id"],purpose="bounded implementation preflight; not formal training or cost timing",
                        batch_size=4,maximum_visits=8,concurrent_with_training=True,optimizer_steps=0)
            atomic(target,info);results.append(info)
            print(job["id"]+" passed",flush=True)
        except BaseException as exc:
            atomic(folder/(job["id"]+".failure.json"),{"error":repr(exc),"physical_gpu":args.gpu})
            raise
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    atomic(OUT/"preflight_checks/summary.json",{"status":"passed","architectures":len(results),
                                             "physical_gpu":args.gpu,"optimizer_steps":0})


if __name__=="__main__":
    main()
