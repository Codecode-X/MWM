"""CUDA audit/evaluation of a genuinely eligible <=150-epoch saved baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from .train import atomic, data_context


def load_native(schema, cfg, saved):
    if cfg.get("capacity_matched"):
        from capacity_matched.models import make_model
        return make_model(schema, cfg).cuda()
    if cfg["model"] in ["deterministic", "ode_ae"]:
        from paper_odeae.models import paper_model_from_config
        return paper_model_from_config(schema, cfg).cuda()
    raise ValueError("Audit currently covers the native eligible ODE-AE checkpoint")


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args()
    task=json.loads(Path(args.config).read_text());out=Path(task["output"]);out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and os.environ["CUDA_VISIBLE_DEVICES"]==str(task["physical_gpu"])
    torch.cuda.set_device(0);torch.set_num_threads(1)
    source=Path(task["source_run"]);history=json.loads((source/"history.json").read_text())
    eligible=[r for r in history if r["epoch"]<=150]
    expected=min(eligible,key=lambda r:r["validation"]["overall"]["reconstruction_error"])
    saved=torch.load(source/"best.pt",map_location="cuda",weights_only=False)
    assert saved["epoch"]==expected["epoch"]<=150
    cfg=dict(saved["config"]);cfg.update(dataset=task["dataset"],physical_gpu=task["physical_gpu"])
    cfg.setdefault("dataset_sha256",hashlib.sha256(Path(cfg["data"]).read_bytes()).hexdigest())
    cfg["family"]="native_baseline"
    payload,schema,splits,batches,evaluate=data_context(cfg)
    model=load_native(schema,cfg,saved);model.load_state_dict(saved["state_dict"]);model.eval()
    def record(module,args):
        if (out/"hardware.json").exists():return
        batch=args[0]
        atomic(out/"hardware.json",{"physical_gpu":task["physical_gpu"],
               "parameter_device":str(next(model.parameters()).device),
               "actual_tensor_devices":{k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
               "parameters":sum(p.numel() for p in model.parameters()),
               "source_checkpoint":str(source/"best.pt")})
    hook=model.register_forward_pre_hook(record)
    validation=evaluate(model,"validation")
    difference=abs(validation["overall"]["reconstruction_error"]-expected["validation"]["overall"]["reconstruction_error"])
    assert difference<1e-6
    result={"status":"complete","best_epoch":saved["epoch"],"epochs_run":len(history),
            "budget_selection_epochs":len(eligible),
            "source_actual_epochs_run":len(history),"evaluation_budget":150,
            "validation":validation,"test":evaluate(model,"test"),
            "checkpoint_reload_difference":difference,"source_checkpoint":str(source/"best.pt"),
            "checkpoint_sha256":hashlib.sha256((source/"best.pt").read_bytes()).hexdigest(),
            "physical_gpu":task["physical_gpu"],"parameter_device":str(next(model.parameters()).device),
            "budget_audit":"chosen exactly by minimum validation error in first150 recorded epochs"}
    if cfg["dataset"]=="mimic":
        result["test_early_contexts"]={str(h):evaluate(model,"test",horizon=h) for h in [6,12]}
    atomic(out/"metrics.json",result)
    hook.remove()


if __name__=="__main__":
    main()
