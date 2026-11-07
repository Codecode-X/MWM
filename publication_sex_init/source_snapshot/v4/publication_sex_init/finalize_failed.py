"""Evaluate the selected valid checkpoint after a recorded numerical training failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from .models import make_model
from .train import data_context,atomic


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg["output"])
    assert torch.cuda.is_available() and os.environ["CUDA_VISIBLE_DEVICES"]==str(cfg["physical_gpu"])
    torch.cuda.set_device(0);torch.set_num_threads(1)
    saved=torch.load(out/"best.pt",map_location="cuda",weights_only=False)
    assert saved["epoch"]<=150
    payload,schema,splits,batches,evaluate=data_context(cfg)
    model=make_model(schema,cfg);model.load_state_dict(saved["state_dict"]);model.eval()
    history=json.loads((out/"history.json").read_text())
    validation=evaluate(model,"validation")
    expected=next(r for r in history if r["epoch"]==saved["epoch"])["validation"]["overall"]["reconstruction_error"]
    difference=abs(validation["overall"]["reconstruction_error"]-expected);assert difference<1e-6
    result={"status":"complete","termination_reason":"numerical_training_instability",
            "nonfinite_events":1,"best_epoch":saved["epoch"],"epochs_run":len(history),
            "max_total_epochs":150,"validation":validation,"test":evaluate(model,"test"),
            "checkpoint_reload_difference":difference,"checkpoint_sha256":hashlib.sha256((out/"best.pt").read_bytes()).hexdigest(),
            "physical_gpu":cfg["physical_gpu"],"parameter_device":"cuda:0",
            "failure_record":str(out/"numerical_failure.json"),"parameter_updates_during_finalization":0}
    if cfg["dataset"]=="mimic":
        result["test_early_contexts"]={str(h):evaluate(model,"test",horizon=h) for h in [6,12]}
    atomic(out/"metrics.json",result)


if __name__=="__main__":
    main()
