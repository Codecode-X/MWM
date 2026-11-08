"""CUDA verification of a budget-eligible deterministic counterfactual checkpoint."""
import json
from pathlib import Path
import torch
from latent2x_experiment.data import load_public,atomic
from latent2x_experiment.odeae_model import ControlledODEAE
from latent2x_experiment.train import evaluate


def main():
    import latent2x_experiment.train as native
    folder=Path("/workspace/output");cfg=json.loads((folder/"launch_config.json").read_text())
    assert torch.cuda.is_available()
    torch.cuda.set_device(0);torch.set_num_threads(1)
    saved=torch.load(folder/"best.pt",map_location="cuda",weights_only=False)
    assert saved["epoch"]==cfg["expected_epoch"]<=150
    payload,_=load_public("/workspace/public");model=ControlledODEAE(cfg).cuda()
    model.load_state_dict(saved["state_dict"]);model.eval()
    validation=evaluate(model,payload["val"],payload["metadata"],cfg["batch_size"])
    trained=json.loads((folder/"training_complete.json").read_text())
    discrepancy=abs(validation["overall"]["reconstruction_error"]-trained["validation"]["overall"]["reconstruction_error"])
    assert discrepancy<1e-6
    atomic(folder/"budget_audit.json",{"status":"passed","checkpoint_epoch":saved["epoch"],
          "device":"cuda:0","validation_reload_difference":discrepancy,
          "source_actual_epochs":trained["source_actual_epochs_run"],"selected_within_budget":150})
    del model,saved;torch.cuda.empty_cache()
    native.main()


if __name__=="__main__":
    main()
