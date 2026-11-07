"""Inside the public-only namespace: observation gating without gating actions."""
import json
from pathlib import Path
import torch
from latent2x_experiment.model import ControlledBoth
from paper_odeae.jacobian import squared_jacobian_norm


class PublicationControlledBoth(ControlledBoth):
    def event(self,state,values,mask,action,observation):
        proposed=super().event(state,values,mask,action,observation)
        if observation:
            alpha=self.cfg["update_alpha"]
            return (1-alpha)*state+alpha*proposed
        return proposed


def objective(model,out,batch):
    errors=(out["prediction"]-batch["values"]).square()*batch["mask"]
    mse=(errors.sum((0,1))/batch["mask"].sum((0,1)).clamp_min(1)).sum()
    zero=mse.new_zeros(())
    jac=squared_jacobian_norm(model.forward_ode,out["states"][batch["valid"]].detach()).mean()
    l2=model.ode_regularization()
    loss=mse+model.cfg["jacobian_weight"]*jac+model.cfg["l2_weight"]*l2
    assert model.cfg["prior_weight"]==0
    return loss,{"mse":mse,"prior_mse":zero,"weighted_prior":zero,
                "jacobian":jac,"weighted_jacobian":model.cfg["jacobian_weight"]*jac,
                "ode_l2":l2,"weighted_ode_l2":model.cfg["l2_weight"]*l2}


def main():
    import latent2x_experiment.train as native
    cfg=json.loads(Path("/workspace/output/launch_config.json").read_text())
    assert cfg["max_epochs"]<=150 and cfg["update_alpha"]==.1 and cfg["prior_weight"]==0
    native.ControlledBoth=PublicationControlledBoth
    native.objective=objective
    native.main()


if __name__=="__main__":
    main()
