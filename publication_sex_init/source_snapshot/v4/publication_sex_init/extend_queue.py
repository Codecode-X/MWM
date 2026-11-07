"""Append all remaining baseline and capacity jobs without disturbing running jobs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from .queue import ROOT, OUT, read, atomic, now


def snapshot():
    source=OUT/"source_snapshot/v2"
    if not (source/"source_hashes.json").exists():
        shutil.copytree(OUT/"source_snapshot/v1",source,dirs_exist_ok=True)
        for path in (ROOT/"publication_sex_init").glob("*.py"):
            shutil.copy2(path,source/"publication_sex_init"/path.name)
        atomic(source/"source_hashes.json",{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in source.rglob("*") if p.is_file() and p.name!="source_hashes.json"})
    return source


def main():
    source=snapshot();manifest=read(OUT/"manifest.json");jobs=manifest["jobs"];ids={j["id"] for j in jobs}
    def add(ident,cfg,module,priority,source_override=None,depends=()):
        if ident in ids:
            return
        folder=OUT/"results"/cfg["dataset"]/ident;cfg.update(output=str(folder),physical_gpu=None)
        path=folder/"launch_config.json";atomic(path,cfg)
        jobs.append({"id":ident,"kind":module.rsplit(".",1)[-1],"priority":priority,"config":str(path),
                     "output":str(folder),"source":str(source_override or source),
                     "module":module,"depends_on":list(depends)})
        ids.add(ident)
    for dataset in ["adni","mimic"]:
        base=read(OUT/f"results/{dataset}/main/launch_config.json")
        # Native deterministic ODE-AE: same six-dimensional architecture, exact count.
        target=44875 if dataset=="adni" else 43280;features=49 if dataset=="adni" else 44
        candidates=[]
        for w in range(1,1000):
            numerator=target-(2*features+28)*w-(features+24)
            if numerator>0 and numerator%26==0:
                v=numerator//26;candidates.append((abs(v/w-118/154),w,v))
        _,w,v=min(candidates)
        cfg={**base,"family":"deterministic_baseline","variant":"deterministic_matched","model":"deterministic",
             "latent":6,"mlp_width":w,"ode_width":v,"init_width":w,"update_width":w,"decoder_width":w,
             "encoder_ode_width":v,"generator_ode_width":v,"encoder_direction":"backward",
             "encoder_initialization":"age","prior_weight":0.,"jacobian_weight":0.,"kl_weight":0.,
             "shared_vector_field":False,"collect_jacobian_states":False,"collect_prior_predictions":False,
             "target_parameters":target,"init_covariate":"native_age"}
        if dataset=="mimic":
            old=ROOT/"mimic_experiment/results/baselines_with_deterministic_cuda/deterministic_matched/seed_0/config.json"
            if old.exists():
                native=read(old);cfg.update({k:native[k] for k in ["latent","mlp_width","ode_width","init_width","update_width","decoder_width",
                               "encoder_ode_width","generator_ode_width"] if k in native})
        add(dataset+"_deterministic_matched",cfg,"publication_sex_init.train",2)
        for kind in (["latent_ode"] if dataset=="adni" else ["ode_rnn"]):
            from capacity_matched.models import definition
            old=ROOT/f'{"paper_odeae" if dataset=="adni" else "mimic_experiment"}/results/baselines_matched_latent2x/{kind}/seed_0/config.json'
            cfg={**read(old),**definition(dataset,kind),**{k:base[k] for k in ["data","dataset_sha256","max_epochs","min_epochs","patience","batch_size"]},
                 "dataset":dataset,"dataset_name":dataset,"family":"matched_baseline","variant":kind+"_matched",
                 "model":kind,"capacity_matched":True,"gradient_accumulation_steps":1,
                 "prior_weight":0.,"jacobian_weight":0.,"collect_prior_predictions":False,"collect_jacobian_states":False,
                 "init_covariate":"native","target_parameters":target,"source_snapshot":str(source)}
            add(dataset+"_"+kind+"_matched_150",cfg,"publication_sex_init.train",2)
        if dataset=="mimic":
            cfg={**read(ROOT/"mimic_experiment/results/baselines_single_cuda/ode_rnn/seed_0/config.json"),
                 "dataset":dataset,"family":"native_baseline","variant":"ode_rnn_native","max_epochs":150,
                 "min_epochs":50,"batch_size":128,"gradient_accumulation_steps":1,"model":"ode_rnn",
                 "prior_weight":0.,"jacobian_weight":0.,"target_parameters":31813,
                 "init_covariate":"native","source_snapshot":str(source)}
            add("mimic_ode_rnn_native_150",cfg,"publication_sex_init.train",10)
            add("mimic_ode_ae_native_budget_audit",{"dataset":"mimic",
                "source_run":str(ROOT/"mimic_experiment/results/baselines_single_cuda/ode_ae/seed_0")},
                "publication_sex_init.baseline_evaluate",2)
        for name,change in [
            ("latent_6",{"latent":6}),("latent_73",{"latent":73}),
            ("init_expand",{"init_width":4*base["mlp_width"]}),
            ("update_expand",{"update_width":4*base["mlp_width"]}),
            ("ode_expand",{"ode_width":4*base["ode_width"]}),
        ]:
            cfg={**base,**change,"variant":name}
            d=cfg["latent"];f=features
            def mlp(a,b,w):return (a+1)*w+(w+1)*b
            cfg["target_parameters"]=mlp(1,d,cfg.get("init_width",cfg["mlp_width"]))+mlp(d+f,d,cfg.get("update_width",cfg["mlp_width"]))+mlp(d,f,cfg["mlp_width"])+mlp(d,d,cfg["ode_width"])
            add(dataset+"_"+name,cfg,"publication_sex_init.train",22)
    cfg=read(ROOT/"CausalMed/outputs/latent2x_both_cuda3/manifest.json")["configs"]["gamma_1"]
    cfg.update(dataset="causalmed",max_epochs=150,prior_weight=0.,jacobian_weight=.01,
               update_alpha=.1,init_covariate="public_train_normalized_c",gradient_accumulation_steps=1)
    add("causalmed_main",cfg,"publication_sex_init.cf_launch",2,ROOT)
    old=ROOT/"CausalMed/outputs/deterministic_matched_gamma1_cuda1/gamma_1/model"
    cfg=read(old/"config.json");cfg.update(dataset="causalmed",max_epochs=150,
        audit_source=str(old),expected_epoch=145,gradient_accumulation_steps=1)
    add("causalmed_deterministic_budget_audit",cfg,"publication_sex_init.cf_launch",2,ROOT)
    manifest["updated_at_utc"]=now();atomic(OUT/"manifest.json",manifest)
    print(json.dumps({"jobs":len(jobs),"source":str(source)}))


if __name__=="__main__":
    main()
