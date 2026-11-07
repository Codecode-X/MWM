"""Register dependency-aware frozen GPU diagnostics after the training jobs."""
import hashlib
from pathlib import Path
import shutil
from .queue import ROOT,OUT,read,atomic,now


def main():
    source=OUT/"source_snapshot/v3"
    if not (source/"source_hashes.json").exists():
        shutil.copytree(OUT/"source_snapshot/v2",source,dirs_exist_ok=True)
        for package in ["publication_sex_init","healthflux"]:
            for path in (ROOT/package).glob("*.py"):
                target=source/package/path.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
        atomic(source/"source_hashes.json",{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in source.rglob("*") if p.is_file() and p.name!="source_hashes.json"})
    manifest=read(OUT/"manifest.json");jobs=manifest["jobs"];ids={j["id"] for j in jobs}
    def add(ident,cfg,module="publication_sex_init.analysis",depends=(),priority=8,environment="ode"):
        if ident in ids:return
        path=Path(cfg["output"])/"launch_config.json";atomic(path,{**cfg,"physical_gpu":None})
        job={"id":ident,"kind":"frozen_analysis","priority":priority,"config":str(path),"output":cfg["output"],
             "module":module,"source":str(source),"depends_on":list(depends),"completion_file":"complete.json"}
        if environment!="ode":
            job["command"]=[str(Path("/home/xjh/miniforge3/envs")/environment/"bin/python"),"-u","-m",module,"--config",str(path)]
        jobs.append(job);ids.add(ident)
    byid={j["id"]:j for j in jobs}
    for dataset in ["adni","mimic"]:
        base=read(OUT/f"results/{dataset}/main/launch_config.json")
        checkpoint=str(Path(base["output"])/"best.pt")
        for analysis in ["state","horizons","clinical","local","geometry","dynamics","cost"]:
            deps=[dataset+"_main"]
            if analysis in ["clinical","local","geometry","dynamics"]:deps.append(dataset+"_state")
            cfg={"dataset":dataset,"analysis":analysis,"checkpoint":checkpoint,"data":base["data"],
                 "output":str(OUT/f"analysis/{dataset}/{analysis}" if analysis not in ["geometry","dynamics"] else OUT/f"analysis/{dataset}/manifold/{dataset}_{analysis}")}
            add(dataset+"_"+analysis,cfg,depends=deps)
        cfg={"dataset":dataset,"analysis":"jacobian_utility","checkpoint":checkpoint,
             "models":{"main":checkpoint,"no_jacobian":str(OUT/f"results/{dataset}/no_jacobian/best.pt")},
             "output":str(OUT/f"analysis/{dataset}/jacobian_utility")}
        add(dataset+"_jacobian_utility",cfg,depends=[dataset+"_main",dataset+"_no_jacobian"])
        baselines=[
            ("ode_ae",ROOT/f'{"paper_odeae" if dataset=="adni" else "mimic_experiment"}/results/baselines_matched_latent2x/ode_ae/seed_0/best.pt',[], "ode"),
            ("deterministic",Path(byid[dataset+"_deterministic_matched"]["output"])/"best.pt",[dataset+"_deterministic_matched"],"ode"),
            ("latent_ode",Path(byid["adni_latent_ode_matched_150"]["output"])/"best.pt" if dataset=="adni" else ROOT/"mimic_experiment/results/baselines_matched_latent2x/latent_ode/seed_0/best.pt",
             ["adni_latent_ode_matched_150"] if dataset=="adni" else [],"ode"),
            ("ode_rnn",ROOT/"paper_odeae/results/baselines_matched_latent2x/ode_rnn/seed_0/best.pt" if dataset=="adni" else Path(byid["mimic_ode_rnn_matched_150"]["output"])/"best.pt",
             [] if dataset=="adni" else ["mimic_ode_rnn_matched_150"],"ode"),
            ("healthflux",ROOT/"healthflux/results/adni_comparison/common_task/seed_0/best.pt" if dataset=="adni" else ROOT/"mimic_experiment/results/baselines_single_cuda/healthflux_kl/seed_0/best.pt",[],"healthflux"),
        ]
        for key,path,deps,environment in baselines:
            for analysis in ["state","cost"]:
                folder=OUT/f"analysis/{dataset}/baseline_{analysis}/{key}"
                add(dataset+"_baseline_"+analysis+"_"+key,
                    {"dataset":dataset,"analysis":analysis,"model_key":key,"checkpoint":str(path),"output":str(folder)},
                    module="publication_sex_init.baseline_analysis",depends=deps,
                    priority=9,environment=environment)
    for analysis in ["stage","mmse"]:
        add("adni_"+analysis,{"dataset":"adni","analysis":analysis,
            "checkpoint":str(OUT/"results/adni/main/best.pt"),"output":str(OUT/f"analysis/adni/{analysis}")},
            depends=["adni_state"]+ (["adni_clinical"] if analysis=="mmse" else []))
    add("mimic_stability",{"dataset":"mimic","analysis":"stability",
        "checkpoint":str(OUT/"results/mimic/main/best.pt"),
        "models":{"main":str(OUT/"results/mimic/main/best.pt"),"alpha1":str(OUT/"results/mimic/alpha_1/best.pt")},
        "output":str(OUT/"analysis/mimic/stability")},depends=["mimic_main","mimic_alpha_1"],priority=18)
    manifest["updated_at_utc"]=now();atomic(OUT/"manifest.json",manifest)
    print({"jobs":len(jobs),"source":str(source)})


if __name__=="__main__":
    main()
