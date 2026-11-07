"""Parent orchestration: isolated GPU model execution, then independent CPU scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from .queue import atomic, read, ROOT


def source_snapshot(folder):
    source=folder/"model_source"
    if (source/"source_hashes.json").exists():
        return source
    template=ROOT/"CausalMed/outputs/gamma1_extensions_cuda/model_source"
    shutil.copytree(template,source,dirs_exist_ok=True)
    pkg=source/"publication_sex_init";pkg.mkdir(exist_ok=True)
    (pkg/"__init__.py").write_text("")
    for name in ["cf_entry.py","cf_audit_entry.py"]:
        shutil.copy2(ROOT/"publication_sex_init"/name,pkg/name)
    # The public trainer is copied independently and capped before it can step.
    train=source/"latent2x_experiment/train.py"
    body=train.read_text()
    marker="cfg=json.loads(Path(args.config).read_text());output=Path('/workspace/output');stage=args.stage"
    assert marker in body
    body=body.replace(marker,marker+"\n    assert cfg['max_epochs']<=150\n    cfg['max_epochs']=min(cfg['max_epochs'],150)")
    train.write_text(body)
    atomic(source/"source_hashes.json",{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in source.rglob("*") if p.is_file() and p.name!="source_hashes.json"})
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args()
    cfg=read(args.config);folder=Path(cfg["output"]);folder.mkdir(parents=True,exist_ok=True)
    source=source_snapshot(folder);model_out=folder/"model";model_out.mkdir(exist_ok=True)
    gpu=cfg["physical_gpu"]
    from CausalMed.latent2x_experiment.isolation import command
    if cfg.get("audit_source"):
        old=Path(cfg["audit_source"]);history=read(old/"history.json")
        selected=min([r for r in history if r["epoch"]<=150],key=lambda r:r["validation"]["overall"]["reconstruction_error"])
        assert selected["epoch"]==cfg["expected_epoch"]<=150
        # Loading, verifying and scoring weights remains inside the CUDA namespace.
        shutil.copy2(old/"best.pt",model_out/"best.pt")
        trained=read(old/"training_complete.json")
        assert trained["best_epoch"]==selected["epoch"]
        atomic(model_out/"training_complete.json",{**trained,"source_actual_epochs_run":trained["epochs_run"],
                                                 "budget_selection_max_epochs":150,"budget_audit_epoch":selected["epoch"]})
    atomic(model_out/"launch_config.json",cfg)
    for stage in (["training"] if cfg.get("audit_source") else ["checks","training"]):
        target=model_out if stage=="training" else folder/"checks"
        target.mkdir(exist_ok=True);atomic(target/"launch_config.json",cfg)
        cmd=command(source,ROOT/"CausalMed/outputs/level3_suite/gamma_1/public",target,ROOT/"swanlog",
                    stage,gpu,"ode")
        if not cfg.get("audit_source"):
            cmd[cmd.index("latent2x_experiment.train")]="publication_sex_init.cf_entry"
        else:
            cmd[cmd.index("latent2x_experiment.train")]="publication_sex_init.cf_audit_entry"
        subprocess.run(cmd,check=True)
    submission=read(model_out/"submission.json")
    assert submission["status"]=="predictions_sealed"
    for variant in ["with_actions","action_off"]:
        sealed=read(model_out/variant/"sealed.json")
        for name,sha in sealed["sha256"].items():
            assert hashlib.sha256((model_out/variant/name).read_bytes()).hexdigest()==sha
        target=folder/"evaluation"/variant;target.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-m","synthetic_cf_level3.evaluation","--predictions",str(model_out/variant/"predictions.parquet"),
             "--predicted-treatments",str(model_out/variant/"predicted_treatments.parquet"),
             "--oracle-dir",str(ROOT/"CausalMed/outputs/level3_suite/gamma_1/oracle_private"),
             "--out-dir",str(target)]
        subprocess.run(cmd,env={**os.environ,"PYTHONPATH":str(ROOT/"CausalMed"),"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"},check=True)
    atomic(folder/"metrics.json",{"status":"complete","best_epoch":submission["best_epoch"],
           "epochs_run":submission["epochs_run"],"submission":submission,
           "validation":read(model_out/"training_complete.json")["validation"],
           "with_actions":read(folder/"evaluation/with_actions/metrics.json"),
           "action_off":read(folder/"evaluation/action_off/metrics.json"),
           "physical_gpu":gpu,"model_execution":"isolated CUDA namespace"})


if __name__=="__main__":
    main()
