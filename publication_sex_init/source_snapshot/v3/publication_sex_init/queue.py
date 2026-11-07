"""Persistent four-GPU scheduler for all publication checklist jobs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
OUT = ROOT/"publication_sex_init"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)); tmp.replace(path)


def freeze():
    dest = OUT/"source_snapshot/v1"
    if (dest/"source_hashes.json").exists():
        return dest
    for pkg in ["publication_sex_init", "paper_odeae", "mimic_experiment", "capacity_matched", "src"]:
        for path in (ROOT/pkg).rglob("*.py"):
            rel = path.relative_to(ROOT/pkg)
            if any(p in {"results","source_snapshot","__pycache__","cache"} for p in rel.parts):
                continue
            target = dest/pkg/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copytree(ROOT/"paper_odeae/reference", dest/"paper_odeae/reference", dirs_exist_ok=True)
    atomic(dest/"source_hashes.json", {str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in dest.rglob("*") if p.is_file()})
    return dest


def prepare():
    OUT.mkdir(exist_ok=True)
    source = freeze()
    jobs = []
    for dataset in ["mimic", "adni"]:
        base = read(ROOT/f"ablation/results/alpha01_components/{dataset}_no_prior/launch_config.json")
        # Resolve missing-sex fill using one first record per training patient.
        with Path(base["data"]).open("rb") as stream:
            payload = pickle.load(stream)
        si = next(i for i, f in enumerate(payload["schema"]) if f["name"]=="sex")
        labels = [int(s["values"][0,si]) for s in payload["sequences"]
                  if s["split"]=="train" and s["mask"][0,si]]
        if dataset=="adni":
            labels = [1-x for x in labels]
        fill = int(sum(labels)*2 >= len(labels))
        base.update(dataset=dataset, dataset_name=dataset, family="publication", variant="main",
                    init_covariate="sex", sex_missing_fill=fill, prior_weight=0., jacobian_weight=.01,
                    kl_weight=0., update_alpha=.1, latent=28, max_epochs=150, min_epochs=50,
                    gradient_accumulation_steps=1, history_order="chronological", physical_gpu=None,
                    shared_vector_field=True, collect_prior_predictions=False, collect_jacobian_states=True,
                    study="publication_sex_init", source_snapshot=str(source))
        base["batch_size"] = 128 if dataset=="mimic" else 32
        variants = [
            ("main", {}, 0),
            ("no_jacobian", {"jacobian_weight":0}, 1),
            ("reverse", {"history_order":"reverse"}, 3),
            ("separate_fields", {}, 4),
            ("no_encoder_ode", {}, 5),
            ("attention", {}, 6),
            ("add_prior", {"prior_weight":.01, "collect_prior_predictions":True}, 7),
        ]
        variants += [(f"init_{name}", {"init_covariate":name}, 20) for name in ["first_age","cutoff_age","zero_scalar"]]
        variants += [(f"alpha_{str(a).replace('.','')}", {"update_alpha":a}, 21)
                     for a in [0, .2, .3, .4, .5, 1]]
        for name, change, priority in variants:
            cfg = {**base, "variant":name, **change}
            if name not in {"separate_fields","no_encoder_ode","attention"}:
                cfg["variant"] = name
            folder = OUT/"results"/dataset/name
            cfg["output"] = str(folder)
            cfgpath = folder/"launch_config.json"
            if not cfgpath.exists():
                atomic(cfgpath, cfg)
            jobs.append({"id":dataset+"_"+name, "kind":"clinical_training", "priority":priority,
                         "config":str(cfgpath), "output":str(folder), "source":str(source),
                         "module":"publication_sex_init.train", "depends_on":[]})
    manifest = {"created_at_utc":now(),"status":"ready","source":str(source),
                "physical_gpus":[0,1,2,3],"max_total_epochs":150,"jobs":jobs}
    # Additional baseline/analysis jobs can be appended while this controller runs.
    existing = read(OUT/"manifest.json")
    ids = {j["id"] for j in jobs}
    manifest["jobs"] += [j for j in existing.get("jobs",[]) if j["id"] not in ids]
    atomic(OUT/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    assert Path(sys.prefix).name=="ode"
    OUT.mkdir(exist_ok=True)
    controller_lock = (OUT/"controller.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest = prepare()
    if args.prepare_only:
        return
    state = read(OUT/"queue.json", {"jobs":{}})
    state.update(status="running", controller_pid=os.getpid(), started_at_utc=now(),
                 physical_gpus=[0,1,2,3])
    running = {}; stopping = {"value":False}
    def stop(signum, frame):
        stopping["value"] = True
        for jobid, worker in running.items():
            try:
                os.kill(worker["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        state["status"] = "stopping"
        atomic(OUT/"queue.json", state)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    for job in manifest["jobs"]:
        q = state["jobs"].setdefault(job["id"], {"status":"queued"})
        if Path(job["output"],job.get("completion_file","metrics.json")).exists():
            q["status"] = "complete"
        elif q.get("pid") and Path(f'/proc/{q["pid"]}').exists():
            cmd = Path(f'/proc/{q["pid"]}/cmdline').read_bytes().decode(errors="replace")
            if job["config"] in cmd.replace("\0"," "):
                gpu = q["physical_gpu"]; lock = Path(f"/tmp/latentode_gpu_{gpu}.lock").open("a")
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                running[job["id"]]={"pid":q["pid"],"gpu":gpu,"lock":lock,"proc":None}
            else:
                q["status"]="queued"
        elif q["status"] in ["running","interrupted"]:
            q["status"]="queued"
    while True:
        manifest = read(OUT/"manifest.json")
        jobs = {j["id"]:j for j in manifest["jobs"]}
        for ident in jobs:
            state["jobs"].setdefault(ident,{"status":"queued"})
        for ident, worker in list(running.items()):
            proc = worker["proc"]
            alive = proc.poll() is None if proc else Path(f'/proc/{worker["pid"]}').exists()
            if alive:
                continue
            job = jobs[ident]; path = Path(job["output"])/job.get("completion_file","metrics.json")
            q = state["jobs"][ident]
            q.update(status="complete" if path.exists() else "interrupted" if stopping["value"] else "failed",
                     finished_at_utc=now(),exit_code=proc.returncode if proc else None)
            worker["lock"].close(); running.pop(ident)
        if stopping["value"] and not running:
            state["status"]="stopped"; atomic(OUT/"queue.json",state); return
        if not stopping["value"]:
            pending = sorted([j for j in jobs.values() if state["jobs"][j["id"]]["status"]=="queued"
                              and all(state["jobs"].get(d,{}).get("status")=="complete" for d in j.get("depends_on",[]))],
                             key=lambda j:j["priority"])
            for gpu in [0,1,2,3]:
                if not pending or any(w["gpu"]==gpu for w in running.values()):
                    continue
                lock=Path(f"/tmp/latentode_gpu_{gpu}.lock").open("a")
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close(); continue
                apps=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
                if apps:
                    lock.close(); continue
                job=pending.pop(0); cfg=read(job["config"]); cfg["physical_gpu"]=gpu
                atomic(job["config"],cfg); folder=Path(job["output"]); folder.mkdir(parents=True,exist_ok=True)
                env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONPATH":job["source"],
                     "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","SWANLAB_LOG_DIR":str(ROOT/"swanlog"),
                     "PYTHONUNBUFFERED":"1"}
                with (folder/"training.log").open("a") as stream:
                    stream.write("\nPublication dispatch "+now()+" physical GPU "+str(gpu)+"\n");stream.flush()
                    cmd=job.get("command") or [sys.executable,"-u","-m",job["module"],"--config",job["config"]]
                    proc=subprocess.Popen(cmd,cwd=job["source"],env=env,stdout=stream,stderr=subprocess.STDOUT)
                running[job["id"]]={"pid":proc.pid,"proc":proc,"gpu":gpu,"lock":lock}
                state["jobs"][job["id"]].update(status="running",pid=proc.pid,physical_gpu=gpu,started_at_utc=now())
                print(json.dumps({"launch":job["id"],"gpu":gpu,"pid":proc.pid}),flush=True)
        state["updated_at_utc"]=now();atomic(OUT/"queue.json",state)
        if not running and all(q["status"] in ["complete","failed"] for q in state["jobs"].values()):
            state["status"]="complete" if all(q["status"]=="complete" for q in state["jobs"].values()) else "completed_with_failures"
            atomic(OUT/"queue.json",state); return
        time.sleep(10)


if __name__=="__main__":
    main()
