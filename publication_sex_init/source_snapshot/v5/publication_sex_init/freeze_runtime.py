"""Freeze final queue implementation and switch only not-yet-started jobs."""
import hashlib
from pathlib import Path
import shutil
from .queue import ROOT,OUT,read,atomic,now


def main():
    source=OUT/"source_snapshot/v5"
    if not (source/"source_hashes.json").exists():
        shutil.copytree(OUT/"source_snapshot/v3",source,dirs_exist_ok=True)
        for path in (ROOT/"publication_sex_init").glob("*.py"):
            shutil.copy2(path,source/"publication_sex_init"/path.name)
        atomic(source/"source_hashes.json",{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in source.rglob("*") if p.is_file() and p.name!="source_hashes.json"})
    manifest=read(OUT/"manifest.json");queue=read(OUT/"queue.json");changed=[]
    for job in manifest["jobs"]:
        if queue.get("jobs",{}).get(job["id"],{}).get("status") in {"running","complete"}:
            continue
        if job["module"]!="publication_sex_init.cf_launch":
            job["source"]=str(source);cfg=read(job["config"]);cfg["source_snapshot"]=str(source)
            atomic(job["config"],cfg);changed.append(job["id"])
    manifest["updated_at_utc"]=now();manifest["queued_runtime_snapshot"]=str(source)
    atomic(OUT/"manifest.json",manifest)
    atomic(OUT/"runtime_revision.json",{"time_utc":now(),"source":str(source),
           "queued_jobs_updated":changed,"running_training_jobs_unchanged":True})
    print({"queued_jobs_updated":len(changed),"source":str(source)})


if __name__=="__main__":
    main()
