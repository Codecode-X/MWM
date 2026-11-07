"""CPU-only live tables, plots and checkpoint archiving for the publication run."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from .queue import ROOT,OUT,read,atomic

DOC=ROOT/"reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.md"


def fmt(x):
    return "" if x is None else f"{x:.6f}"


def table_section(text,number,mutator):
    pattern=rf"(表 {number}：[^\n]*\n\n)(\|[^\n]*\n(?:\|[^\n]*\n)*)"
    found=re.search(pattern,text)
    if not found:return text
    lines=found.group(2).strip().splitlines()
    rows=[[v.strip() for v in line.split("|")[1:-1]] for line in lines]
    mutator(rows)
    body="\n".join("| "+" | ".join(row)+" |" for row in rows)+"\n"
    return text[:found.start(2)]+body+text[found.end(2):]


def archive(job):
    folder=Path(job["output"]);history=read(folder/"history.json",[])
    if not history or not (folder/"best.pt").exists():return
    best=min(history,key=lambda r:r["validation"]["overall"]["reconstruction_error"])
    dest=folder/"diagnostic_checkpoints"/f'best_epoch{best["epoch"]:03d}.pt'
    if dest.exists():return
    dest.parent.mkdir(exist_ok=True)
    try:
        os.link(folder/"best.pt",dest)
    except FileExistsError:
        return
    atomic(dest.with_suffix(".json"),{"history_best_epoch":best["epoch"],
        "source":str(folder/"best.pt"),"archived_at_utc":datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
        "weights_and_Adam_copied_without_model_execution":True})


def curves(jobs,dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected={}
    for job in jobs:
        cfg=read(job["config"])
        if cfg.get("dataset")==dataset and cfg.get("family")=="publication":
            history=read(Path(job["output"])/"history.json",[])
            if history:selected[cfg["variant"]]=(job,history)
    if not selected:return []
    groups={"core":["main","no_jacobian","reverse","separate_fields","no_encoder_ode","attention","add_prior"],
            "init":["main","init_first_age","init_cutoff_age","init_zero_scalar"],
            "alpha":["main","alpha_0","alpha_02","alpha_03","alpha_04","alpha_05","alpha_1"],
            "capacity":["main","latent_6","latent_73","init_expand","update_expand","ode_expand"]}
    folder=OUT/"curves";folder.mkdir(exist_ok=True);paths=[]
    for group,names in groups.items():
        active=[n for n in names if n in selected]
        if not active:continue
        stamp=max(Path(selected[n][0]["output"],"history.json").stat().st_mtime for n in active)
        target=folder/f"{dataset}_{group}.png"
        if target.exists() and target.stat().st_mtime>=stamp:
            paths.append(target);continue
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for name in active:
            job,h=selected[name];x=[r["epoch"] for r in h]
            sequences=[
                [r["training_loss"] for r in h],
                [r["training_components"].get("history") for r in h],
                [r["training_components"].get("future") for r in h],
                [r["validation"]["interpolation"]["reconstruction_error"] for r in h],
                [r["validation"]["forecast"]["reconstruction_error"] for r in h],
                [r["validation"]["overall"]["reconstruction_error"] for r in h]]
            for ax,values in zip(axes.flat,sequences):
                ax.plot(x,values,label=name,lw=1.3)
        for ax,title in zip(axes.flat,["Training objective","Training history","Training future",
                                     "Validation history","Validation future","Validation overall"]):
            ax.set(xlabel="Epoch",ylabel="Loss",title=title);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle(dataset.upper()+" / "+group+" / sex INIT")
        fig.tight_layout();fig.savefig(target,dpi=160);plt.close(fig);paths.append(target)
    return paths


def generate():
    manifest=read(OUT/"manifest.json");queue=read(OUT/"queue.json");jobs=manifest.get("jobs",[])
    byid={j["id"]:j for j in jobs}
    results={j["id"]:read(Path(j["output"])/"metrics.json") for j in jobs}
    def result(ident):
        r=results.get(ident,{})
        return r if r.get("status")=="complete" else {}
    text=DOC.read_text().split("\n## 七、实时执行进度")[0].rstrip()+"\n"
    def score(ident):
        r=result(ident)
        if not r:return None
        return [fmt(r["test"][k]["reconstruction_error"]) for k in ["interpolation","forecast","overall"]]
    for dataset,number in [("adni",1),("mimic",3)]:
        mapping={"M0（性别 INIT）":dataset+"_main",
                 "确定性 ODE-AE（同预算）":dataset+"_deterministic_matched"}
        mapping.update({"Latent ODE（同预算）":"adni_latent_ode_matched_150"} if dataset=="adni" else
                       {"ODE-RNN（同预算）":"mimic_ode_rnn_matched_150"})
        def main_rows(rows,mapping=mapping):
            for row in rows[2:]:
                ident=mapping.get(row[0])
                if not ident:continue
                r=result(ident)
                if r:
                    row[3]=f'{r["epochs_run"]} / {r["best_epoch"]}';row[4:7]=score(ident);row[7]="complete"
                else:
                    row[7]=queue.get("jobs",{}).get(ident,{}).get("status","queued")
        text=table_section(text,number,main_rows)
    def native(rows):
        mapping={"ODE-AE（概率，原容量）":"mimic_ode_ae_native_budget_audit",
                 "ODE-RNN（原容量）":"mimic_ode_rnn_native_150"}
        for row in rows[2:]:
            ident=mapping.get(row[0]);r=result(ident) if ident else {}
            if r:
                row[3]=f'{r["epochs_run"]} / {r["best_epoch"]}';row[4:7]=score(ident);row[7]="complete"
    text=table_section(text,4,native)
    for number,variant in [(10,"reverse"),(11,"separate_fields"),(12,"no_encoder_ode"),
                           (13,"attention"),(14,"no_jacobian"),(15,"add_prior")]:
        def core_rows(rows,variant=variant):
            for row in rows[2:]:
                dataset=row[0].lower();ident=dataset+("_main" if row[1]=="M0" else "_"+variant)
                r=result(ident)
                if r:row[3]=str(r["best_epoch"]);row[4:7]=score(ident)
        text=table_section(text,number,core_rows)
    def init(rows):
        mapping={"性别（M0）":"main","首次检查年龄":"init_first_age",
                 "截止点年龄":"init_cutoff_age","固定零输入":"init_zero_scalar"}
        for row in rows[2:]:
            s=score(row[0].lower()+"_"+mapping[row[1]])
            if s:row[4:7]=s
    text=table_section(text,27,init)
    def alpha(rows):
        for row in rows[2:]:
            raw=row[0].split("（")[0];variant="main" if raw=="0.1" else "alpha_"+raw.replace(".","")
            for dataset,at in [("adni",2),("mimic",6)]:
                s=score(dataset+"_"+variant)
                if s:row[at:at+3]=s
    text=table_section(text,28,alpha)
    def capacity(rows):
        mapping={"较小状态":"latent_6","M0":"main","较大状态":"latent_73",
                 "仅 INIT 扩容":"init_expand","仅 UPDATE 扩容":"update_expand","仅 ODE 扩容":"ode_expand"}
        for row in rows[2:]:
            ident=row[0].lower()+"_"+mapping[row[1]]
            if ident in byid:
                config=read(byid[ident]["config"]);row[3]=f'{config["target_parameters"]:,}'
            s=score(ident)
            if s:
                row[4:6]=s[1:3]
                base=result(row[0].lower()+"_main")
                if base and ident in byid:
                    p=read(byid[ident]["config"])["target_parameters"];b=44875 if row[0]=="ADNI" else 43280
                    if p!=b:row[6]=fmt((base["test"]["forecast"]["reconstruction_error"]-float(s[1]))/((p-b)/10000))
    text=table_section(text,29,capacity)
    def state_rows(rows):
        mapping={"M0":"main","确定性 ODE-AE（同预算）":"deterministic","ODE-AE（概率，同预算）":"ode_ae",
                 "ODE-RNN（同预算）":"ode_rnn","Latent ODE（同预算）":"latent_ode","HealthFlux-KL":"healthflux"}
        for row in rows[2:]:
            d=row[0].lower();name=mapping[row[1]]
            path=OUT/f"analysis/{d}/state" if name=="main" else OUT/f"analysis/{d}/baseline_state/{name}"
            stats=read(path/"statistics.json")
            if (path/"complete.json").exists():
                row[2:6]=[fmt(stats[k]) for k in ["state_variance_retained_2d","velocity_energy_retained_2d",
                                                 "velocity_plane_energy_retained_2d","effective_rank"]]
    text=table_section(text,16,state_rows)
    for job in jobs:
        archive(job)
    images=[]
    for d in ["adni","mimic"]:
        images+=curves(jobs,d)
    status=queue.get("jobs",{})
    lines=["","## 七、实时执行进度","",
           f'更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}。控制器状态：{queue.get("status","starting")}。',
           "","旧 GPU 作业已中断，本轮独立目录保留配置、CUDA 预检、完整恢复状态和 SwanLab。下表展示实际进度；测试指标在完整评测后写入前文。","",
           "| 任务 | GPU | 状态 | 当前 / 完成轮次 | 最佳验证整体 | 记录 |",
           "|---|---|---|---|---|---|"]
    for job in jobs:
        q=status.get(job["id"],{});folder=Path(job["output"]);progress=read(folder/"progress.json")
        h=read(folder/"history.json",[]);epoch=progress.get("epoch",progress.get("completed_epoch",""))
        if h and not epoch:epoch=h[-1]["epoch"]
        best=min((r["validation"]["overall"]["reconstruction_error"] for r in h),default=None)
        rel=os.path.relpath(folder,DOC.parent)
        lines.append(f'| {job["id"]} | {q.get("physical_gpu","")} | {q.get("status","queued")} | {epoch} | {fmt(best)} | [目录]({rel}) |')
    for image in images:
        lines+=["",f'![{image.stem} 训练与验证曲线]({os.path.relpath(image,DOC.parent)})']
    lines+=["","[队列状态](../publication_sex_init/queue.json) · [完整实验配置](../publication_sex_init/manifest.json) · [旧作业中断记录](../publication_sex_init/preemption_record.json)",""]
    DOC.write_text(text+"\n".join(lines))
    atomic(OUT/"progress_summary.json",{"updated_at_utc":datetime.now(timezone.utc).isoformat(),
           "jobs":len(jobs),"status":queue.get("status"),"counts":{s:sum(q.get("status")==s for q in status.values())
               for s in ["queued","running","complete","failed"]}})
    planpath=ROOT/"reproduction/no_weak_prior_rerun_plan.json";plan=read(planpath)
    plan.update(status="executing",execution_started=True,execution_manifest=str(OUT/"manifest.json"),
                execution_queue=str(OUT/"queue.json"))
    atomic(planpath,plan)


def main():
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");args=p.parse_args()
    while True:
        try:generate()
        except Exception as exc:
            atomic(OUT/"report_error.json",{"error":repr(exc),"time":datetime.now(timezone.utc).isoformat()})
            if not args.watch:raise
        if not args.watch:return
        time.sleep(30)


if __name__=="__main__":
    main()
