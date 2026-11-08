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


def cf_figures(results,jobs):
    if not all(results.get(k,{}).get("status")=="complete" for k in ["causalmed_main","causalmed_deterministic_budget_audit"]):
        return
    out=OUT/"analysis/causalmed/comparison";out.mkdir(parents=True,exist_ok=True)
    if (out/"case_metadata.json").exists():return
    byid={j["id"]:j for j in jobs}
    specs=[
        ("M0-c",Path(byid["causalmed_main"]["output"])/"model"),
        ("ODE-AE",ROOT/"CausalMed/outputs/ode_ae_gamma1_cuda1/gamma_1/model"),
        ("Deterministic ODE-AE (matched)",Path(byid["causalmed_deterministic_budget_audit"]["output"])/"model"),
        ("HealthFlux-KL",ROOT/"CausalMed/outputs/healthflux_kl_gamma1_cuda3/gamma_1/model"),
    ]
    rows=[]
    for i,(name,run) in enumerate(specs):
        view=out/"inputs"/str(i);(view/"gamma_1").mkdir(parents=True,exist_ok=True)
        link=view/"gamma_1/model"
        if not link.exists():link.symlink_to(run,target_is_directory=True)
        rows.append({"name":name,"folder":str(view)})
    from CausalMed.latent2x_experiment.compare import case_plots
    case_plots(out,rows)


def generate():
    manifest=read(OUT/"manifest.json");queue=read(OUT/"queue.json");jobs=manifest.get("jobs",[])
    byid={j["id"]:j for j in jobs}
    results={j["id"]:read(Path(j["output"])/"metrics.json") for j in jobs}
    cf_figures(results,jobs)
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
                    row[3]=f'{r["epochs_run"]} / {r["best_epoch"]}';row[4:7]=score(ident)
                    row[7]="不稳定停止；评测最佳模型" if r.get("termination_reason") else "complete"
                else:
                    row[7]=queue.get("jobs",{}).get(ident,{}).get("status","queued")
        text=table_section(text,number,main_rows)
    def native(rows):
        mapping={"ODE-AE（概率，原容量）":"mimic_ode_ae_native_budget_audit",
                 "ODE-RNN（原容量）":"mimic_ode_rnn_native_150"}
        for row in rows[2:]:
            ident=mapping.get(row[0]);r=result(ident) if ident else {}
            if r:
                row[3]=f'{r["epochs_run"]}（前150选模） / {r["best_epoch"]}' if r.get("budget_selection_epochs") else f'{r["epochs_run"]} / {r["best_epoch"]}'
                row[4:7]=score(ident);row[7]="预算内复评完成" if r.get("budget_selection_epochs") else "complete"
    text=table_section(text,4,native)
    def clinical_main(rows):
        mapping={"M0":"mimic_main","确定性 ODE-AE（同预算）":"mimic_deterministic_matched",
                 "ODE-RNN（同预算）":"mimic_ode_rnn_matched_150"}
        for row in rows[2:]:
            r=result(mapping.get(row[0],""))
            if r:
                item=r["test"];c=item["clinical_forecast"]
                row[1:7]=[fmt(item["mean_standardized_numeric_mse"]),fmt(c["heart_rate"]["mae"]),fmt(c["map"]["mae"]),
                          *[fmt(c["gcs_"+f]["accuracy"]) for f in ["eye","motor","verbal"]]]
    text=table_section(text,5,clinical_main)
    def horizons_main(rows):
        mapping={"M0":"mimic_main","确定性 ODE-AE（同预算）":"mimic_deterministic_matched",
                 "ODE-RNN（同预算）":"mimic_ode_rnn_matched_150"}
        for row in rows[2:]:
            r=result(mapping.get(row[0],""))
            if r:row[1:4]=[fmt(r["test"][f"forecast_{h}h"]["reconstruction_error"]) for h in [6,12,24]]
    text=table_section(text,6,horizons_main)
    def cf_main(rows):
        for row in rows[2:]:
            ident="causalmed_main" if row[0].startswith("M0") else "causalmed_deterministic_budget_audit" if row[0].startswith("确定性") else None
            r=result(ident) if ident else {}
            if r:
                a=r["with_actions"];h=a["horizons"]["3.0"];e=h["effects_vs_never"]["1"]
                row[2:7]=[fmt(v) for v in [a["cf_rmse_trajectory"],h["cf_rmse"],e["sqrt_pehe"],e["ate_error"],h["policy_regret"]]]
                row[7]="complete"
    text=table_section(text,8,cf_main)
    def cf_actions(rows):
        r=result("causalmed_main")
        if not r:return
        for row,key in zip(rows[2:],["with_actions","action_off"]):
            a=r[key];h=a["horizons"]["3.0"];e=h["effects_vs_never"]["1"]
            row[1:5]=[fmt(v) for v in [a["cf_rmse_trajectory"],e["sqrt_pehe"],e["ate_error"],h["policy_regret"]]]
    text=table_section(text,9,cf_actions)
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
    def horizon_table(rows):
        for row in rows[2:]:
            dataset=row[0].lower();metrics=read(OUT/f"analysis/{dataset}/horizons/metrics.json",[])
            if dataset=="adni":
                windows={"0–12 个月":(0,1),"12–24 个月":(1,2),"24–36 个月":(2,3),"36–60 个月":(3,5)}
                key=windows[row[1]];item=next((r for r in metrics if (r["lo"],r["hi"])==key),None)
                if item:row[2:6]=[str(item["patients"]),str(item["observed_targets"]),fmt(item["future_error"]),fmt(item["mean_standardized_numeric_mse"])]
            else:
                r=result("mimic_main")
                if not r:continue
                hours=row[1].split(" ")[0]
                item=r["test"] if hours=="24" else r["test_early_contexts"][hours]
                row[2:6]=[str(item["patients"]),str(item["forecast"]["observed_scalars"]),
                          fmt(item["forecast"]["reconstruction_error"]),fmt(item["mean_standardized_numeric_mse"])]
    text=table_section(text,7,horizon_table)
    def clinical(rows):
        for row in rows[2:]:
            d=row[0].lower();m=read(OUT/f"analysis/{d}/clinical/metrics.json")
            key="mmse" if "MMSE" in row[1] else "cdrsb" if "CDR" in row[1] else "heart_rate" if "心率" in row[1] else "map"
            if d=="adni":
                item=m.get("metrics",{}).get(key,{}).get("test",{}).get("patient_weighted",{})
                n=item.get("intervals")
            else:
                item=m.get(key,{});n=item.get("n")
            if item:
                row[2:8]=[str(n),str(item["patients"]),fmt(item["mae"]),fmt(item["pearson"]),
                          fmt(item["spearman"]),fmt(item["up_down_accuracy_nonzero"])]
    text=table_section(text,17,clinical)
    def mechanism(rows):
        m=read(OUT/"analysis/adni/mmse/metrics.json")
        groups={g["diagnosis"]:g for g in m.get("groups",[])}
        for row in rows[2:]:
            item=groups.get(row[0])
            if item:
                row[1:8]=[str(item["patients"]),*[fmt(item[k]) for k in
                    ["speed_mean","gradient_norm_mean","cosine_mean","observed_rate_mean","predicted_rate_mean"]],
                    fmt(item["predicted_rate_mean"]-item["observed_rate_mean"])]
            elif row[0]=="快速下降区间" and m:
                import pandas as pd
                f=pd.read_csv(OUT/"analysis/adni/mmse/interval_decomposition.csv")
                f=f[(f.diagnosis==2)&(f.observed_rate<=m["training_dementia_fast_decline_threshold"])]
                if len(f):
                    row[1:8]=[str(f.subject_id.nunique()),*[fmt(float(f[k].mean())) for k in
                        ["speed","gradient_norm","cosine","observed_rate","predicted_rate"]],
                        fmt(float((f.predicted_rate-f.observed_rate).mean()))]
    text=table_section(text,18,mechanism)
    def local(rows):
        m=read(OUT/"analysis/adni/local/metrics.json")
        for row,item in zip(rows[2:],m.get("regions",[])[:5]):
            row[1:6]=[str(item["patients"]),fmt(item.get("patient_weighted_C")),fmt(item.get("patient_weighted_D")),
                      fmt(item.get("patient_weighted_J")),fmt(item.get("patient_weighted_overlap"))]
    text=table_section(text,19,local)
    def local_error(rows):
        for row in rows[2:]:
            d=row[0].lower();m=read(OUT/f"analysis/{d}/local/metrics.json")
            key=row[1].split(" ")[0]
            if d=="adni":
                item=m.get("forecast",{}).get("horizons",{}).get(key,{})
                if item:
                    row[2:6]=[str(item["test_patients"]),fmt(item["sensitivity_error"]["spearman"]),
                              fmt(item["adjusted_sensitivity_error"]),str(item["patient_bootstrap_95ci"]["adjusted_D"])]
            else:
                item=m.get("horizons",{}).get(key,{})
                if item:row[2:6]=[str(item["patients"]),fmt(item["D_error_spearman"]["spearman"]),fmt(item["partial_rank"]),
                                 str(item["partial_rank_bootstrap95"])]
    text=table_section(text,20,local_error)
    def geometry(rows):
        for row in rows[2:]:
            d=row[0].lower();folder=OUT/f"analysis/{d}/manifold"
            geo=read(folder/f"{d}_geometry/summary.json");dyn=read(folder/f"{d}_dynamics/summary.json")
            item=geo.get("primary",{})
            if item:
                raw=item["scales"]["raw"]
                row[1:5]=[str(item["pca_d95"]),fmt(raw["twonn"]),fmt(raw["mle_20"]["median"]),fmt(raw["local_pca"]["80"]["d95"]["median"])]
            if dyn:
                hs=sorted(dyn["tangent_summary"],key=float)
                row[5]=fmt(dyn["tangent_summary"][hs[-1]]["12"]["normal_velocity"]["median"])
    text=table_section(text,21,geometry)
    def projection(rows):
        for row in rows[2:]:
            d=row[0].lower();dyn=read(OUT/f"analysis/{d}/manifold/{d}_dynamics/summary.json")
            if not dyn:continue
            if row[1]=="原始自由积分":variant="baseline"
            elif row[1]=="向均值收缩对照":variant="mean_shrink_025"
            else:variant=dyn["validation_selected"]["local"]["variant"]
            error=dyn["probes"][variant]["test"]["error"]
            row[2:4]=[fmt(error),fmt(error-dyn["probes"]["baseline"]["test"]["error"])]
            schema=read(OUT/f"results/{d}/main/schema.json",[])
            meta=read(OUT/f"results/{d}/main/data.json")
            marker="mmse" if d=="adni" else "heart_rate"
            item=dyn["probes"][variant]["test"]["per_field"].get(marker)
            if item:
                j=next(i for i,f in enumerate(schema) if f["name"]==marker)
                row[4]=fmt(item["loss"]**.5*meta["scale"][j])
    text=table_section(text,22,projection)
    def jacobian(rows):
        for row in rows[2:]:
            d=row[0].lower();folder=OUT/f"analysis/{d}/jacobian_utility";m=read(folder/"metrics.json")
            label="main" if row[1]=="M0" else "no_jacobian";item=m.get(label,{})
            if item:
                row[2:6]=[fmt(item["local_jacobian_squared"]["mean"]),fmt(item["longest_flow_gain"]["median"]),
                          fmt(item["noise"]["0.05"]["output_shift_rmse"]),fmt(item["noise"]["0.0"]["clinical_rmse"])]
                path=folder/f"clinical_horizons_{label}.csv"
                if path.exists():
                    import csv
                    records=[r for r in csv.DictReader(path.open()) if float(r["lo"])>=(3 if d=="adni" else 12)]
                    count=sum(int(r["count"]) for r in records)
                    if count:row[6]=fmt((sum(float(r["square_error_sum"]) for r in records)/count)**.5)
    text=table_section(text,23,jacobian)
    def stage(rows):
        m=read(OUT/"analysis/adni/stage/metrics.json")
        mapping={"当前 s₀":"stage_0","瞬时速度 ṡ₀":"instantaneous_speed","阶段增量 Δs":"delta_stage",
                 "未来 Dementia 概率":"future_dementia_probability"}
        for row in rows[2:]:
            item=m.get("conversion_prediction",{}).get(row[0].split(" ")[0],{})
            if not item:continue
            score=item["scores"][mapping[row[2]]]
            row[1]=f'{item["stable_mci"]} / {item["converters"]}'
            row[3:5]=[fmt(score["auroc"]),fmt(score["auprc_average_precision"])]
            base=item["scores"]["stage_0"]["auroc"]
            if score["auroc"] is not None and base is not None:row[5]=fmt(score["auroc"]-base)
    text=table_section(text,24,stage)
    def stability(rows):
        import numpy as np
        measurements=read(OUT/"analysis/mimic/stability/metrics.json",[])
        for row,family,ident in zip(rows[2:],["main","alpha1"],["mimic_main","mimic_alpha_1"]):
            chosen=[r for r in measurements if r["model"]==family or r["model"].startswith(family+"_")]
            r=result(ident)
            if not chosen or not r:continue
            ratio=[r["history_gradient_norm"]/max(r["future_gradient_norm"],1e-30) for r in chosen]
            recursive=[x for r in chosen for x in r["recursive_state_gains"]]
            row[1:7]=[str(r["best_epoch"]),str(r["epochs_run"]),fmt(float(np.quantile(ratio,.95))),
                      fmt(float(np.median([r["gradient_cosine"] for r in chosen]))),
                      fmt(float(np.quantile(recursive,.95))),str(int(r.get("nonfinite_events",0)))]
    text=table_section(text,25,stability)
    def costs(rows):
        mapping={"M0":"cost","确定性 ODE-AE（同预算）":"baseline_cost/deterministic",
                 "ODE-AE（概率，同预算）":"baseline_cost/ode_ae","ODE-RNN（同预算）":"baseline_cost/ode_rnn",
                 "Latent ODE（同预算）":"baseline_cost/latent_ode","HealthFlux-KL":"baseline_cost/healthflux"}
        for row in rows[2:]:
            d=row[0].lower();m=read(OUT/f"analysis/{d}/{mapping[row[1]]}/metrics.json")
            if m:
                row[2:7]=[str(m["encoding_nfe"]),str(m["generation_nfe"]),f'{m["accepted"]} / {m["rejected"]}',
                          fmt(m["milliseconds_per_patient"]),fmt(m["peak_memory_gib"])]
                ident=d+("_main" if row[1]=="M0" else "_deterministic_matched" if "确定性" in row[1] else
                         "_ode_rnn_matched_150" if "ODE-RNN" in row[1] else "_latent_ode_matched_150")
                r=result(ident)
                if r:row[7]=fmt(r["test"]["forecast"]["reconstruction_error"])
    text=table_section(text,26,costs)
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
    for image in sorted((OUT/"analysis").glob("**/*.png")):
        if "inputs" in image.parts or "workspace" in image.parts:continue
        lines+=["",f'![{image.parent.name} / {image.stem}]({os.path.relpath(image,DOC.parent)})']
    lines+=["","[队列状态](../publication_sex_init/queue.json) · [完整实验配置](../publication_sex_init/manifest.json) · [旧作业中断记录](../publication_sex_init/preemption_record.json)",""]
    temporary=DOC.with_suffix(".tmp.md");temporary.write_text(text+"\n".join(lines));temporary.replace(DOC)
    atomic(OUT/"progress_summary.json",{"updated_at_utc":datetime.now(timezone.utc).isoformat(),
           "jobs":len(jobs),"status":queue.get("status"),"counts":{s:sum(q.get("status")==s for q in status.values())
               for s in ["queued","running","complete","failed"]}})
    planpath=ROOT/"reproduction/no_weak_prior_rerun_plan.json";plan=read(planpath)
    plan.update(status="executing",execution_started=True,execution_manifest=str(OUT/"manifest.json"),
                execution_queue=str(OUT/"queue.json"))
    highlevel={"SETUP":[], "M-ADNI":["adni_main"],"M-MIMIC":["mimic_main"],"M-CF":["causalmed_main"],
        "B-ADNI":["adni_deterministic_matched","adni_latent_ode_matched_150"],
        "B-MIMIC":["mimic_deterministic_matched","mimic_ode_rnn_matched_150","mimic_ode_rnn_native_150","mimic_ode_ae_native_budget_audit"],
        "B-CF":["causalmed_deterministic_budget_audit"],
        "D-HORIZON":["adni_horizons","mimic_horizons"],"D-STATE":["adni_state","mimic_state"]+
          [d+"_baseline_state_"+k for d in ["adni","mimic"] for k in ["deterministic","ode_ae","ode_rnn","latent_ode","healthflux"]],
        "D-CLINICAL":["adni_clinical","mimic_clinical"],"D-MMSE":["adni_mmse"],
        "D-LOCAL":["adni_local","mimic_local"],"D-GEOM":[d+"_"+a for d in ["adni","mimic"] for a in ["geometry","dynamics"]],
        "D-JAC":["adni_jacobian_utility","mimic_jacobian_utility"],"D-STAGE":["adni_stage"],
        "D-STABILITY":["mimic_stability"],"D-COST":["adni_cost","mimic_cost"]+
          [d+"_baseline_cost_"+k for d in ["adni","mimic"] for k in ["deterministic","ode_ae","ode_rnn","latent_ode","healthflux"]],
        "D-CF":["causalmed_main","causalmed_deterministic_budget_audit"]}
    variants={"A-REV":"reverse","A-SPLIT":"separate_fields","A-ENCODE":"no_encoder_ode",
              "A-ATTN":"attention","A-J0":"no_jacobian","A-PRIOR":"add_prior"}
    for d in ["adni","mimic"]:
        for prefix,v in variants.items():highlevel[prefix+"-"+d.upper()]=[d+"_"+v]
        highlevel["X-INIT-"+d.upper()]=[d+"_init_"+x for x in ["first_age","cutoff_age","zero_scalar"]]
        highlevel["X-ALPHA-"+d.upper()]=[d+"_alpha_"+x for x in ["0","02","03","04","05","1"]]
        highlevel["X-CAPACITY-"+d.upper()]=[d+"_"+x for x in ["latent_6","latent_73","init_expand","update_expand","ode_expand"]]
    for t in plan["tasks"]:
        names=highlevel.get(t["id"],[]);t["execution_jobs"]=names
        if t["id"]=="SETUP":
            t["status"]="complete" if read(OUT/"preflight_checks/summary.json").get("status")=="passed" else "running"
        elif names:
            statuses=[status.get(n,{}).get("status","queued") for n in names]
            t["status"]="complete" if all(s=="complete" for s in statuses) else "failed" if "failed" in statuses else "running" if "running" in statuses else "queued"
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
