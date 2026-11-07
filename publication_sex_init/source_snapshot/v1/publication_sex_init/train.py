"""Single-GPU full-batch training, exact restart state, max 150 total epochs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import sys
import time
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from paper_odeae.train import seed_all, device_batch
from paper_odeae.data import Prefixes, collate as adni_collate
from paper_odeae.models import feature_errors
from src.tracking import SwanLabTracker
from .models import make_model, objective, fields, jacobian_regularization, parameter_hash


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(path)


def save(path, value):
    path = Path(path); tmp = path.with_suffix(".partial.pt")
    torch.save(value, tmp); tmp.replace(path)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(rng):
    torch.set_rng_state(rng["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in rng["cuda"]])
    np.random.set_state(rng["numpy"]); random.setstate(rng["python"])


def data_context(cfg):
    raw = Path(cfg["data"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == cfg["dataset_sha256"]
    payload = pickle.loads(raw); schema = payload["schema"]
    splits = {k: [s for s in payload["sequences"] if s["split"] == k] for k in ["train", "validation", "test"]}
    if cfg["dataset"] == "mimic":
        from mimic_experiment.data import ICUContexts, ExactBatchSampler, collate
        from mimic_experiment.train_baseline import normalization_and_modes
        from mimic_experiment.evaluation import evaluate
        means, modes = normalization_and_modes(payload)
        cfg.update(raw_scale=payload["metadata"]["scale"], normalized_train_means=means)
        dataset = ICUContexts(splits["train"], training=True, seed=cfg["seed"])
        sampler = ExactBatchSampler(len(dataset), cfg["batch_size"], 0, 1, cfg["seed"])
        def batches(epoch):
            dataset.set_epoch(epoch); sampler.set_epoch(epoch)
            for indices in sampler:
                b = device_batch(collate([dataset[i] for i in indices], cfg["input_dropout"]))
                b["patient_ids"] = [str(dataset[i][0]["subject_id"]) for i in indices]
                yield b
        def score(model, split, horizon=24):
            return evaluate(model, splits[split], schema, cfg, payload["metadata"]["raw_train_std"], modes, horizon=horizon)
    else:
        from paper_odeae.train import evaluate
        def batches(epoch):
            loader = DataLoader(Prefixes(splits["train"]), batch_size=cfg["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(cfg["seed"]+epoch),
                                collate_fn=partial(adni_collate, dropout=cfg["input_dropout"]))
            for b in loader:
                yield device_batch(b)
        def score(model, split, horizon=None):
            return evaluate(model, splits[split], schema, 32)
    if cfg.get("family", "publication") == "publication":
        import paper_odeae.train as scoring
        import paper_odeae.jacobian as jac
        scoring.jacobian_regularization = jacobian_regularization
        jac.jacobian_regularization = jacobian_regularization
    return payload, schema, splits, batches, score


def preflight(model, batch, schema, cfg):
    # Keep the checks bounded while retaining both historical and future targets.
    b = {k: v[:4] if torch.is_tensor(v) else v for k, v in batch.items()}
    n = min(8, b["values"].shape[1])
    for key in ["values", "ages", "mask", "context_mask", "valid"]:
        b[key] = b[key][:, :n].clone()
    b["cutoff"] = b["cutoff"].clamp(max=max(1,n-1))
    idx = torch.arange(n, device="cuda")[None]
    b["context_mask"] *= (idx < b["cutoff"][:,None])[...,None]
    b["max_context"] = int(b["cutoff"].max())
    before = parameter_hash(model); model.eval()
    with torch.no_grad():
        out = model(b, sample=False)
        altered = {**b, "values": b["values"].clone()}
        future = idx >= b["cutoff"][:,None]
        for j, field in enumerate(schema):
            changed = (b["values"][...,j]+1)%field["classes"] if field["kind"]=="categorical" else b["values"][...,j]+3
            altered["values"][...,j] = torch.where(future, changed, b["values"][...,j])
        other = model(altered, sample=False)
        torch.testing.assert_close(out["prediction"], other["prediction"], atol=0, rtol=0)
    model.train(); out = model(b, sample=True)
    loss, _ = objective(model, out, b, schema, cfg); loss.backward()
    missing, nonfinite = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            # Alpha=0 intentionally removes the UPDATE from all task paths.
            if not (cfg.get("update_alpha") == 0 and name.startswith("update.")):
                missing.append(name)
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            nonfinite.append(name)
    assert not missing, missing
    assert not nonfinite, nonfinite
    model.zero_grad(set_to_none=True)
    assert parameter_hash(model) == before
    count = sum(p.numel() for p in model.parameters())
    assert count == cfg["target_parameters"], (count, cfg["target_parameters"])
    info = {"status": "passed", "parameters": count, "physical_gpu": cfg["physical_gpu"],
            "parameter_device": str(next(model.parameters()).device),
            "prediction_device": str(out["prediction"].device), "future_values_isolated": True,
            "finite_parameter_gradients": True, "optimizer_steps": 0,
            "initial_parameter_hash": before}
    if hasattr(model, "initialization_scalar"):
        x = model.initialization_scalar(b)
        if cfg["init_covariate"] == "sex":
            assert bool(((x == 0) | (x == 1)).all())
        info.update(actual_INIT_input=x.detach().cpu().tolist(),
                    init_covariate=cfg["init_covariate"], actual_INIT_device=str(x.device),
                    INIT_parameters=sum(p.numel() for p in model.init.parameters()),
                    shared_field_alias=model.backward_ode is model.forward_ode,
                    unique_ode_parameters=sum(p.numel() for f in fields(model) for p in f.parameters()))
    return info


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    p.add_argument("--checks-only", action="store_true")
    args = p.parse_args(); cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
    assert Path(sys.prefix).name == "ode"
    assert torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") == str(cfg["physical_gpu"])
    assert cfg["max_epochs"] <= 150 and cfg["gradient_accumulation_steps"] == 1
    torch.set_num_threads(1); torch.cuda.set_device(0); seed_all(cfg["seed"])
    payload, schema, splits, batches, evaluate = data_context(cfg)
    model = make_model(schema, cfg)
    hardware = {"physical_gpu": cfg["physical_gpu"], "gpu_name": torch.cuda.get_device_name(),
                "parameter_device": str(next(model.parameters()).device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "time_unit": "year" if cfg["dataset"] == "adni" else "hour",
                "batch_size": cfg["batch_size"], "gradient_accumulation_steps": 1,
                "max_total_epochs": 150, "dataset_sha256": cfg["dataset_sha256"],
                "init_covariate": cfg.get("init_covariate"), "solver": "Tsit5",
                "rtol": cfg["rtol"], "atol": cfg["atol"],
                "source_snapshot": cfg.get("source_snapshot"), "torch_version": torch.__version__}
    atomic(out/"config.json", cfg); atomic(out/"hardware.json", hardware)
    atomic(out/"schema.json", schema); atomic(out/"data.json", payload["metadata"])
    if not (out/"checks.json").exists():
        state = rng_state()
        atomic(out/"checks.json", preflight(model, next(batches(1)), schema, cfg))
        restore_rng(state)
    if args.checks_only:
        return
    if (out/"metrics.json").exists():
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: cfg["lr_decay"]**(k/cfg["lr_decay_steps"]))
    history, best, bad, steps, start_epoch = [], float("inf"), 0, 0, 1
    resume = None; sums = {}; count = 0; maxnorm = 0.
    if (out/"last.pt").exists():
        resume = torch.load(out/"last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(resume["state_dict"]); optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history, best, bad, steps = resume["history"], resume["best_score"], resume["bad_epochs"], resume["optimizer_steps"]
        start_epoch = resume["epoch"] + (0 if resume.get("in_epoch") else 1)
        assert start_epoch <= 151
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    def checkpoint(epoch, in_epoch=False, next_batch=0):
        return {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng_state": rng_state(), "config": cfg,
                "schema": schema, "metadata": payload["metadata"], "epoch": epoch,
                "in_epoch": in_epoch, "next_batch": next_batch, "history": history,
                "best_score": best, "bad_epochs": bad, "optimizer_steps": steps,
                "epoch_sums": sums, "epoch_batches": count, "maximum_batch_gradient_norm": maxnorm}
    if not (out/"initial.pt").exists():
        save(out/"initial.pt", checkpoint(0))
    with SwanLabTracker(out, cfg, "publication_"+cfg["dataset"]+"_"+cfg["variant"], model, cfg["seed"]) as tracker:
        tracker.step = steps
        for epoch in range(start_epoch, cfg["max_epochs"]+1):
            if epoch > cfg["min_epochs"] and bad >= cfg["patience"]:
                break
            seed_all(cfg["seed"]+epoch); model.train(); start = time.monotonic()
            sums, count, maxnorm = {}, 0, 0.
            skip = 0
            if resume is not None and resume.get("in_epoch") and resume["epoch"] == epoch:
                skip = resume["next_batch"]; sums = resume["epoch_sums"]
                count = resume["epoch_batches"]; maxnorm = resume["maximum_batch_gradient_norm"]
            torch.cuda.reset_peak_memory_stats()
            for batch_index, batch in enumerate(batches(epoch)):
                if batch_index < skip:
                    continue
                if resume is not None and resume.get("in_epoch") and batch_index == skip:
                    restore_rng(resume["rng_state"]); resume = None
                    # Regenerate this batch's dropout from the exact boundary RNG.
                    # Sampling/order use independent deterministic generators.
                    idx = torch.arange(batch["values"].shape[1], device="cuda")[None]
                    context = batch["mask"] * (idx < batch["cutoff"][:,None])[...,None]
                    if cfg["input_dropout"]:
                        keep = torch.from_numpy(np.random.random(tuple(context.shape)) >= cfg["input_dropout"]).cuda()
                        context = context * keep
                    batch["context_mask"] = context
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch, sample=cfg.get("family", "publication") != "publication")
                loss, components = objective(model, prediction, batch, schema, cfg)
                if not bool(torch.isfinite(loss)):
                    save(out/"failure_preupdate.pt", checkpoint(epoch, True, batch_index))
                    raise FloatingPointError("Nonfinite objective")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                postnorm = min(float(norm), 10.)
                optimizer.step(); scheduler.step(); steps += 1; count += 1
                maxnorm = max(maxnorm, float(norm))
                for key, value in {**components, "gradient_norm": norm, "gradient_norm_after_clip": postnorm}.items():
                    sums[key] = sums.get(key, 0.) + float(value.detach() if torch.is_tensor(value) else value)
                tracker.batch(loss, scheduler.get_last_lr()[0], **{k:v for k,v in components.items() if k!="loss"},
                              gradient_norm=norm, gradient_norm_after_clip=postnorm)
                if steps == 1:
                    hardware.update(actual_tensor_devices={k:str(v.device) for k,v in batch.items() if torch.is_tensor(v)},
                                    prediction_device=str(prediction["prediction"].device))
                    atomic(out/"hardware.json", hardware)
                if steps % 25 == 0:
                    atomic(out/"progress.json", {"status":"training", "epoch":epoch, "batch":batch_index+1,
                           "optimizer_steps":steps, "physical_gpu":cfg["physical_gpu"], "last_loss":float(loss),
                           "last_history_loss":float(components["history"]), "last_future_loss":float(components["future"])})
                if steps % 100 == 0 or stop["requested"]:
                    save(out/"last.pt", checkpoint(epoch, True, batch_index+1))
                del prediction, loss, components
                if stop["requested"]:
                    atomic(out/"interrupted.json", {"epoch":epoch,"next_batch":batch_index+1,"optimizer_steps":steps})
                    return
            validation = evaluate(model, "validation")
            values = {k:v/count for k,v in sums.items()}
            row = {"epoch":epoch, "optimizer_steps":steps, "training_loss":values["loss"],
                   "training_components":values, "validation":validation, "epoch_seconds":time.monotonic()-start,
                   "maximum_batch_gradient_norm":maxnorm, "peak_memory_bytes":torch.cuda.max_memory_allocated(),
                   "lr":scheduler.get_last_lr()[0]}
            history.append(row)
            score = validation["overall"]["reconstruction_error"]
            if score < best:
                best = score; bad = 0; save(out/"best.pt", checkpoint(epoch))
            else:
                bad += 1
            save(out/"last.pt", checkpoint(epoch)); atomic(out/"history.json", history)
            tracker.epoch(epoch, values, validation, scheduler.get_last_lr()[0], epoch_seconds=row["epoch_seconds"])
            atomic(out/"progress.json", {"status":"training","completed_epoch":epoch,"best_score":best,
                                        "optimizer_steps":steps,"physical_gpu":cfg["physical_gpu"]})
            print(json.dumps({"epoch":epoch,"training_loss":values["loss"],"validation_overall":score,
                              "validation_future":validation["forecast"]["reconstruction_error"]}), flush=True)
        saved = torch.load(out/"best.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        validation = evaluate(model, "validation")
        difference = abs(validation["overall"]["reconstruction_error"]-history[saved["epoch"]-1]["validation"]["overall"]["reconstruction_error"])
        assert difference < 1e-6
        metrics = {"status":"complete","best_epoch":saved["epoch"],"epochs_run":len(history),
                   "validation":validation,"test":evaluate(model,"test"),
                   "checkpoint_reload_difference":difference,"physical_gpu":cfg["physical_gpu"],
                   "checkpoint_sha256":hashlib.sha256((out/"best.pt").read_bytes()).hexdigest()}
        if cfg["dataset"] == "mimic":
            metrics["test_early_contexts"] = {str(h):evaluate(model,"test",horizon=h) for h in [6,12]}
        atomic(out/"metrics.json", metrics); tracker.final(metrics)
        atomic(out/"progress.json", {"status":"complete","best_epoch":saved["epoch"],"completed_epoch":len(history)})


if __name__ == "__main__":
    main()
