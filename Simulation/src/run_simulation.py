
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
import torch

# Keep parallel Monte Carlo workers from oversubscribing CPU threads.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from config import ExperimentConfig, PriorConfig, SimulationConfig, FitConfig
from svgp import SparseVariationalGP, choose_inducing, fit_svgp
from core import (
    set_default_dtype, to_tensor, simulate_dataset,
    AdaptiveNSSTBNF, StationarySkewTBNF, GaussianBNF,
    fit_model, predictive_metrics, convergence_diagnostics, r2_score, adaptive_correlation
)

MODEL_FACTORIES = {
    "Gaussian Neural Model": GaussianBNF,
    "Stationary Skew-t Neural Model": StationarySkewTBNF,
    "Sparse Variational GP": "SVGP",
    "Proposed Adaptive NSST-BNF": AdaptiveNSSTBNF,
}

def run_one(scenario, rep, model_name, cfg, out_dir):
    seed=cfg.simulation.seed_base + 1000*scenario + rep
    d=simulate_dataset(scenario, seed, cfg.simulation)
    tr,te=d["train_idx"],d["test_idx"]
    xtr=to_tensor(d["coords"][tr])
    xte=to_tensor(d["coords"][te])
    ytr=to_tensor(d["y"][tr])
    yte=d["y"][te]

    torch.manual_seed(seed+17)
    t0=time.perf_counter()
    if model_name=="Sparse Variational GP":
        z=choose_inducing(d["coords"][tr],cfg.fit.svgp_inducing,seed=seed+19)
        model=SparseVariationalGP(z,jitter=cfg.simulation.jitter)
        svgp_iterations=max(cfg.fit.iterations,1000)
        trace=fit_svgp(
            model,ytr,xtr,iterations=svgp_iterations,
            lr=cfg.fit.svgp_learning_rate,print_every=cfg.fit.print_every,
            early_stop_min_iter=cfg.fit.early_stop_min_iter,
            early_stop_window=cfg.fit.early_stop_window,
            early_stop_check_every=cfg.fit.early_stop_check_every,
            early_stop_rel_slope=cfg.fit.early_stop_rel_slope,
            early_stop_cv=cfg.fit.early_stop_cv
        )
        samples=model.predictive_samples(
            xte,n_draws=cfg.fit.posterior_draws
        )
    else:
        model=MODEL_FACTORIES[model_name](cfg.fit,cfg.prior,cfg.simulation.jitter)
        trace=fit_model(model,ytr,xtr,cfg.fit,seed=seed+31)
        samples=model.predictive_samples(
            ytr,xtr,xte,n_draws=cfg.fit.posterior_draws
        )
        if model_name=="Proposed Adaptive NSST-BNF":
            # Fixed development-set variance correction for mean-field VI.
            # It changes interval dispersion only, not posterior predictive means.
            center=samples.mean(axis=0,keepdims=True)
            samples=center+cfg.fit.adaptive_predictive_scale*(samples-center)
    fit_sec=time.perf_counter()-t0
    met=predictive_metrics(yte,samples,nominal=.95)
    conv=convergence_diagnostics(trace)

    row={
        "Scenario":scenario,"Replication":rep,"Model":model_name,
        **met,"FitSeconds":fit_sec,
        "IterationsUsed":int(trace[-1]["iteration"]),
        "EarlyStopped":int(trace[-1].get("early_stopped",0)),
        **conv
    }

    if model_name=="Proposed Adaptive NSST-BNF" and scenario in (4,5):
        allx=to_tensor(d["coords"])
        ahat,phat=model.posterior_surface_summary(xtr,allx,cfg.fit.surface_draws)
        delta_true=d["alpha_true"]/np.sqrt(1.0+d["alpha_true"]**2)
        delta_hat=ahat/np.sqrt(1.0+ahat**2)
        xt_all=to_tensor(d["coords"])
        ct=adaptive_correlation(
            xt_all,xt_all,to_tensor(d["phi_true"]),to_tensor(d["phi_true"])
        ).cpu().numpy()
        ch=adaptive_correlation(
            xt_all,xt_all,to_tensor(phat),to_tensor(phat)
        ).cpu().numpy()
        iu=np.triu_indices(len(phat),1)
        row.update({
            "Alpha_RMSE":float(np.sqrt(np.mean((ahat-d["alpha_true"])**2))),
            "Alpha_R2":r2_score(d["alpha_true"],ahat),
            "Delta_RMSE":float(np.sqrt(np.mean((delta_hat-delta_true)**2))),
            "Delta_R2":r2_score(delta_true,delta_hat),
            "Phi_RMSE":float(np.sqrt(np.mean((phat-d["phi_true"])**2))),
            "Phi_R2":r2_score(d["phi_true"],phat),
            "Corr_Frobenius_RelErr":float(
                np.linalg.norm(ch-ct,"fro")/(np.linalg.norm(ct,"fro")+1e-12)
            ),
            "Corr_OffDiag_R":float(np.corrcoef(ct[iu],ch[iu])[0,1]),
        })
        surf=pd.DataFrame({
            "s1":d["coords"][:,0],"s2":d["coords"][:,1],"t":d["coords"][:,2],
            "alpha_true":d["alpha_true"],"alpha_hat":ahat,
            "phi_true":d["phi_true"],"phi_hat":phat
        })
        surf.to_csv(out_dir/f"surface_s{scenario}_rep{rep:02d}.csv",index=False)

    pd.DataFrame(trace).to_csv(
        out_dir/f"trace_s{scenario}_rep{rep:02d}_{model_name.replace(' ','_').replace('$','')}.csv",
        index=False
    )
    return row

def summarize(raw):
    metrics=["RMSE","MAE","CRPS","Coverage","AIW","FitSeconds",
             "Alpha_RMSE","Alpha_R2","Delta_RMSE","Delta_R2","Phi_RMSE","Phi_R2","Corr_Frobenius_RelErr","Corr_OffDiag_R",
             "tail_slope","tail_cv"]
    metrics=[m for m in metrics if m in raw.columns]
    rows=[]
    for (s,m),g in raw.groupby(["Scenario","Model"]):
        row={"Scenario":s,"Model":m,"N":len(g)}
        for col in metrics:
            z=g[col].dropna()
            if len(z)==0: continue
            row[f"{col}_mean"]=z.mean()
            row[f"{col}_sd"]=z.std(ddof=1) if len(z)>1 else np.nan
            row[f"{col}_median"]=z.median()
            row[f"{col}_iqr"]=z.quantile(.75)-z.quantile(.25)
            row[f"{col}_se"]=z.std(ddof=1)/np.sqrt(len(z)) if len(z)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--reps",type=int,default=None)
    p.add_argument("--rep-start",type=int,default=1)
    p.add_argument("--iterations",type=int,default=None)
    p.add_argument("--draws",type=int,default=None)
    p.add_argument("--surface-draws",type=int,default=None)
    p.add_argument("--n-total",type=int,default=None)
    p.add_argument("--scenarios",default="1,2,3,4,5")
    p.add_argument("--models",default="all")
    p.add_argument("--out",default="outputs/final_simulation")
    args=p.parse_args()

    cfg=ExperimentConfig()
    if args.smoke:
        cfg.simulation.n_total=40
        cfg.simulation.n_rep=1
        cfg.fit.iterations=30
        cfg.fit.posterior_draws=40
        cfg.fit.surface_draws=20
        cfg.fit.mean_pretrain_steps=50
        cfg.fit.print_every=0
    if args.reps is not None:
        cfg.simulation.n_rep=args.reps
    if args.iterations is not None:
        cfg.fit.iterations=args.iterations
    if args.draws is not None:
        cfg.fit.posterior_draws=args.draws
    if args.surface_draws is not None:
        cfg.fit.surface_draws=args.surface_draws
    if args.n_total is not None:
        cfg.simulation.n_total=args.n_total

    set_default_dtype(cfg.fit.dtype)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cfg.save(out/"config.json")
    scenarios=[int(x) for x in args.scenarios.split(",")]
    models=list(MODEL_FACTORIES) if args.models=="all" else args.models.split(",")

    rows=[]
    for s in scenarios:
        for rep in range(args.rep_start,args.rep_start+cfg.simulation.n_rep):
            for model in models:
                print(f"[S{s} rep {rep}/{cfg.simulation.n_rep}] {model}")
                row=run_one(s,rep,model,cfg,out)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out/"raw_results.csv",index=False)
    raw=pd.DataFrame(rows)
    summarize(raw).to_csv(out/"summary_mean_sd_iqr.csv",index=False)
    print("\nSaved:",out.resolve())

if __name__=="__main__":
    main()
