from pathlib import Path
import argparse, pandas as pd, numpy as np, torch
from config import ExperimentConfig
from core import set_default_dtype,to_tensor,simulate_dataset,AdaptiveNSSTBNF,fit_model,predictive_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario',type=int,required=True)
    ap.add_argument('--reps',type=int,default=10)
    ap.add_argument('--rep-start',type=int,default=2001)
    ap.add_argument('--iterations',type=int,default=1000)
    ap.add_argument('--draws',type=int,default=300)
    ap.add_argument('--n-total',type=int,default=80)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    cfg=ExperimentConfig(); cfg.simulation.n_total=a.n_total; cfg.fit.iterations=a.iterations; cfg.fit.posterior_draws=a.draws; cfg.fit.print_every=0
    set_default_dtype(cfg.fit.dtype)
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    factors=[1.0,1.15,1.25,1.35,1.5,1.7,2.0]
    rows=[]; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for rep in range(a.rep_start,a.rep_start+a.reps):
        seed=cfg.simulation.seed_base+1000*a.scenario+rep
        d=simulate_dataset(a.scenario,seed,cfg.simulation)
        tr,te=d['train_idx'],d['test_idx']
        xtr,xte=to_tensor(d['coords'][tr]),to_tensor(d['coords'][te]); ytr=to_tensor(d['y'][tr]); yte=d['y'][te]
        torch.manual_seed(seed+17)
        model=AdaptiveNSSTBNF(cfg.fit,cfg.prior,cfg.simulation.jitter)
        trace=fit_model(model,ytr,xtr,cfg.fit,seed=seed+31)
        samples=model.predictive_samples(ytr,xtr,xte,n_draws=a.draws)
        center=samples.mean(axis=0,keepdims=True)
        for f in factors:
            ss=center+f*(samples-center)
            rows.append({'Scenario':a.scenario,'Replication':rep,'Factor':f,**predictive_metrics(yte,ss,.95)})
        pd.DataFrame(rows).to_csv(out/'raw_calibration.csv',index=False)
    raw=pd.DataFrame(rows)
    sm=raw.groupby('Factor')[['RMSE','MAE','CRPS','Coverage','AIW']].agg(['mean','std']).reset_index()
    sm.to_csv(out/'summary_calibration.csv',index=False)
    print(sm.to_string(index=False))
if __name__=='__main__': main()
