"""
Final simulation driver for the revised manuscript.

Design frozen after development/calibration:
- 5 scenarios
- 40 independent replications per scenario
- n = 80 observations per replication
- 80/20 train/test split
- 4 models
- 300 posterior predictive draws
- proposed-model predictive dispersion factor = 1.20
- Gaussian/Stationary skew-t neural models: max 600 iterations
- SVGP: max 1000 iterations
- Proposed Adaptive NSST-BNF: max 1000 iterations

Final replication IDs are 1001--1040 and are distinct from development/
calibration IDs. The script is resumable: completed Scenario x Replication x
Model rows are skipped.
"""
from pathlib import Path
from copy import deepcopy
import pandas as pd
import torch

from config import ExperimentConfig
from core import set_default_dtype
from run_simulation import run_one

MODELS = [
    "Gaussian Neural Model",
    "Stationary Skew-t Neural Model",
    "Sparse Variational GP",
    "Proposed Adaptive NSST-BNF",
]

def main():
    cfg0=ExperimentConfig()
    cfg0.simulation.n_total=80
    cfg0.fit.posterior_draws=300
    cfg0.fit.surface_draws=50
    cfg0.fit.print_every=0

    set_default_dtype(cfg0.fit.dtype)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    out=Path("outputs/final_40rep")
    out.mkdir(parents=True,exist_ok=True)
    raw_path=out/"final_raw_results_40rep.csv"

    if raw_path.exists():
        rows=pd.read_csv(raw_path).to_dict("records")
    else:
        rows=[]

    done={(int(r["Scenario"]),int(r["Replication"]),r["Model"]) for r in rows}

    for scenario in range(1,6):
        for rep in range(1001,1041):
            for model in MODELS:
                key=(scenario,rep,model)
                if key in done:
                    continue

                cfg=deepcopy(cfg0)
                cfg.fit.iterations = 1000 if model in (
                    "Sparse Variational GP",
                    "Proposed Adaptive NSST-BNF",
                ) else 600

                row=run_one(scenario,rep,model,cfg,out)
                rows.append(row)
                done.add(key)
                pd.DataFrame(rows).to_csv(raw_path,index=False)

    raw=pd.DataFrame(rows)
    metrics=["RMSE","MAE","CRPS","Coverage","AIW"]
    summary=raw.groupby(["Scenario","Model"])[metrics].agg(
        ["mean","std","median"]
    )
    summary.to_csv(out/"final_summary.csv")
    print(summary)

if __name__=="__main__":
    main()
