"""
Reproduce the prior-sensitivity analysis reported in the revised manuscript.

Design:
- Scenarios 4 and 5
- Strong / baseline / weak neural priors = 0.5 / 1 / 2 times baseline SDs
- 5 independent replications per setting
- n = 80
- 1000 maximum VI iterations
- 300 posterior predictive draws
- same fixed predictive-dispersion correction as the final proposed model

Replication IDs 3001--3005 are disjoint from the final Monte Carlo IDs
1001--1040.
"""
from pathlib import Path
from copy import deepcopy
import math
import pandas as pd
import torch

from config import ExperimentConfig
from core import set_default_dtype
from run_simulation import run_one

SETTINGS = {
    "strong": 0.5,
    "baseline": 1.0,
    "weak": 2.0,
}

def main():
    cfg0 = ExperimentConfig()
    cfg0.simulation.n_total = 80
    cfg0.fit.iterations = 1000
    cfg0.fit.posterior_draws = 300
    cfg0.fit.surface_draws = 30
    cfg0.fit.print_every = 0

    set_default_dtype(cfg0.fit.dtype)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    out = Path("outputs/prior_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in (4, 5):
        for label, multiplier in SETTINGS.items():
            for rep in range(3001, 3006):
                cfg = deepcopy(cfg0)
                cfg.prior.mean_neural_sd = 0.35 * multiplier
                cfg.prior.alpha_neural_sd = 0.50 * multiplier
                cfg.prior.range_neural_sd = 0.35 * multiplier

                row = run_one(
                    scenario,
                    rep,
                    "Proposed Adaptive NSST-BNF",
                    cfg,
                    out,
                )
                row.update({
                    "PriorSetting": label,
                    "PriorMultiplier": multiplier,
                    "MeanPriorSD": cfg.prior.mean_neural_sd,
                    "AlphaPriorSD": cfg.prior.alpha_neural_sd,
                    "RangePriorSD": cfg.prior.range_neural_sd,
                })
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    out/"prior_sensitivity_raw_5rep.csv", index=False
                )

    raw = pd.DataFrame(rows)
    metrics = ["RMSE","MAE","CRPS","Coverage","AIW"]
    summary_rows = []
    for (scenario, label), g in raw.groupby(["Scenario","PriorSetting"]):
        r = {"Scenario": scenario, "PriorSetting": label, "N": len(g)}
        for c in metrics:
            x = g[c].astype(float)
            r[f"{c}_mean"] = x.mean()
            r[f"{c}_sd"] = x.std(ddof=1)
            r[f"{c}_se"] = x.std(ddof=1)/math.sqrt(len(x))
            r[f"{c}_median"] = x.median()
            r[f"{c}_iqr"] = x.quantile(.75)-x.quantile(.25)
        summary_rows.append(r)

    pd.DataFrame(summary_rows).sort_values(
        ["Scenario","PriorSetting"]
    ).to_csv(out/"prior_sensitivity_summary_5rep.csv", index=False)

if __name__ == "__main__":
    main()
