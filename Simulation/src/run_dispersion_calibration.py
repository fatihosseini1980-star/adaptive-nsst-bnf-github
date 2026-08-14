"""Select the proposed-model predictive-dispersion multiplier on development data.

Frozen development design used for the reported revision:
- scenarios 1--5
- development replication IDs 2001--2005 (25 fitted datasets total)
- n = 80, 80/20 train/test split
- 1000 maximum VI iterations
- 300 raw posterior predictive draws
- candidate multipliers: 0.75, 0.85, 0.95, 1.00, 1.10, 1.20, 1.35, 1.50
- selection criterion: lowest mean CRPS pooled over the 25 development fits

The selected value is 1.20. These replication IDs are disjoint from the final
Monte Carlo IDs 1001--1040 and the prior-sensitivity IDs 3001--3005.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig
from core import (
    set_default_dtype, to_tensor, simulate_dataset, AdaptiveNSSTBNF,
    fit_model, predictive_metrics,
)

FACTORS = [0.75, 0.85, 0.95, 1.00, 1.10, 1.20, 1.35, 1.50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/development_calibration")
    ap.add_argument("--rep-start", type=int, default=2001)
    ap.add_argument("--reps-per-scenario", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--n-total", type=int, default=80)
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.simulation.n_total = args.n_total
    cfg.fit.iterations = args.iterations
    cfg.fit.posterior_draws = args.draws
    cfg.fit.print_every = 0
    set_default_dtype(cfg.fit.dtype)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in range(1, 6):
        for rep in range(args.rep_start, args.rep_start + args.reps_per_scenario):
            seed = cfg.simulation.seed_base + 1000 * scenario + rep
            d = simulate_dataset(scenario, seed, cfg.simulation)
            tr, te = d["train_idx"], d["test_idx"]
            xtr = to_tensor(d["coords"][tr])
            xte = to_tensor(d["coords"][te])
            ytr = to_tensor(d["y"][tr])
            yte = d["y"][te]

            torch.manual_seed(seed + 17)
            model = AdaptiveNSSTBNF(cfg.fit, cfg.prior, cfg.simulation.jitter)
            fit_model(model, ytr, xtr, cfg.fit, seed=seed + 31)
            samples = model.predictive_samples(ytr, xtr, xte, n_draws=args.draws)
            center = samples.mean(axis=0, keepdims=True)

            for factor in FACTORS:
                scaled = center + factor * (samples - center)
                rows.append({
                    "Scenario": scenario,
                    "Replication": rep,
                    "Factor": factor,
                    **predictive_metrics(yte, scaled, nominal=0.95),
                })

            pd.DataFrame(rows).to_csv(out / "development_calibration_raw.csv", index=False)

    raw = pd.DataFrame(rows)
    summary = raw.groupby("Factor")[["RMSE", "MAE", "CRPS", "Coverage", "AIW"]].agg(["mean", "std"])
    summary.to_csv(out / "development_calibration_summary.csv")
    crps_means = raw.groupby("Factor")["CRPS"].mean()
    chosen = float(crps_means.idxmin())
    (out / "chosen_factor.txt").write_text(f"{chosen:.2f}\n", encoding="utf-8")

    print(summary)
    print(f"\nChosen factor by pooled development CRPS: {chosen:.2f}")


if __name__ == "__main__":
    main()
