"""Fit/evaluate one ozone2 model restart in an isolated process."""
import argparse
from pathlib import Path
import pandas as pd
import torch

from config import ExperimentConfig
from core import set_default_dtype
from ozone2 import load_ozone2_csv, make_design, prepare_arrays, evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["gaussian", "adaptive"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig(); cfg.fit.restart_seeds = (args.seed,)
    if args.quick:
        cfg.fit.iterations_gaussian = 80
        cfg.fit.iterations_adaptive = 100
        cfg.fit.mean_pretrain_steps = 80
        cfg.fit.posterior_draws = 60
        cfg.fit.surface_draws = 20
    set_default_dtype(cfg.numerical.dtype); torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    df = load_ozone2_csv(args.data); design = make_design(df, cfg.ozone); arr = prepare_arrays(df, design)
    name = "Gaussian BNF" if args.model == "gaussian" else "Adaptive NSST-BNF"
    summary, restart, grid, pred, trace = evaluate_model(name, arr, cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    design["qc"].to_csv(out / "station_qc.csv", index=False)
    pd.DataFrame({
        "role": (["fit"]*len(design["fit_stations"]) + ["calibration"]*len(design["calibration_stations"]) + ["test"]*len(design["test_stations"])),
        "station": design["fit_stations"] + design["calibration_stations"] + design["test_stations"],
    }).to_csv(out / "station_split.csv", index=False)
    pd.DataFrame({"day": design["days"]}).to_csv(out / "selected_days.csv", index=False)
    pd.DataFrame([{"temporal_weight": arr["temporal_weight"]}]).to_csv(out / "space_time_scaling.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    restart.to_csv(out / "restart.csv", index=False)
    grid.to_csv(out / "calibration_grid.csv", index=False)
    pred.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(trace).to_csv(out / "trace.csv", index=False)


if __name__ == "__main__":
    main()
