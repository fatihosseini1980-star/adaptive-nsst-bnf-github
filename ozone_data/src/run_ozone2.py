"""Orchestrate the ozone2 analysis without importing torch in the parent process.

Each restart is executed in a fresh Python process. Selection is based only on
`objective` from training. The parent process only aggregates CSV outputs.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
from config import ExperimentConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/ozone2")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.quick:
        cfg.fit.restart_seeds = (111,)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")

    selected = []
    first_worker_dir = None
    worker = Path(__file__).with_name("run_ozone2_restart.py")
    for slug in ("gaussian", "adaptive"):
        restart_rows = []
        for seed in cfg.fit.restart_seeds:
            rdir = out / "restarts" / slug / f"seed_{seed}"
            if first_worker_dir is None:
                first_worker_dir = rdir
            cmd = [sys.executable, str(worker), "--data", args.data,
                   "--model", slug, "--seed", str(seed), "--out", str(rdir)]
            if args.quick:
                cmd.append("--quick")
            print(f"[{slug}] seed={seed}", flush=True)
            subprocess.run(cmd, check=True)
            restart_rows.append(pd.read_csv(rdir / "restart.csv").iloc[0].to_dict())

        restarts = pd.DataFrame(restart_rows).sort_values(["objective", "seed"]).reset_index(drop=True)
        restarts.to_csv(out / f"{slug}_restarts.csv", index=False)
        best_seed = int(restarts.iloc[0].seed)
        best = out / "restarts" / slug / f"seed_{best_seed}"
        for f in ("summary.csv", "calibration_grid.csv", "test_predictions.csv", "trace.csv"):
            shutil.copy2(best / f, out / f"{slug}_{f}")
        selected.append(pd.read_csv(best / "summary.csv").iloc[0].to_dict())
        print(f"[{slug}] selected seed={best_seed} by training objective", flush=True)

    for f in ("station_qc.csv", "station_split.csv", "selected_days.csv", "space_time_scaling.csv"):
        shutil.copy2(first_worker_dir / f, out / f)
    final = pd.DataFrame(selected)
    final.to_csv(out / "two_model_comparison.csv", index=False)
    print("\nFinal two-model comparison")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
