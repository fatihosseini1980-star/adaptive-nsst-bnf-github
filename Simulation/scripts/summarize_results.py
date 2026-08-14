"""Rebuild summary tables from the saved 40-replication raw results."""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "final_raw_results_40rep.csv"
OUT = ROOT / "results" / "recomputed_summary.csv"

metrics = ["RMSE", "MAE", "CRPS", "Coverage", "AIW"]
raw = pd.read_csv(RAW)
rows = []
for (scenario, model), g in raw.groupby(["Scenario", "Model"]):
    row = {"Scenario": scenario, "Model": model, "N": len(g)}
    for metric in metrics:
        x = g[metric].astype(float)
        row[f"{metric}_mean"] = x.mean()
        row[f"{metric}_sd"] = x.std(ddof=1)
        row[f"{metric}_se"] = x.std(ddof=1) / math.sqrt(len(x))
        row[f"{metric}_median"] = x.median()
        row[f"{metric}_iqr"] = x.quantile(0.75) - x.quantile(0.25)
    rows.append(row)

pd.DataFrame(rows).sort_values(["Scenario", "Model"]).to_csv(OUT, index=False)
print(f"Saved {OUT}")
