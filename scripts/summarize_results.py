"""Rebuild manuscript-ready summaries from saved raw result files."""
from pathlib import Path
import math
import numpy as np
import pandas as pd

def summarize_main(raw_path, out_path):
    raw = pd.read_csv(raw_path)
    metrics = ["RMSE","MAE","CRPS","Coverage","AIW"]
    rows = []
    for (s,m), g in raw.groupby(["Scenario","Model"]):
        r = {"Scenario":s,"Model":m,"N":len(g)}
        for c in metrics:
            x = g[c].astype(float)
            r[f"{c}_mean"] = x.mean()
            r[f"{c}_sd"] = x.std(ddof=1)
            r[f"{c}_se"] = x.std(ddof=1)/math.sqrt(len(x))
            r[f"{c}_median"] = x.median()
            r[f"{c}_iqr"] = x.quantile(.75)-x.quantile(.25)
        rows.append(r)
    pd.DataFrame(rows).sort_values(["Scenario","Model"]).to_csv(out_path,index=False)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    summarize_main(
        root/"results"/"final_raw_results_40rep.csv",
        root/"results"/"recomputed_summary.csv",
    )
    print("Saved results/recomputed_summary.csv")
