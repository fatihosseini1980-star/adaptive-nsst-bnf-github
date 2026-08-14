"""Lightweight integrity checks for the saved reproducibility outputs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "results" / "final_raw_results_40rep.csv")
assert len(raw) == 800, f"Expected 800 final rows, found {len(raw)}"
assert not raw.duplicated(["Scenario", "Replication", "Model"]).any()
counts = raw.groupby(["Scenario", "Model"]).size()
assert (counts == 40).all(), counts[counts != 40]
assert set(raw["Scenario"]) == {1, 2, 3, 4, 5}

cal = pd.read_csv(ROOT / "results" / "development_calibration_raw.csv")
assert set(cal["Replication"]) == {2001, 2002, 2003, 2004, 2005}
assert len(cal[["Scenario", "Replication"]].drop_duplicates()) == 25
chosen = cal.groupby("Factor")["CRPS"].mean().idxmin()
assert abs(float(chosen) - 1.20) < 1e-12, chosen

prior = pd.read_csv(ROOT / "results" / "prior_sensitivity_raw_5rep.csv")
assert set(prior["Replication"]) == {3001, 3002, 3003, 3004, 3005}
assert len(prior) == 30

print("Repository validation passed.")
print("Final rows: 800")
print("Development fits: 25")
print("Chosen predictive-dispersion factor: 1.20")
print("Prior-sensitivity rows: 30")
