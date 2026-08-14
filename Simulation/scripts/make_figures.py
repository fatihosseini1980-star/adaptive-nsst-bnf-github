"""Create compact CRPS and coverage figures from the saved final results."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "results" / "final_raw_results_40rep.csv")

model_order = [
    "Gaussian Neural Model",
    "Stationary Skew-t Neural Model",
    "Sparse Variational GP",
    "Proposed Adaptive NSST-BNF",
]
labels = ["Gaussian", "Stationary skew-t", "SVGP", "Proposed"]

for metric, ylabel, fname, ref in [
    ("CRPS", "CRPS", "crps_final.png", None),
    ("Coverage", "Empirical 95% coverage", "coverage_final.png", 0.95),
]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions, data = [], []
    pos = 1
    centers = []
    for scenario in range(1, 6):
        block = []
        for model in model_order:
            positions.append(pos)
            block.append(pos)
            data.append(df[(df.Scenario == scenario) & (df.Model == model)][metric].values)
            pos += 1
        centers.append(sum(block) / len(block))
        pos += 1
    ax.boxplot(data, positions=positions, widths=0.65, showfliers=False)
    if ref is not None:
        ax.axhline(ref, linestyle="--", linewidth=1)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"S{s}" for s in range(1, 6)])
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across 40 replications")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(0.01, -0.18, "Within each scenario: " + " | ".join(labels), transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / fname}")
