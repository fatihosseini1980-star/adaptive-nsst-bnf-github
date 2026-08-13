"""Reproduce the CRPS and coverage figures from the saved final results."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"results"/"final_raw_results_40rep.csv")

model_order = [
    "Gaussian Neural Model",
    "Stationary Skew-t Neural Model",
    "Sparse Variational GP",
    "Proposed Adaptive NSST-BNF",
]
labels = {
    "Gaussian Neural Model":"Gaussian Neural Model",
    "Stationary Skew-t Neural Model":"Stationary Skew-t Neural Model",
    "Sparse Variational GP":"SVGP",
    "Proposed Adaptive NSST-BNF":"Proposed Adaptive NSST-BNF",
}
colors = ["#9fd0ea","#a8e6a3","#f3d37a","#f7b0b0"]

def draw(metric, ylabel, title, filename, ylim=None, reference=None):
    fig, ax = plt.subplots(figsize=(13.5,7.2))
    positions=[]; data=[]; box_colors=[]; centers=[]
    x=1.0
    for scenario in range(1,6):
        start=x
        for model,color in zip(model_order,colors):
            positions.append(x)
            data.append(df[(df.Scenario==scenario)&(df.Model==model)][metric].values)
            box_colors.append(color)
            x += 1
        centers.append((start+x-1)/2)
        x += 1.4

    bp=ax.boxplot(
        data,positions=positions,widths=.7,patch_artist=True,showfliers=True,
        medianprops=dict(color="red",linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    for patch,color in zip(bp["boxes"],box_colors):
        patch.set_facecolor(color); patch.set_alpha(.8)

    for i in range(1,5):
        ax.axvline((positions[i*4-1]+positions[i*4])/2+.5,
                   linestyle="--",linewidth=1,color="#bdbdbd")

    if ylim is not None:
        ax.set_ylim(*ylim)
    ymin,ymax=ax.get_ylim()
    y=ymax-.055*(ymax-ymin)
    for c,s in zip(centers,range(1,6)):
        ax.text(c,y,f"Scenario {s}",ha="center",va="center",
                fontsize=14,fontweight="bold")

    if reference is not None:
        ax.axhline(reference,linestyle="--",linewidth=1.2,color="red")

    ax.set_xticks([])
    ax.set_ylabel(ylabel,fontsize=17,fontweight="bold")
    ax.set_title(title,fontsize=20,fontweight="bold")
    ax.grid(axis="y",linestyle="--",alpha=.45)

    handles=[Patch(facecolor=c,label=labels[m],alpha=.8)
             for m,c in zip(model_order,colors)]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.08),
              ncol=4,fontsize=11,frameon=True)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/filename,dpi=300,bbox_inches="tight")
    plt.close(fig)

draw("CRPS","CRPS","CRPS across Scenarios and Models (Lower is Better)",
     "crps_final.png",ylim=(0,.70))
draw("Coverage","Coverage","Coverage across Scenarios and Models",
     "coverage_final.png",ylim=(.85,1.02),reference=.95)
