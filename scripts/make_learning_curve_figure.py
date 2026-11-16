"""Learning curves per symptom, with the date-only null overlaid."""

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCR = Path(
    os.environ.get("AGDAMAGE_REPORT_INPUTS", ROOT / "reports" / "report_out_inputs")
)
INK, MUTE = "#16201c", "#7c8a83"
REPC = {"full": "#1f6feb", "cropped": "#9a6c15", "segmented": "#1d5c56"}
NULL = "#a8412f"
SY = ["chlorosis", "necrosis", "epinasty", "foliar_distortion"]
NICE = {
    "chlorosis": "Chlorosis",
    "necrosis": "Necrosis",
    "epinasty": "Epinasty",
    "foliar_distortion": "Foliar distortion",
}
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    }
)

s = pd.read_csv(
    f"{ROOT}/reports/symptom_vs_untreated_logistic/classifier_metric_summary.csv"
)
f1 = s[s.metric == "f1"]
d0 = pd.read_csv(f"{SCR}/date_only_f1.csv")

fig, axes = plt.subplots(1, 4, figsize=(17, 4.9), sharey=True)
for ax, sym in zip(axes, SY, strict=False):
    g = f1[f1.symptom == sym]
    for rep, c in REPC.items():
        r = (
            g[g.representation == rep]
            .groupby("support_images_per_class")["mean"]
            .mean()
        )
        lo = g[g.representation == rep].groupby("support_images_per_class")["q10"].min()
        hi = g[g.representation == rep].groupby("support_images_per_class")["q90"].max()
        x = np.arange(len(r))
        ax.fill_between(x, lo.values, hi.values, color=c, alpha=0.10, lw=0)
        ax.plot(x, r.values, "o-", lw=2.2, ms=6, color=c, label=rep)
        ticks = list(r.index)
    n = d0[d0.symptom == sym].sort_values("support_images_per_class")
    ax.plot(
        np.arange(len(n)),
        n.f1_date_only.values,
        "--",
        lw=2.4,
        color=NULL,
        label="date alone (no image)",
    )
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_title(NICE[sym], loc="left", fontsize=13)
    ax.set_xlabel("support images per class")
    ax.set_ylim(0.2, 1.0)
    npos = (
        g.symptom.size
        and {"chlorosis": 130, "necrosis": 54, "epinasty": 36, "foliar_distortion": 11}[
            sym
        ]
    )
    pots = {"chlorosis": 52, "necrosis": 31, "epinasty": 18, "foliar_distortion": 9}[
        sym
    ]
    ax.text(
        0.03,
        0.04,
        f"{npos} positives · {pots} pots",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTE,
    )
axes[0].set_ylabel("F1  (pot-held-out)")
axes[0].legend(frameon=False, fontsize=10, loc="upper left")
fig.suptitle(
    "More labelled images helps — but only epinasty and foliar distortion ever clear the null\n"
    "each line averages the four encoders for that representation; band spans their 10th–90th percentiles",
    fontsize=14,
    x=0.02,
    ha="left",
    y=1.13,
)
fig.text(
    0.02,
    -0.12,
    "Chlorosis and necrosis plateau below or at the date-only baseline at every support size: the extra labels buy accuracy on a task the calendar already solves.\n"
    "Epinasty and foliar distortion rise above it, but their negatives come only from untreated controls, so the gap is not evidence of symptom detection either.\n"
    "Read this as a data-efficiency curve for the pipeline, not as symptom performance — the honest symptom numbers are the within-date contrasts.",
    fontsize=10,
    color=INK,
    va="top",
    linespacing=1.7,
)
out = f"{ROOT}/reports/report_out_figures/fig11_learning_curves.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
