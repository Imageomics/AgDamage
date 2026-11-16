"""Is the rise on untreated plants real? A model-free pixel statistic beside the
classifier score, both split control vs treated."""

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
CTRL, TRT = "#1d5c56", "#a8412f"
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

px = pd.read_csv(f"{SCR}/pixels.csv")
sc = pd.read_csv(
    f"{ROOT}/reports/symptom_vs_untreated_logistic/image_symptom_scores.csv"
)
md = pd.read_csv(f"{ROOT}/data/metadata.csv", dtype=str).fillna("")
ever = set(md.loc[md.herbicide_treated == "true", "pot_id"])
sc["arm"] = np.where(sc.pot_id.isin(ever), "treated", "control")
sc = sc[(sc.representation == "segmented") & (sc.model == "BioCLIP")]

DATES = ["2026-05-07", "2026-05-08", "2026-05-11", "2026-05-15", "2026-05-22"]


def agg(df, col):
    g = df[df.date.isin(DATES)] if "date" in df else df[df.capture_date.isin(DATES)]
    key = "date" if "date" in df else "capture_date"
    out = {}
    for a in ["control", "treated"]:
        s = g[g.arm == a].groupby(key)[col]
        out[a] = (
            s.mean().reindex(DATES),
            s.sem().reindex(DATES),
            s.size().reindex(DATES),
        )
    return out


fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
panels = [
    (
        axes[0],
        agg(px, "rg"),
        "Pixel senescence index  (R−G)/(R+G)",
        "measured from plant-only pixels — no model involved",
    ),
    (
        axes[1],
        agg(sc, "chlorosis_probability"),
        "Classifier score for chlorosis",
        "segmented / BioCLIP, out-of-fold",
    ),
]
for ax, a, title, sub in panels:
    for _name, col in [("control", "control"), ("treated", "treated")]:
        mu, se, n = a[col]
        c = CTRL if col == "control" else TRT
        ax.plot(
            range(len(DATES)),
            mu.values,
            "o-",
            lw=2.4,
            ms=7,
            color=c,
            label="never treated (n=6/date)"
            if col == "control"
            else "treated (n≈70/date)",
        )
        ax.fill_between(
            range(len(DATES)),
            (mu - se).values,
            (mu + se).values,
            color=c,
            alpha=0.18,
            lw=0,
        )
    ax.axvline(1.5, color=TRT, ls=":", lw=1.2, alpha=0.8)
    ax.set_xticks(range(len(DATES)))
    ax.set_xticklabels([d[5:] for d in DATES], rotation=45)
    ax.text(
        0,
        1.13,
        title,
        transform=ax.transAxes,
        fontsize=13,
        color=INK,
        va="bottom",
        weight="bold",
    )
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10, color=MUTE, va="bottom")
axes[0].legend(frameon=False, fontsize=10.5, loc="upper left")
axes[0].text(
    1.6,
    axes[0].get_ylim()[0] + 0.004,
    "after herbicide\n+ irrigation stop",
    fontsize=9,
    color=TRT,
    va="bottom",
    linespacing=1.4,
)

fig.suptitle(
    "The untreated plants really did change — and the treated ones changed more\n"
    "left panel uses no classifier at all, so it separates 'plants changed' from 'model reads the calendar'",
    fontsize=14,
    x=0.02,
    ha="left",
    y=1.18,
)
fig.text(
    0.02,
    -0.10,
    "Both curves rise on plants that were never sprayed, so a positive flag on a control is not automatically a false positive — "
    "the drought produced genuine colour change.\n"
    "Treated plants separate from controls in both panels, so herbicide adds signal above that trend. Caution: the two pre-treatment "
    "dates differ from each other by as much as some\npost-treatment steps (control index −0.030 → −0.050), so day-to-day acquisition "
    "variation is not negligible even in the pixel statistic.",
    fontsize=10,
    color=INK,
    va="top",
    linespacing=1.7,
)
out = f"{ROOT}/reports/report_out_figures/fig10_pixel_evidence.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
