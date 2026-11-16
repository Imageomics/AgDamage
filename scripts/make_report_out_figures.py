"""Figures for the hackathon report-out.

Supplemental CSVs are read from ``reports/report_out_inputs`` by default. Set
``AGDAMAGE_REPORT_INPUTS`` to use another location.
"""

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "report_out_figures"
SCR = Path(
    os.environ.get("AGDAMAGE_REPORT_INPUTS", ROOT / "reports" / "report_out_inputs")
)
OUT.mkdir(parents=True, exist_ok=True)
SY = ["chlorosis", "necrosis", "epinasty", "foliar_distortion"]
NICE = {
    "chlorosis": "Chlorosis",
    "necrosis": "Necrosis",
    "epinasty": "Epinasty",
    "foliar_distortion": "Foliar distortion",
}
C = {"model": "#1f6feb", "date": "#d1495b", "ctrl": "#2a9d8f", "grey": "#8b949e"}
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)

m = pd.read_csv(f"{ROOT}/data/metadata.csv", dtype=str).fillna("")
m = m[m.is_preferred_for_height == "true"]
m50 = m[m.camera_height_cm == "50"]
ever = set(m.loc[m.herbicide_treated == "true", "pot_id"])
DATES = sorted(m.capture_date.unique())


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: design
fig, ax = plt.subplots(figsize=(11, 5.2))
untr = [
    ((m50.capture_date == d) & (m50.herbicide_treated == "false")).sum() for d in DATES
]
symp = [
    ((m50.capture_date == d) & (m50.herbicide_treated == "true")).sum() for d in DATES
]
x = np.arange(len(DATES))
ax.bar(x - 0.2, untr, 0.4, label="untreated reference  (n=173)", color=C["ctrl"])
ax.bar(x + 0.2, symp, 0.4, label="symptom-positive  (n=207)", color=C["model"])
ax.axvspan(1.5, 2.0, color=C["date"], alpha=0.18)
ax.axvline(1.5, color=C["date"], lw=1.2, ls=":", alpha=0.8)
ax.annotate(
    "herbicide applied\n+ irrigation stopped\n(greenhouse error)",
    xy=(1.75, 70),
    xytext=(2.35, 66),
    fontsize=11,
    color=C["date"],
    ha="left",
    va="top",
    arrowprops={"arrowstyle": "->", "color": C["date"], "lw": 1.4},
)
for i, (u, s) in enumerate(zip(untr, symp, strict=False)):
    if u:
        ax.text(i - 0.2, u + 1.5, str(u), ha="center", fontsize=10, color=C["ctrl"])
    if s:
        ax.text(i + 0.2, s + 1.5, str(s), ha="center", fontsize=10, color=C["model"])
ax.set_xticks(x)
ax.set_xticklabels([d[5:] for d in DATES])
ax.set_xlabel("capture date (2026)")
ax.set_ylabel("images (50 cm cohort)")
ax.set_title(
    "The two classes barely share a calendar\n"
    "90% of the untreated reference is pre-treatment baseline",
    loc="left",
)
ax.legend(frameon=False, loc="upper right")
save(fig, "fig1_design_confound")

# ---------------------------------------------------------------- fig 2: date-only
s = pd.read_csv(
    f"{ROOT}/reports/symptom_vs_untreated_logistic/classifier_metric_summary.csv"
)
d0 = pd.read_csv(f"{SCR}/date_only_f1.csv")
f1 = s[s.metric == "f1"]
best = f1.loc[f1.groupby("symptom")["support_images_per_class"].idxmax()]
best = f1.merge(
    best[["symptom", "support_images_per_class"]],
    on=["symptom", "support_images_per_class"],
)
best = best.groupby("symptom")["mean"].max()
dbest = d0.groupby("symptom").f1_date_only.max()
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(SY))
ax.bar(
    x - 0.2,
    [best[k] for k in SY],
    0.4,
    label="best of 12 embedding arms",
    color=C["model"],
)
ax.bar(
    x + 0.2,
    [dbest[k] for k in SY],
    0.4,
    label="calendar date alone (1 number, no image)",
    color=C["date"],
)
for i, k in enumerate(SY):
    ax.text(i - 0.2, best[k] + 0.012, f"{best[k]:.2f}", ha="center", fontsize=11)
    ax.text(i + 0.2, dbest[k] + 0.012, f"{dbest[k]:.2f}", ha="center", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([NICE[k] for k in SY])
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1  (pot-held-out)")
ax.set_title(
    "A single scalar reproduces the result\n"
    "for chlorosis, the date beats every model we trained",
    loc="left",
)
ax.legend(frameon=False, loc="lower right")
save(fig, "fig2_date_only_baseline")

# ---------------------------------------------------------------- fig 3: control drift
sc = pd.read_csv(
    f"{ROOT}/reports/symptom_vs_untreated_logistic/image_symptom_scores.csv"
)
sc["untr"] = sc.herbicide_treated.astype(str).str.lower() == "false"
never = sc[sc.untr & ~sc.pot_id.isin(ever)]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
n_rise = n_mono = 0
for ax, k in zip(axes, SY, strict=False):
    curves = []
    for (_rep, _mod), g in never.groupby(["representation", "model"]):
        t = g.groupby("capture_date")[f"{k}_probability"].mean()
        curves.append(t)
        ax.plot(range(len(t)), t.values, "-", lw=1.1, color=C["grey"], alpha=0.55)
        post = t.loc[[d for d in t.index if d >= "2026-05-11"]].values
        n_rise += post[-1] > post[0]
        n_mono += bool(np.all(np.diff(post) > 0))
    mean = pd.concat(curves, axis=1).mean(axis=1)
    ax.plot(
        range(len(mean)), mean.values, "o-", lw=2.8, ms=7, color=C["date"], zorder=5
    )
    ax.axhline(0.5, color="#444", ls="--", lw=1, alpha=0.5)
    ax.axvspan(1.5, len(mean) - 0.5, color=C["date"], alpha=0.06)
    ax.axvline(1.5, color=C["date"], lw=1.2, ls=":", alpha=0.8)
    if k == SY[0]:
        ax.text(
            1.62,
            0.035,
            "after herbicide + irrigation stop",
            fontsize=9.5,
            color=C["date"],
            va="bottom",
            ha="left",
        )
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels([i[5:] for i in mean.index], rotation=45, fontsize=11)
    ax.set_title(NICE[k], fontsize=13, loc="left")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("mean predicted P(symptom)\non never-treated pots")
axes[0].text(
    0.04,
    0.95,
    "bold = mean of all 12 arms\nthin = each arm",
    transform=axes[0].transAxes,
    fontsize=10,
    va="top",
    color="#222",
)
fig.suptitle(
    f"Six pots we never sprayed \u2014 every model grows more confident they are symptomatic\n"
    f"all {n_rise}/48 arm\u00d7symptom curves rise after treatment ({n_mono}/48 strictly monotonic)",
    fontsize=14,
    x=0.02,
    ha="left",
    y=1.12,
)
fig.text(
    0.02,
    -0.06,
    "These pots were never sprayed, but they were droughted along with everything else — so a positive flag here "
    "is not automatically a false positive.\n"
    "Drought plausibly produces real chlorosis, and marginal necrosis; wilting can read as epinasty. Either reading "
    "breaks the same assumption: whether the model is tracking\n"
    'the calendar or tracking genuine drought symptoms, "untreated" is not a symptom-negative reference class, '
    "and cannot serve as one in training or evaluation.",
    fontsize=10,
    color=C["date"],
    va="top",
    linespacing=1.7,
)
save(fig, "fig3_control_pot_drift")

# ---------------------------------------------------------------- fig 4: what heads encode
z = np.load(
    f"{ROOT}/data/embeddings/rekeyed/plant_only__bioclip.npz", allow_pickle=True
)
idx = {str(i): k for k, i in enumerate(z["ids"])}
F = z["features"]


def feat(ids):
    return F[[idx[i] for i in ids]]


def unit(v):
    return v / np.linalg.norm(v)


def direction(pos, neg):
    X = np.vstack([feat(pos), feat(neg)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return unit(
        LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced")
        .fit(X, y)
        .coef_.ravel()
    )


post = m50[m50.capture_date >= "2026-05-11"]
early = list(m50.loc[m50.capture_date <= "2026-05-08", "image_id"])
late = list(m50.loc[m50.capture_date >= "2026-05-22", "image_id"])
TIME = unit(feat(late).mean(0) - feat(early).mean(0))
TRT = direction(
    list(post.loc[post.herbicide_treated == "true", "image_id"]),
    list(post.loc[post.herbicide_treated == "false", "image_id"]),
)
print(f"  [diagnostic] cos(time axis, treatment axis) = {float(TIME @ TRT):+.2f}")

refA_ids = list(m50.loc[m50.herbicide_treated == "false", "image_id"])
postref = list(post.loc[post.herbicide_treated == "false", "image_id"])
dd = post[post.capture_date == "2026-05-22"]
co = list(dd.loc[dd.source_label == "chlorosis_only", "image_id"])
no = list(dd.loc[dd.source_label == "necrosis_only", "image_id"])
cn = list(dd.loc[dd.source_label == "chlorosis_necrosis", "image_id"])

HEADS = {
    "chlorosis": {
        "Reported\n(untreated ref)": direction(
            list(m50.loc[m50.chlorosis == "true", "image_id"]), refA_ids
        ),
        "Date-matched\ncontrols only": direction(
            list(post.loc[post.chlorosis == "true", "image_id"]), postref
        ),
        "Within one date,\ntreated vs treated": direction(co + cn, no),
    },
    "necrosis": {
        "Reported\n(untreated ref)": direction(
            list(m50.loc[m50.necrosis == "true", "image_id"]), refA_ids
        ),
        "Date-matched\ncontrols only": direction(
            list(post.loc[post.necrosis == "true", "image_id"]), postref
        ),
        "Within one date,\ntreated vs treated": direction(no + cn, co),
    },
}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
for ax, sym in zip(axes, ["chlorosis", "necrosis"], strict=False):
    labs = list(HEADS[sym])
    xs = np.arange(len(labs))
    ct = [abs(float(HEADS[sym][k] @ TIME)) for k in labs]
    cr = [abs(float(HEADS[sym][k] @ TRT)) for k in labs]
    ax.bar(
        xs - 0.2, ct, 0.4, color=C["date"], label="calendar time  (early \u2192 late)"
    )
    ax.bar(xs + 0.2, cr, 0.4, color="#6f42c1", label="treated vs control")
    for i, (a, b) in enumerate(zip(ct, cr, strict=False)):
        ax.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=11)
        ax.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=11)
    ax.set_title(NICE[sym], loc="left")
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("|cosine| of the fitted direction\nwith each nuisance axis")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    fontsize=12,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.07),
)
fig.suptitle(
    "What each classifier's direction is actually aligned with   \u00b7   50 cm cohort\n"
    "the reported heads track the calendar; date-matching swaps that for treatment; "
    "only within-date escapes both",
    fontsize=14,
    x=0.02,
    ha="left",
    y=1.05,
)
save(fig, "fig4_what_heads_encode")

# ---------------------------------------------------------------- fig 5: recovery
rb = pd.read_csv(f"{SCR}/refbalance2.csv")
top = rb.loc[rb.groupby(["contrast", "symptom", "arm"]).shots.idxmax()]
emb = top[top.arm != "DATE-ONLY"]
bb = emb.loc[emb.groupby(["contrast", "symptom"]).roc_auc.idxmax()].set_index(
    ["contrast", "symptom"]
)
dz = top[top.arm == "DATE-ONLY"].set_index(["contrast", "symptom"])
fo = pd.read_csv(f"{SCR}/fifty_only.csv")
fo = fo[fo.cohort == "50 cm only"].set_index(["date", "symptom"])
rb1 = pd.read_csv(f"{SCR}/refbalance.csv")
rb1 = rb1[rb1.arm != "DATE-ONLY"]
rb1 = rb1.loc[rb1.groupby(["reference", "symptom"]).shots.idxmax()]
rb1 = rb1.loc[rb1.groupby(["reference", "symptom"]).roc_auc.idxmax()].set_index(
    ["reference", "symptom"]
)
rows = [
    (
        "Reported\n(untreated ref)",
        "173 ref images\n78 pots",
        rb1.loc[("refA", "chlorosis"), "roc_auc"],
        0.942,
        rb1.loc[("refA", "necrosis"), "roc_auc"],
        0.971,
    ),
    (
        "Date-matched\ncontrols only",
        "18 ref images\n6 pots",
        rb1.loc[("refB", "chlorosis"), "roc_auc"],
        0.529,
        rb1.loc[("refB", "necrosis"), "roc_auc"],
        0.710,
    ),
    (
        "Within one date,\ntreated vs treated",
        "47 pots\n(05-22)",
        fo.loc[("2026-05-22", "chlorosis"), "auc"],
        0.5,
        fo.loc[("2026-05-22", "necrosis"), "auc"],
        0.5,
    ),
]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, (si, sym) in zip(axes, enumerate(["chlorosis", "necrosis"]), strict=False):
    xs = np.arange(3)
    mv = [r[2 + 2 * si] for r in rows]
    dv = [r[3 + 2 * si] for r in rows]
    ax.bar(xs, mv, 0.5, color=C["model"], label="best embedding arm")
    ax.plot(
        xs + 0.30,
        dv,
        "D",
        ms=12,
        color=C["date"],
        ls="none",
        label="date alone",
        zorder=5,
    )
    for i, (a, b) in enumerate(zip(mv, dv, strict=False)):
        ax.text(
            i,
            a + 0.022,
            f"{a:.2f}",
            ha="center",
            fontsize=12,
            color=C["model"],
            weight="bold",
        )
        ax.text(
            i + 0.30,
            b + 0.022,
            f"{b:.2f}",
            ha="center",
            fontsize=11,
            color=C["date"],
            weight="bold",
        )
    ax.axhline(0.5, color=C["grey"], ls="--", lw=1)
    ax.text(
        -0.45, 0.512, "chance", fontsize=10, color=C["grey"], va="bottom", ha="left"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r[0]}\n{r[1]}" for r in rows], fontsize=10.5, linespacing=1.5
    )
    ax.set_title(NICE[sym], loc="left")
    ax.set_ylim(0.4, 1.10)
    ax.set_xlim(-0.5, 2.75)
axes[0].set_ylabel("ROC-AUC  (pot-held-out)")
h5, l5 = axes[0].get_legend_handles_labels()
fig.legend(
    h5,
    l5,
    frameon=False,
    fontsize=12,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.19),
)
fig.suptitle(
    "Control for time and the inflation disappears — but real signal survives\n"
    "50 cm cohort throughout \u00b7 only the gap above the red diamond is attributable to the image",
    fontsize=14,
    x=0.02,
    ha="left",
    y=1.06,
)
save(fig, "fig5_time_controlled_recovery")

# ---------------------------------------------------------------- fig 6: power
fig, ax = plt.subplots(figsize=(11.5, 3.4))
ax.axis("off")
tbl = []
for k in SY:
    p = m50[m50[k] == "true"]
    neg = (
        "necrosis_only"
        if k == "chlorosis"
        else ("chlorosis_only" if k == "necrosis" else "— none —")
    )
    ok = "yes" if k in ("chlorosis", "necrosis") else "NO"
    tbl.append([NICE[k], f"{len(p)}", f"{p.pot_id.nunique()}", neg, ok])
t = ax.table(
    cellText=tbl,
    colLabels=[
        "symptom",
        "positive\nimages",
        "positive\npots",
        "treated-plant\nnegatives",
        "time-controlled\ncontrast?",
    ],
    cellLoc="center",
    loc="center",
)
t.auto_set_font_size(False)
t.set_fontsize(12)
t.scale(1, 2.1)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#d0d7de")
    if r == 0:
        cell.set_facecolor("#f0f3f6")
        cell.set_text_props(weight="bold")
    elif c == 4:
        cell.set_facecolor("#e8f5e9" if tbl[r - 1][4] == "yes" else "#fdecea")
ax.set_title(
    "Where the data runs out\n"
    "epinasty and foliar distortion have no negatives except untreated controls",
    loc="left",
    fontsize=15,
    pad=18,
)
save(fig, "fig6_power_and_data_needs")
print("\nall figures ->", OUT)
