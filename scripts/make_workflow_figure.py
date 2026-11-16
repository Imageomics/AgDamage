"""Bare workflow strip (fig8) and pot-grouped CV explainer (fig9).
Deliberately unannotated — narration lives in the speaker notes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
INK, MUTE, RULE = "#16201c", "#7c8a83", "#d7ded8"
TEAL, TSOFT = "#1d5c56", "#e4efec"
BRICK, BSOFT = "#a8412f", "#f7e8e4"
PANEL = "#fbfcfb"

# ==================== fig 8 ====================
fig, ax = plt.subplots(figsize=(17, 4.4))
fig.patch.set_facecolor("white")
Y, H = 3, 42
W = [26, 24, 34, 20, 34, 30]
GAP = 5.0
X, cur = [], 2.0
for w in W:
    X.append(cur)
    cur += w + GAP
ax.set_xlim(0, cur - GAP + 2.0)
ax.set_ylim(0, H + 6)
ax.axis("off")


def panel(x, w, title):
    ax.add_patch(Rectangle((x, Y), w, H, fc=PANEL, ec=RULE, lw=1.1, zorder=1))
    ax.text(
        x + w / 2,
        Y + H - 4.0,
        title,
        fontsize=12,
        color=INK,
        weight="bold",
        ha="center",
        va="top",
    )


def arrow(x):
    ax.add_patch(
        FancyArrowPatch(
            (x, Y + H / 2),
            (x + 3.4, Y + H / 2),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.6,
            color=TEAL,
            zorder=4,
        )
    )


def rows(x, w, labels, right=None, top=None, h=5.8, gap=1.4, fs=9.5):
    for k, lab in enumerate(labels):
        yy = top - k * (h + gap)
        ax.add_patch(
            Rectangle((x + 3.0, yy), w - 6.0, h, fc="white", ec=RULE, lw=1.0, zorder=2)
        )
        if right:
            ax.text(x + 4.4, yy + h / 2, lab, fontsize=fs, color=INK, va="center")
            ax.text(
                x + w - 4.4,
                yy + h / 2,
                right[k],
                fontsize=8.8,
                color=TEAL,
                va="center",
                ha="right",
                family="monospace",
            )
        else:
            ax.text(
                x + w / 2,
                yy + h / 2,
                lab,
                fontsize=fs,
                color=INK,
                ha="center",
                va="center",
            )


# 01 cohort
panel(X[0], W[0], "Cohort")
cw, chh, gx, gy = 2.5, 3.2, X[0] + 5.5, Y + 6.5
for i in range(6):
    yy = gy + (5 - i) * (chh + 0.9)
    for j in range(6):
        ax.add_patch(
            Rectangle(
                (gx + j * (cw + 0.7), yy), cw, chh, fc=TSOFT, ec=TEAL, lw=0.8, zorder=2
            )
        )
ax.text(
    gx - 2.0,
    gy + 3 * (chh + 0.9),
    "pots",
    ha="right",
    va="center",
    fontsize=8.5,
    color=MUTE,
    rotation=90,
)
ax.text(gx + 3 * (cw + 0.7), gy - 3.2, "time →", ha="center", fontsize=8.5, color=MUTE)

# 02 representation
panel(X[1], W[1], "Representation")
rows(X[1], W[1], ["full frame", "cropped", "segmented"], top=Y + 24, h=6.4, gap=2.0)
arrow(X[1] - 4.2)

# 03 encoder
panel(X[2], W[2], "Frozen encoder")
rows(
    X[2],
    W[2],
    ["BioCLIP", "BioCLIP 2", "BioCLIP 2.5", "DINOv3 ViT-7B/16"],
    right=["512", "768", "1024", "4096"],
    top=Y + 25,
)
arrow(X[2] - 4.2)

# 04 embeddings
panel(X[3], W[3], "Embeddings")
for r in range(3):
    for c in range(4):
        ax.add_patch(
            Rectangle(
                (X[3] + 2.8 + c * 4.0, Y + 20 - r * 4.4),
                3.4,
                3.6,
                fc=TSOFT,
                ec=TEAL,
                lw=0.9,
                zorder=2,
            )
        )
ax.text(X[3] + W[3] / 2, Y + 4.0, "3 × 4 = 12", ha="center", fontsize=9.5, color=MUTE)
arrow(X[3] - 4.2)

# 05 split by pot
panel(X[4], W[4], "Split by pot")
NPOT = [16, 16, 16, 15, 15]
for k in range(5):
    fc, ec = (BSOFT, BRICK) if k == 2 else (TSOFT, TEAL)
    by = Y + 25 - k * 4.6
    ax.add_patch(
        Rectangle(
            (X[4] + 3.0, by), W[4] - 6.0, 3.8, fc="white", ec=ec, lw=1.0, zorder=2
        )
    )
    for q in range(NPOT[k]):
        ax.add_patch(
            Rectangle(
                (X[4] + 3.8 + q * 1.45, by + 1.0),
                1.1,
                1.8,
                fc=fc,
                ec=ec,
                lw=0.5,
                zorder=3,
            )
        )
    ax.text(
        X[4] + W[4] - 3.8,
        by + 1.9,
        str(NPOT[k]),
        ha="right",
        va="center",
        fontsize=7.5,
        color=ec,
        family="monospace",
    )
ax.text(
    X[4] + W[4] / 2,
    Y + 2.0,
    "5 groups · 78 pots",
    ha="center",
    fontsize=9.5,
    color=MUTE,
)
arrow(X[4] - 4.2)

# 06 heads
panel(X[5], W[5], "Linear heads")
rows(
    X[5],
    W[5],
    ["chlorosis", "necrosis", "epinasty", "foliar distortion"],
    top=Y + 25,
    fs=9.2,
)
arrow(X[5] - 4.2)

fig.savefig(
    f"{ROOT}/reports/report_out_figures/fig8_workflow.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)
print("wrote fig8_workflow")

# ==================== fig 9 ====================
fig, ax = plt.subplots(figsize=(13.5, 5.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 152)
ax.set_ylim(8, 78)
ax.axis("off")

py = 64
ax.add_patch(Rectangle((2, py), 9, 8, fc=BSOFT, ec=BRICK, lw=1.3, zorder=2))
ax.text(
    6.5,
    py + 4,
    "one\npot",
    ha="center",
    va="center",
    fontsize=9,
    color=BRICK,
    weight="bold",
    linespacing=1.3,
)
ax.add_patch(
    FancyArrowPatch(
        (12.5, py + 4),
        (18, py + 4),
        arrowstyle="-|>",
        mutation_scale=13,
        lw=1.5,
        color=BRICK,
        zorder=4,
    )
)
for j in range(6):
    for h in range(2):
        for r in range(3):
            ax.add_patch(
                Rectangle(
                    (20 + j * 8.4 + r * 2.3, py + 5.0 - h * 4.4),
                    2.0,
                    3.6,
                    fc=BSOFT,
                    ec=BRICK,
                    lw=0.6,
                    zorder=2,
                )
            )
ax.text(
    20 + 3 * 8.4, py - 2.6, "all of its images", ha="center", fontsize=9, color=MUTE
)

ax.add_line(Line2D([0, 150], [54, 54], color=RULE, lw=1))

NPOT = [16, 16, 16, 15, 15]
TOT = sum(NPOT)
mx, mw, mh, gapy = 24, 96.0, 5.4, 1.6
my = 39
edges, acc = [], 0.0
for n in NPOT:
    w = mw * n / TOT
    edges.append((mx + acc, w))
    acc += w
for c, (ex, w) in enumerate(edges):
    ax.text(
        ex + w / 2,
        my + mh + 4.8,
        f"group {c + 1}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
    )
    ax.text(
        ex + w / 2,
        my + mh + 1.6,
        f"{NPOT[c]}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTE,
    )
for r in range(5):
    yy = my - r * (mh + gapy)
    ax.text(
        mx - 2.4,
        yy + mh / 2,
        f"round {r + 1}",
        ha="right",
        va="center",
        fontsize=9,
        color=INK,
    )
    for c, (ex, w) in enumerate(edges):
        test = c == r
        ax.add_patch(
            Rectangle(
                (ex, yy),
                w,
                mh,
                fc=BSOFT if test else TSOFT,
                ec=BRICK if test else TEAL,
                lw=1.0,
                zorder=2,
            )
        )
        ax.text(
            ex + w / 2,
            yy + mh / 2,
            "test" if test else "train",
            ha="center",
            va="center",
            fontsize=8.5,
            color=BRICK if test else TEAL,
            weight="bold" if test else "normal",
        )
    ax.text(
        mx + mw + 2.6,
        yy + mh / 2,
        f"{TOT - NPOT[r]} train · {NPOT[r]} test",
        ha="left",
        va="center",
        fontsize=8,
        color=MUTE,
        family="monospace",
    )

fig.savefig(
    f"{ROOT}/reports/report_out_figures/fig9_pot_grouped_cv.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
print("wrote fig9_pot_grouped_cv")
