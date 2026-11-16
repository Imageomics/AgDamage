"""Label-structure figure: five pots down the page, timepoints across.

Column 1 is the cropped 50 cm image strip for one pot; columns 2-5 are the four
symptoms, marked only where the original folder label positively asserts them.
Everything else is unrecorded, not negative.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CROP = ROOT / "data" / "Waterhemp_GreenHouse_Images_cropped"
SY = ["chlorosis", "necrosis", "epinasty", "foliar_distortion"]
NICE = ["Chlorosis", "Necrosis", "Epinasty", "Foliar distortion"]
POTS = [
    ("109", "never treated"),
    ("209", "treated"),
    ("112", "treated"),
    ("113", "treated"),
    ("101", "treated"),
]
INK, MUTE, RULE = "#16201c", "#8b949e", "#d7ded8"
YES, UNK = "#1d5c56", "#b9c2bb"

m = pd.read_csv(f"{ROOT}/data/metadata.csv", dtype=str).fillna("")
m = m[(m.camera_height_cm == "50") & (m.is_preferred_for_height == "true")]
DATES = sorted(m.capture_date.unique())
ND = len(DATES)


def thumb(rel, px=200):
    im = Image.open(f"{CROP}/{rel}")
    im.draft("RGB", (px * 3, px * 3))
    im = im.convert("RGB")
    w, h = im.size
    k = min(w, h)
    im = im.crop(((w - k) // 2, (h - k) // 2, (w + k) // 2, (h + k) // 2))
    return np.asarray(im.resize((px, px), Image.LANCZOS))


# ---- geometry (inches), zero gap inside a block
IMG, MARK, ROW = 0.80, 0.42, 0.80
GAP, LEFT, TOP, BOT = 0.26, 1.34, 1.62, 1.02
BLOCKS = [ND * IMG] + [ND * MARK] * 4
W = LEFT + sum(BLOCKS) + GAP * 4 + 0.18
H = TOP + len(POTS) * ROW + BOT
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")
x0 = [LEFT]
for b in BLOCKS[:-1]:
    x0.append(x0[-1] + b + GAP)

n_yes = n_unk = 0
for r, (pot, arm) in enumerate(POTS):
    d = m[m.pot_label == pot].set_index("capture_date")
    ytop = H - TOP - r * ROW
    fig.text(
        (LEFT - 0.14) / W,
        (ytop - ROW / 2) / H,
        f"pot {pot}",
        ha="right",
        va="center",
        fontsize=12,
        color=INK,
        weight="bold",
    )
    fig.text(
        (LEFT - 0.14) / W,
        (ytop - ROW / 2 - 0.17) / H,
        arm,
        ha="right",
        va="center",
        fontsize=9,
        color=MUTE,
    )
    for j, date in enumerate(DATES):
        # ---- image cell
        ax = fig.add_axes([(x0[0] + j * IMG) / W, (ytop - ROW) / H, IMG / W, ROW / H])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("white")
            sp.set_linewidth(1.0)
        if date in d.index:
            ax.imshow(thumb(d.loc[date, "original_relative_path"]), aspect="auto")
        else:
            ax.set_facecolor("#f2f4f2")
            for sp in ax.spines.values():
                sp.set_color(RULE)
        # ---- symptom cells
        for c, sym in enumerate(SY):
            ax = fig.add_axes(
                [(x0[c + 1] + j * MARK) / W, (ytop - ROW) / H, MARK / W, ROW / H]
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            for sp in ax.spines.values():
                sp.set_color("#e6ebe7")
                sp.set_linewidth(0.7)
            if date not in d.index:
                ax.set_facecolor("#f2f4f2")
                continue
            if d.loc[date, sym] == "true":
                ax.text(
                    0.5,
                    0.5,
                    "✓",
                    ha="center",
                    va="center",
                    fontsize=19,
                    color=YES,
                    weight="bold",
                    transform=ax.transAxes,
                )
                n_yes += 1
            else:
                ax.text(
                    0.5,
                    0.5,
                    "?",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=UNK,
                    transform=ax.transAxes,
                )
                n_unk += 1

# ---- headers
heads = ["Cropped image  (50 cm)"] + NICE
ybar = H - TOP + 0.50
for i, (h, b) in enumerate(zip(heads, BLOCKS, strict=False)):
    fig.text(
        (x0[i] + b / 2) / W,
        (ybar + 0.10) / H,
        h,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=INK,
        weight="bold",
    )
    fig.lines.append(
        plt.Line2D(
            [x0[i] / W, (x0[i] + b) / W],
            [ybar / H, ybar / H],
            transform=fig.transFigure,
            color=RULE,
            lw=1,
        )
    )
    w = IMG if i == 0 else MARK
    for j, date in enumerate(DATES):
        fig.text(
            (x0[i] + j * w + w / 2) / W,
            (H - TOP + 0.16) / H,
            date[5:] if i == 0 else date[8:],
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=MUTE,
        )

fig.text(
    0.011,
    (H - 0.28) / H,
    "What the original labels actually assert",
    ha="left",
    va="top",
    fontsize=17,
    color=INK,
    weight="bold",
)
fig.text(
    0.011,
    (H - 0.62) / H,
    f"✓ = symptom positively asserted by the folder label       "
    f"?  = never recorded, which is not the same as absent       "
    f"{n_yes} asserted vs {n_unk} unrecorded across these {len(POTS)} pots",
    ha="left",
    va="top",
    fontsize=10.5,
    color=MUTE,
)
fig.text(
    0.011,
    (BOT - 0.16) / H,
    "Pot 209 is labelled chlorosis, then chlorosis+necrosis, then foliar distortion, then necrosis. Each label records only what prompted the folder,\n"
    "so a symptom appears to vanish between timepoints when it was simply not written down again. Pot 109 was never treated and carries no symptom\n"
    "statement at all — its four columns are unknown, not negative. This is why every classifier here is a positive-versus-reference model.",
    ha="left",
    va="top",
    fontsize=10.2,
    color=INK,
    linespacing=1.6,
)

out = f"{ROOT}/reports/report_out_figures/fig7_label_structure.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out, f"| {n_yes} asserted, {n_unk} unrecorded")
