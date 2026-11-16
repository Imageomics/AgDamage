"""t-SNE and UMAP projections of the ag embeddings, one column per model."""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from agdamage.data import DATA_DIR, WATERHEMP_GREENHOUSE
from agdamage.embed import MODEL_SPECS, load_bundle

# Six-slot categorical palette built for this chart: scatter puts every pair on
# screen at once, so it is validated on the all-pairs pairlist in both modes
# (light CVD dE 8.7 / normal 17.8; dark 8.4 / 16.6) rather than the adjacent one.
# Both modes clear the >=8 CVD target, so hue alone carries class identity.
PALETTE_LIGHT = ["#783577", "#ae370f", "#c68612", "#0d6f47", "#2776bf", "#ac7cef"]
PALETTE_DARK = ["#8a4688", "#a62f03", "#c5850e", "#53a77d", "#2273bb", "#a776e9"]

SURFACE_LIGHT, SURFACE_DARK = "#fcfcfb", "#1a1a19"
INK_LIGHT, INK_DARK = "#0b0b0b", "#ffffff"
MUTED_LIGHT, MUTED_DARK = "#52514e", "#c3c2b7"
GRID_LIGHT, GRID_DARK = "#e8e7e3", "#2f2f2d"

METHODS = ["t-SNE", "UMAP"]

# Panes are forced square: pick a width, then solve the height that makes each
# subplot box exactly as tall as it is wide (see square_height).
FIG_WIDTH = 1680
MARGIN = {"l": 70, "r": 20, "t": 110, "b": 70}
H_SPACING = 0.05
V_SPACING = 0.10


# Plotly's subplot-title annotations eat a few vertical pixels that the domain
# arithmetic below cannot see; measured against the rendered boxes.
TITLE_RESERVE = 9


def square_height(n_cols: int, n_rows: int) -> int:
    """Figure height that makes every subplot box square at FIG_WIDTH."""
    plot_width = FIG_WIDTH - MARGIN["l"] - MARGIN["r"]
    col_fraction = (1 - H_SPACING * (n_cols - 1)) / n_cols
    row_fraction = (1 - V_SPACING * (n_rows - 1)) / n_rows
    pane = plot_width * col_fraction
    return round(pane / row_fraction + MARGIN["t"] + MARGIN["b"] + TITLE_RESERVE)


def project(features: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """t-SNE and UMAP in one pass. Cosine metric: the features are L2-normalized."""
    import umap

    print("    t-SNE...", flush=True)
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=30,
        max_iter=1000,
        random_state=seed,
    ).fit_transform(features)

    print("    UMAP...", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
    )
    return {"t-SNE": tsne, "UMAP": reducer.fit_transform(features)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DATA_DIR / "embeddings")
    parser.add_argument("--dataset", default="Waterhemp_GreenHouse_Images")
    parser.add_argument("--thumbnails", type=Path, default=DATA_DIR / "thumbnails.json")
    parser.add_argument(
        "--out", type=Path, default=Path("reports/embedding_projections.html")
    )
    parser.add_argument(
        "--title",
        help="Figure title; defaults to a readable form of --dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-id", default=WATERHEMP_GREENHOUSE)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    keys = list(MODEL_SPECS)
    bundles = {
        k: load_bundle(args.embeddings / f"{args.dataset}__{k}.npz") for k in keys
    }

    labels = bundles[keys[0]]["labels"]
    ids = [str(i) for i in bundles[keys[0]]["ids"]]
    for key in keys[1:]:
        if not np.array_equal(bundles[key]["ids"], bundles[keys[0]]["ids"]):
            raise SystemExit(f"{key} image IDs or order do not match {keys[0]}")
        if not np.array_equal(bundles[key]["labels"], labels):
            raise SystemExit(f"{key} labels do not match {keys[0]}")
    classes = sorted(set(labels.tolist()))
    if len(classes) > len(PALETTE_LIGHT):
        raise SystemExit(
            f"{len(classes)} classes exceeds the validated {len(PALETTE_LIGHT)}-slot palette"
        )

    # Inline thumbnails keep hover instant; fall back to the Hub if absent.
    if args.thumbnails.exists():
        thumbs = json.loads(args.thumbnails.read_text())
        missing = [i for i in ids if i not in thumbs]
        if missing:
            raise SystemExit(
                f"thumbnail cache missing {len(missing)} entries; rerun make_thumbnails.py"
            )
    else:
        base = (
            f"https://huggingface.co/datasets/{args.repo_id}/resolve/{args.revision}/"
        )
        thumbs = {i: base + quote(i) for i in ids}
        print(f"no thumbnail cache at {args.thumbnails}; falling back to remote URLs")

    projections = {}
    for key in keys:
        print(f"projecting {MODEL_SPECS[key]['name']}...", flush=True)
        projections[key] = project(bundles[key]["features"], args.seed)

    fig = make_subplots(
        rows=len(METHODS),
        cols=len(keys),
        # Model names on the top row only; the row is named by its y-axis title.
        subplot_titles=[MODEL_SPECS[k]["name"] for k in keys] + [""] * len(keys),
        horizontal_spacing=H_SPACING,
        vertical_spacing=V_SPACING,
    )

    index = np.arange(len(ids))
    for row, method in enumerate(METHODS, start=1):
        for col, key in enumerate(keys, start=1):
            coords = projections[key][method]
            for slot, class_name in enumerate(classes):
                mask = labels == class_name
                fig.add_trace(
                    go.Scatter(
                        x=coords[mask, 0],
                        y=coords[mask, 1],
                        mode="markers",
                        name=class_name,
                        legendgroup=class_name,
                        showlegend=(row == 1 and col == 1),
                        marker={
                            "color": PALETTE_LIGHT[slot],
                            "size": 7,
                            "line": {"width": 1, "color": SURFACE_LIGHT},
                            "opacity": 0.9,
                        },
                        customdata=np.stack(
                            [
                                np.array(ids, dtype=object)[mask],
                                np.repeat(class_name, mask.sum()),
                                index[mask].astype(object),
                            ],
                            axis=1,
                        ),
                        hoverinfo="none",
                    ),
                    row=row,
                    col=col,
                )

    # One empty overlay per panel; hovering fills them with the matching point
    # so the same image is marked in every panel at once.
    highlight_start = len(fig.data)
    panel_coords = []
    for row, method in enumerate(METHODS, start=1):
        for col, key in enumerate(keys, start=1):
            panel_coords.append(projections[key][method].tolist())
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    showlegend=False,
                    hoverinfo="skip",
                    marker={
                        "size": 16,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 2.5, "color": INK_LIGHT},
                    },
                ),
                row=row,
                col=col,
            )

    for row, method in enumerate(METHODS, start=1):
        fig.update_yaxes(title_text=method, row=row, col=1)

    axis_style = {
        "showticklabels": False,
        "showgrid": False,
        "zeroline": False,
        "ticks": "",
        "showline": True,
        "mirror": True,
        "linewidth": 1,
        "linecolor": GRID_LIGHT,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # Square boxes alone would still stretch the data; anchoring y to its own x
    # at ratio 1 keeps one data unit the same size on both axes, so distances
    # in the projection read honestly.
    for row in range(1, len(METHODS) + 1):
        for col in range(1, len(keys) + 1):
            n = (row - 1) * len(keys) + col
            fig.update_yaxes(
                scaleanchor=f"x{'' if n == 1 else n}", scaleratio=1, row=row, col=col
            )

    title = args.title or f"Embeddings of {args.dataset.replace('_', ' ')}"
    fig.update_layout(
        title={
            "text": f"{title} ({len(ids):,} images)"
            f"<br><sub>t-SNE and UMAP using cosine distance between L2-normalized"
            f" embeddings, colored by damage class</sub>"
        },
        width=FIG_WIDTH,
        height=square_height(len(keys), len(METHODS)),
        autosize=False,
        margin=MARGIN,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.09,
            "xanchor": "left",
            "x": 0,
            "title": "",
        },
        hovermode="closest",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_html(
        fig,
        args.out,
        labels,
        classes,
        ids,
        thumbs,
        panel_coords,
        list(range(highlight_start, len(fig.data))),
    )
    print(f"wrote {args.out}")


def write_html(fig, out, labels, classes, ids, thumbs, panel_coords, highlight_traces):
    """Emit the figure with a theme-aware surface and a class-count table.

    The table is the relief the palette's sub-3:1 contrast warning obligates:
    class identity stays readable without relying on the marker color.
    """
    counts = {c: int((labels == c).sum()) for c in classes}

    def swatch(i):
        return (
            f'<span class="sw" style="background:{PALETTE_LIGHT[i]}" '
            f'data-light="{PALETTE_LIGHT[i]}" data-dark="{PALETTE_DARK[i]}"></span>'
        )

    rows = "".join(
        f"<tr><td>{swatch(i)} {c}</td><td>{counts[c]:,}</td>"
        f"<td>{counts[c] / len(labels) * 100:.1f}%</td></tr>"
        for i, c in enumerate(classes)
    )

    plot = fig.to_html(full_html=False, include_plotlyjs="inline", div_id="viz")
    data = json.dumps(
        {
            "thumbs": [thumbs[i] for i in ids],
            "ids": ids,
            "coords": panel_coords,
            "highlight": highlight_traces,
            "light": PALETTE_LIGHT,
            "dark": PALETTE_DARK,
        }
    )

    out.write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>AgDamage embedding projections</title>
<style>
  :root {{
    color-scheme: light;
    --surface: {SURFACE_LIGHT}; --ink: {INK_LIGHT};
    --muted: {MUTED_LIGHT}; --grid: {GRID_LIGHT};
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface: {SURFACE_DARK}; --ink: {INK_DARK};
      --muted: {MUTED_DARK}; --grid: {GRID_DARK};
    }}
  }}
  body {{ background: var(--surface); color: var(--ink);
         font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 24px; }}
  table {{ border-collapse: collapse; margin: 8px 0 32px; font-variant-numeric: tabular-nums; }}
  caption {{ text-align: left; color: var(--muted); padding-bottom: 6px; }}
  th, td {{ text-align: left; padding: 4px 20px 4px 0; border-bottom: 1px solid var(--grid); }}
  th {{ color: var(--muted); font-weight: 500; }}
  td:nth-child(n+2) {{ text-align: right; }}
  .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
         margin-right: 8px; vertical-align: baseline; }}
  #peek {{ position: fixed; z-index: 9; pointer-events: none; display: none;
           background: var(--surface); border: 1px solid var(--grid);
           border-radius: 6px; padding: 6px; box-shadow: 0 6px 24px #0003; width: 236px; }}
  #peek img {{ display: block; width: 224px; height: 169px; object-fit: cover;
               border-radius: 3px; background: var(--grid); }}
  #peek .cls {{ font-weight: 600; margin: 6px 2px 1px; }}
  #peek .id {{ color: var(--muted); font-size: 11px; margin: 0 2px 2px;
               word-break: break-all; line-height: 1.35; }}
</style>
{plot}
<div id="peek"><img alt=""><div class="cls"></div><div class="id"></div></div>
<table>
  <caption>Images per damage class</caption>
  <tr><th>class</th><th>n</th><th>share</th></tr>
  {rows}
</table>
<script>
  const D = {data};
  const gd = document.getElementById('viz');
  const peek = document.getElementById('peek');
  const peekImg = peek.querySelector('img');

  function applyTheme() {{
    const dark = matchMedia('(prefers-color-scheme: dark)').matches
                 && document.documentElement.dataset.theme !== 'light';
    const pal = dark ? D.dark : D.light;
    const css = getComputedStyle(document.documentElement);
    const surface = css.getPropertyValue('--surface').trim();
    const ink = css.getPropertyValue('--ink').trim();
    const muted = css.getPropertyValue('--muted').trim();
    const grid = css.getPropertyValue('--grid').trim();

    // Class traces run class-0..class-5 per panel, so position modulo the
    // palette length recovers each trace's class slot.
    const classTraces = [], colors = [];
    for (let i = 0; i < gd.data.length; i++) {{
      if (D.highlight.includes(i)) continue;
      colors.push(pal[classTraces.length % pal.length]);
      classTraces.push(i);
    }}
    Plotly.restyle(gd, {{'marker.color': colors,
                        'marker.line.color': colors.map(() => surface)}}, classTraces);
    Plotly.restyle(gd, {{'marker.line.color': ink}}, D.highlight);
    Plotly.relayout(gd, {{
      'paper_bgcolor': surface, 'plot_bgcolor': surface,
      'font.color': ink, 'title.font.color': ink, 'legend.font.color': ink,
    }});
    for (const key of Object.keys(gd.layout)) {{
      if (/^[xy]axis\\d*$/.test(key)) {{
        Plotly.relayout(gd, {{[key + '.gridcolor']: grid,
                             [key + '.title.font.color']: muted,
                             [key + '.linecolor']: grid}});
      }}
    }}
    for (const el of document.querySelectorAll('.sw')) {{
      el.style.background = dark ? el.dataset.dark : el.dataset.light;
    }}
  }}

  // Mark the same image in every panel; show its thumbnail only under the cursor.
  let shown = -1;
  gd.on('plotly_hover', (ev) => {{
    const pt = ev.points[0];
    if (!pt || !pt.customdata) return;
    const [id, cls, idx] = pt.customdata;
    if (idx !== shown) {{
      shown = idx;
      Plotly.restyle(gd, {{
        x: D.coords.map(p => [p[idx][0]]),
        y: D.coords.map(p => [p[idx][1]]),
      }}, D.highlight);
      peekImg.src = D.thumbs[idx];
      peek.querySelector('.cls').textContent = cls;
      peek.querySelector('.id').textContent = id;
    }}
    peek.style.display = 'block';
    const e = ev.event, pad = 14, w = 248, h = 250;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > innerWidth) x = e.clientX - w - pad;
    if (y + h > innerHeight) y = Math.max(pad, e.clientY - h - pad);
    peek.style.left = x + 'px';
    peek.style.top = y + 'px';
  }});

  gd.on('plotly_unhover', () => {{
    peek.style.display = 'none';
    shown = -1;
    Plotly.restyle(gd, {{x: D.highlight.map(() => []),
                        y: D.highlight.map(() => [])}}, D.highlight);
  }});

  applyTheme();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
</script>
""")


if __name__ == "__main__":
    main()
