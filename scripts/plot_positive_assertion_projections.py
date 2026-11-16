"""Render 50 cm embedding projections with four-quadrant assertion glyphs."""

import argparse
import base64
import io
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from agdamage.assertion_projection import (
    METHODS,
    SYMPTOM_COLORS,
    SYMPTOM_QUADRANTS,
    circle_polygons,
    normalize_panel_coordinates,
    prepare_projection_coordinates,
    symptom_wedge_polygons,
)
from agdamage.data import DATA_DIR
from agdamage.images import load_rgb
from agdamage.rekey import file_digest, manifest_path
from agdamage.symptoms import build_50cm_symptom_cohort
from agdamage.treatment import (
    MODEL_KEYS,
    MODEL_NAMES,
    REPRESENTATIONS,
    SYMPTOMS,
    load_metadata,
)

REPORT_DIR = Path("reports/positive_assertion_projections")
COORDINATE_SOURCE = Path(
    "reports/treated_vs_untreated_logistic/projection_coordinates.csv"
)
IMAGE_ROOTS = {
    "cropped": DATA_DIR / "Waterhemp_GreenHouse_Images_cropped",
    "segmented": DATA_DIR / "Waterhemp_GreenHouse_Images_plant_only",
}
REPRESENTATION_NAMES = {
    "full": "Full image",
    "cropped": "Cropped image",
    "segmented": "Segmented image",
}
SYMPTOM_NAMES = {
    "chlorosis": "Chlorosis",
    "necrosis": "Necrosis",
    "epinasty": "Epinasty",
    "foliar_distortion": "Foliar distortion",
}
GLYPH_RADIUS = 0.016


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv")
    parser.add_argument("--coordinates", type=Path, default=COORDINATE_SOURCE)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument("--thumbnail-width", type=int, default=180)
    parser.add_argument("--thumbnail-quality", type=int, default=65)
    return parser.parse_args()


def make_thumbnail(path: Path, width: int, quality: int) -> str:
    image, _ = load_rgb(path)
    image.draft("RGB", (width * 2, width * 2))
    ratio = width / image.width
    image = image.resize(
        (width, max(1, round(image.height * ratio))), Image.Resampling.LANCZOS
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def thumbnail_path(row, representation: str) -> Path:
    if representation == "full":
        return DATA_DIR / row.file_name
    return IMAGE_ROOTS[representation] / row.original_relative_path


def build_thumbnails(
    cohort: pd.DataFrame, width: int, quality: int
) -> dict[str, list[str]]:
    thumbnails = {}
    for representation in REPRESENTATIONS:
        print(f"building {representation} hover thumbnails", flush=True)
        thumbnails[representation] = [
            make_thumbnail(
                thumbnail_path(row, representation),
                width,
                quality,
            )
            for row in cohort.itertuples(index=False)
        ]
    return thumbnails


def _polygon_trace(
    x,
    y,
    *,
    fillcolor: str,
    linecolor: str,
    linewidth: float,
    name: str | None = None,
    legendgroup: str | None = None,
    showlegend: bool = False,
    visible: bool = True,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        fill="toself" if fillcolor != "rgba(0,0,0,0)" else None,
        fillcolor=fillcolor,
        line={"color": linecolor, "width": linewidth},
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        visible=visible,
        hoverinfo="skip",
    )


def title_for(representation: str) -> str:
    return (
        f"50 cm {REPRESENTATION_NAMES[representation].lower()}: positive symptom assertions"
        "<br><sub>Fixed quadrants: chlorosis upper-left, necrosis upper-right, "
        "epinasty lower-left, foliar distortion lower-right; white means unknown/not asserted</sub>"
    )


def build_figure(
    cohort: pd.DataFrame,
    coordinates: pd.DataFrame,
    thumbnails: dict[str, list[str]],
    output_path: Path,
) -> dict:
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[MODEL_NAMES[key] for key in MODEL_KEYS] + [""] * 4,
        horizontal_spacing=0.045,
        vertical_spacing=0.10,
    )
    trace_representations = []
    panel_ranges = {}
    image_order = {
        image_id: index for index, image_id in enumerate(cohort["image_id_50"])
    }

    for representation in REPRESENTATIONS:
        initial = representation == "full"
        for row_index, method in enumerate(METHODS, start=1):
            for column_index, model_key in enumerate(MODEL_KEYS, start=1):
                panel = coordinates[
                    coordinates["representation"].eq(representation)
                    & coordinates["model_key"].eq(model_key)
                    & coordinates["method"].eq(method)
                ].copy()
                panel["point_index"] = panel["image_id"].map(image_order)
                panel = panel.sort_values("point_index", ignore_index=True)
                if panel["point_index"].isna().any() or len(panel) != len(cohort):
                    raise ValueError("projection panel does not match cohort")
                xy = normalize_panel_coordinates(panel[["x", "y"]].to_numpy())
                panel_ranges[f"{representation}/{model_key}/{method}"] = {
                    "x": [float(xy[:, 0].min()), float(xy[:, 0].max())],
                    "y": [float(xy[:, 1].min()), float(xy[:, 1].max())],
                }

                background_x, background_y = circle_polygons(xy, radius=GLYPH_RADIUS)
                fig.add_trace(
                    _polygon_trace(
                        background_x,
                        background_y,
                        fillcolor="#FFFFFF",
                        linecolor="rgba(0,0,0,0)",
                        linewidth=0,
                        visible=initial,
                    ),
                    row=row_index,
                    col=column_index,
                )
                trace_representations.append(representation)

                for symptom in SYMPTOMS:
                    wedge_x, wedge_y = symptom_wedge_polygons(
                        xy,
                        panel[symptom].to_numpy(bool),
                        symptom,
                        radius=GLYPH_RADIUS,
                    )
                    fig.add_trace(
                        _polygon_trace(
                            wedge_x,
                            wedge_y,
                            fillcolor=SYMPTOM_COLORS[symptom],
                            linecolor="#FFFFFF",
                            linewidth=0.35,
                            name=SYMPTOM_NAMES[symptom],
                            legendgroup=symptom,
                            showlegend=(row_index == 1 and column_index == 1),
                            visible=initial,
                        ),
                        row=row_index,
                        col=column_index,
                    )
                    trace_representations.append(representation)

                outline_x, outline_y = circle_polygons(xy, radius=GLYPH_RADIUS)
                fig.add_trace(
                    _polygon_trace(
                        outline_x,
                        outline_y,
                        fillcolor="rgba(0,0,0,0)",
                        linecolor="#667085",
                        linewidth=0.55,
                        visible=initial,
                    ),
                    row=row_index,
                    col=column_index,
                )
                trace_representations.append(representation)

                customdata = np.column_stack(
                    [
                        panel["point_index"].astype(int),
                        panel["representation"],
                        panel["image_id"],
                        panel["file_name"],
                        panel["pot_label"],
                        panel["capture_date"],
                        panel["source_label"],
                        panel["positive_assertions"],
                        panel["observation_id"],
                        panel["capture_role"],
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=xy[:, 0],
                        y=xy[:, 1],
                        mode="markers",
                        marker={
                            "size": 12,
                            "color": "rgba(0,0,0,0.004)",
                            "line": {"width": 0},
                        },
                        customdata=customdata,
                        showlegend=False,
                        visible=initial,
                        hovertemplate=(
                            "<b>positive assertions: %{customdata[7]}</b><br>"
                            "source label: %{customdata[6]}<br>"
                            "pot: %{customdata[4]}<br>"
                            "date: %{customdata[5]}<br>"
                            "representation: %{customdata[1]}<br>"
                            "capture role: %{customdata[9]}<br>"
                            "image: %{customdata[3]}<br>"
                            "UUID: %{customdata[2]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
                trace_representations.append(representation)

    buttons = []
    for representation in REPRESENTATIONS:
        buttons.append(
            {
                "label": REPRESENTATION_NAMES[representation],
                "method": "update",
                "args": [
                    {
                        "visible": [
                            value == representation for value in trace_representations
                        ]
                    },
                    {
                        "title": {
                            "text": title_for(representation),
                            "x": 0.5,
                            "xanchor": "center",
                        }
                    },
                ],
            }
        )

    for row_index, method in enumerate(METHODS, start=1):
        fig.update_yaxes(title_text=method, row=row_index, col=1)
        for column_index in range(1, 5):
            axis_number = (row_index - 1) * 4 + column_index
            x_reference = "x" if axis_number == 1 else f"x{axis_number}"
            fig.update_xaxes(
                range=[-0.56, 0.56],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=row_index,
                col=column_index,
            )
            fig.update_yaxes(
                range=[-0.56, 0.56],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                scaleanchor=x_reference,
                scaleratio=1,
                row=row_index,
                col=column_index,
            )
    fig.update_layout(
        template="plotly_white",
        title={"text": title_for("full"), "x": 0.5, "xanchor": "center"},
        width=1680,
        height=940,
        margin={"l": 65, "r": 25, "t": 135, "b": 78},
        legend={"orientation": "h", "y": -0.08, "x": 0},
        hovermode="closest",
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "active": 0,
                "x": 0,
                "xanchor": "left",
                "y": 1.13,
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
    )

    plot = fig.to_html(full_html=False, include_plotlyjs="inline", div_id="viz")
    payload = json.dumps({"thumbnails": thumbnails})
    output_path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>50 cm positive-assertion embedding projections</title>
<style>
  body {{ margin: 0; padding: 18px; background: #fff; color: #111;
         font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; }}
  #note {{ max-width: 1100px; margin: 0 auto 8px; color: #475467; }}
  #peek {{ position: fixed; display: none; z-index: 10; pointer-events: none;
           width: 194px; padding: 6px; border: 1px solid #ccc; border-radius: 7px;
           background: white; box-shadow: 0 6px 24px #0003; }}
  #peek img {{ display: block; width: 180px; max-height: 200px; object-fit: contain;
               background: #eee; border-radius: 4px; }}
  #peek div {{ margin: 5px 2px 1px; font-weight: 600; }}
</style>
<div id="note">Colored quadrants are positive assertions only. White quadrants are
unknown/not asserted and must not be interpreted as symptom absence.</div>
{plot}
<div id="peek"><img alt="hovered plant image"><div></div></div>
<script>
  const D = {payload};
  const graph = document.getElementById('viz');
  const peek = document.getElementById('peek');
  graph.on('plotly_hover', (event) => {{
    const data = event.points[0].customdata;
    if (!data) return;
    peek.querySelector('img').src = D.thumbnails[data[1]][data[0]];
    peek.querySelector('div').textContent =
      'pot ' + data[4] + ' · ' + data[5] + ' · ' + data[7];
    peek.style.display = 'block';
    const e = event.event, pad = 14, width = 208, height = 245;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + width > innerWidth) x = e.clientX - width - pad;
    if (y + height > innerHeight) y = Math.max(pad, e.clientY - height - pad);
    peek.style.left = x + 'px'; peek.style.top = y + 'px';
  }});
  graph.on('plotly_unhover', () => {{ peek.style.display = 'none'; }});
</script>
""",
        encoding="utf-8",
    )
    return {"traces": len(fig.data), "panel_display_ranges": panel_ranges}


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    cohort = build_50cm_symptom_cohort(metadata)
    source_coordinates = pd.read_csv(args.coordinates)
    coordinates = prepare_projection_coordinates(cohort, source_coordinates)
    coordinate_path = args.out / "projection_coordinates.csv"
    coordinates.to_csv(coordinate_path, index=False)

    thumbnails = build_thumbnails(
        cohort,
        width=args.thumbnail_width,
        quality=args.thumbnail_quality,
    )
    figure_path = args.out / "positive_assertion_projections_50cm.html"
    figure_details = build_figure(cohort, coordinates, thumbnails, figure_path)

    output_paths = sorted(
        path
        for path in args.out.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    assertion_patterns = (
        cohort[list(SYMPTOMS)]
        .eq("true")
        .value_counts(sort=False)
        .rename("images")
        .reset_index()
    )
    manifest = {
        "schema": "agdamage.positive_assertion_embedding_projection",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort": {
            "rule": "every preferred 50 cm image",
            "images": len(cohort),
            "pots": int(cohort["pot_id"].nunique()),
            "positive_assertion_patterns": assertion_patterns.to_dict("records"),
        },
        "panels": {
            "representations": list(REPRESENTATIONS),
            "models": list(MODEL_KEYS),
            "methods": list(METHODS),
            "coordinate_normalization": (
                "center by panel bounding-box midpoint; divide both axes by the "
                "larger panel span"
            ),
        },
        "glyph": {
            "semantics": "colored quadrant = asserted positive; white = unknown",
            "radius_in_normalized_coordinates": GLYPH_RADIUS,
            "quadrant_angles_degrees": {
                symptom: list(angles) for symptom, angles in SYMPTOM_QUADRANTS.items()
            },
            "colors": SYMPTOM_COLORS,
        },
        "figure": figure_details,
        "thumbnails": {
            "width_px": args.thumbnail_width,
            "jpeg_quality": args.thumbnail_quality,
            "representations": list(REPRESENTATIONS),
        },
        "inputs": {
            "metadata": {
                "path": manifest_path(args.metadata),
                "sha256": file_digest(args.metadata),
            },
            "coordinate_source": {
                "path": manifest_path(args.coordinates),
                "sha256": file_digest(args.coordinates),
            },
        },
        "outputs": {path.name: file_digest(path) for path in output_paths},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "plotly": plotly.__version__,
        },
    }
    (args.out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote positive-assertion projection to {figure_path}", flush=True)


if __name__ == "__main__":
    main()
