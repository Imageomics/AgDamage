"""Project cropped 50 cm predictions from held-out and final symptom heads."""

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
    circle_polygons,
    normalize_panel_coordinates,
    prepare_projection_coordinates,
    symptom_wedge_polygons,
)
from agdamage.data import DATA_DIR
from agdamage.images import load_rgb
from agdamage.prediction_projection import (
    PREDICTION_SOURCES,
    THRESHOLDS,
    build_cropped_prediction_table,
)
from agdamage.rekey import file_digest, manifest_path
from agdamage.symptoms import build_50cm_symptom_cohort
from agdamage.treatment import (
    MODEL_KEYS,
    MODEL_NAMES,
    SYMPTOMS,
    load_metadata,
)

REPORT_DIR = Path("reports/cropped_50cm_prediction_projections")
COORDINATE_SOURCE = Path(
    "reports/treated_vs_untreated_logistic/projection_coordinates.csv"
)
OOF_SCORE_SOURCE = Path(
    "reports/symptom_vs_untreated_logistic/image_symptom_scores.csv"
)
CLASSIFIER_SOURCE = Path("reports/symptom_vs_untreated_logistic/final_classifiers.npz")
EMBEDDING_DIR = DATA_DIR / "embeddings" / "rekeyed"
CROPPED_IMAGE_ROOT = DATA_DIR / "Waterhemp_GreenHouse_Images_cropped"
GLYPH_RADIUS = 0.016
SOURCE_ORDER = tuple(PREDICTION_SOURCES)
THRESHOLD_ORDER = tuple(THRESHOLDS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv")
    parser.add_argument("--coordinates", type=Path, default=COORDINATE_SOURCE)
    parser.add_argument("--oof-scores", type=Path, default=OOF_SCORE_SOURCE)
    parser.add_argument("--classifiers", type=Path, default=CLASSIFIER_SOURCE)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDING_DIR)
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


def build_thumbnails(cohort: pd.DataFrame, width: int, quality: int) -> list[str]:
    print("building cropped hover thumbnails", flush=True)
    return [
        make_thumbnail(
            CROPPED_IMAGE_ROOT / row.original_relative_path,
            width,
            quality,
        )
        for row in cohort.itertuples(index=False)
    ]


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


def view_title(source: str, threshold: str) -> str:
    caution = (
        "pot held out" if source == "pot_held_out_oof" else "IN-SAMPLE / optimistic"
    )
    return (
        f"Cropped 50 cm symptom predictions: {PREDICTION_SOURCES[source]}"
        f"<br><sub>{THRESHOLDS[threshold]} · {caution} · colored quadrant = predicted positive</sub>"
    )


def _hover_customdata(panel: pd.DataFrame, threshold: str) -> list[list]:
    customdata = []
    for row in panel.itertuples(index=False):
        predicted = [
            symptom
            for symptom in SYMPTOMS
            if bool(getattr(row, f"{symptom}_predicted_{threshold}"))
        ]
        asserted = [
            symptom
            for symptom in SYMPTOMS
            if getattr(row, f"{symptom}_label_state") == "positive"
        ]
        values = [
            int(row.point_index),
            row.image_id,
            row.file_name,
            row.pot_label,
            row.capture_date,
            row.source_label,
            ", ".join(asserted) or "none recorded",
            ", ".join(predicted) or "none",
            PREDICTION_SOURCES[row.prediction_source],
            THRESHOLDS[threshold],
        ]
        for symptom in SYMPTOMS:
            values.extend(
                [
                    float(getattr(row, f"{symptom}_logit")),
                    float(getattr(row, f"{symptom}_probability")),
                    float(getattr(row, f"{symptom}_control_percentile")),
                ]
            )
        customdata.append(values)
    return customdata


def build_figure(
    cohort: pd.DataFrame,
    plot_data: pd.DataFrame,
    thumbnails: list[str],
    output_path: Path,
) -> dict:
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[MODEL_NAMES[key] for key in MODEL_KEYS] + [""] * 4,
        horizontal_spacing=0.045,
        vertical_spacing=0.10,
    )
    state_by_trace = []
    image_order = {
        image_id: index for index, image_id in enumerate(cohort["image_id_50"])
    }
    states = [
        (source, threshold) for source in SOURCE_ORDER for threshold in THRESHOLD_ORDER
    ]
    initial_state = states[0]

    for source, threshold in states:
        initial = (source, threshold) == initial_state
        for row_index, method in enumerate(METHODS, start=1):
            for column_index, model_key in enumerate(MODEL_KEYS, start=1):
                panel = plot_data[
                    plot_data["prediction_source"].eq(source)
                    & plot_data["model_key"].eq(model_key)
                    & plot_data["method"].eq(method)
                ].copy()
                panel["point_index"] = panel["image_id"].map(image_order)
                panel = panel.sort_values("point_index", ignore_index=True)
                if panel["point_index"].isna().any() or len(panel) != len(cohort):
                    raise ValueError("prediction panel does not match cohort")
                xy = normalize_panel_coordinates(panel[["x", "y"]].to_numpy())

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
                state_by_trace.append((source, threshold))

                for symptom in SYMPTOMS:
                    wedge_x, wedge_y = symptom_wedge_polygons(
                        xy,
                        panel[f"{symptom}_predicted_{threshold}"].to_numpy(bool),
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
                            name=symptom.replace("_", " ").title(),
                            legendgroup=symptom,
                            showlegend=(row_index == 1 and column_index == 1),
                            visible=initial,
                        ),
                        row=row_index,
                        col=column_index,
                    )
                    state_by_trace.append((source, threshold))

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
                state_by_trace.append((source, threshold))

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
                        customdata=_hover_customdata(panel, threshold),
                        showlegend=False,
                        visible=initial,
                        hovertemplate=(
                            "<b>predicted: %{customdata[7]}</b><br>"
                            "positive assertions: %{customdata[6]}<br>"
                            "source: %{customdata[8]}<br>"
                            "threshold: %{customdata[9]}<br>"
                            "chlorosis: z=%{customdata[10]:.3f}, p=%{customdata[11]:.3f}, ctrl pct=%{customdata[12]:.3f}<br>"
                            "necrosis: z=%{customdata[13]:.3f}, p=%{customdata[14]:.3f}, ctrl pct=%{customdata[15]:.3f}<br>"
                            "epinasty: z=%{customdata[16]:.3f}, p=%{customdata[17]:.3f}, ctrl pct=%{customdata[18]:.3f}<br>"
                            "foliar distortion: z=%{customdata[19]:.3f}, p=%{customdata[20]:.3f}, ctrl pct=%{customdata[21]:.3f}<br>"
                            "pot: %{customdata[3]} · %{customdata[4]}<br>"
                            "source label: %{customdata[5]}<br>"
                            "image: %{customdata[2]}<br>"
                            "UUID: %{customdata[1]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
                state_by_trace.append((source, threshold))

    buttons = []
    for source, threshold in states:
        buttons.append(
            {
                "label": f"{PREDICTION_SOURCES[source]} · {THRESHOLDS[threshold]}",
                "method": "update",
                "args": [
                    {
                        "visible": [
                            value == (source, threshold) for value in state_by_trace
                        ]
                    },
                    {
                        "title": {
                            "text": view_title(source, threshold),
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
        title={
            "text": view_title(*initial_state),
            "x": 0.5,
            "xanchor": "center",
        },
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
<title>Cropped 50 cm symptom prediction projections</title>
<style>
  body {{ margin: 0; padding: 18px; background: #fff; color: #111;
         font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; }}
  #note {{ max-width: 1200px; margin: 0 auto 8px; color: #475467; }}
  #peek {{ position: fixed; display: none; z-index: 10; pointer-events: none;
           width: 194px; padding: 6px; border: 1px solid #ccc; border-radius: 7px;
           background: white; box-shadow: 0 6px 24px #0003; }}
  #peek img {{ display: block; width: 180px; max-height: 200px; object-fit: contain;
               background: #eee; border-radius: 4px; }}
  #peek div {{ margin: 5px 2px 1px; font-weight: 600; }}
</style>
<div id="note">OOF views exclude the image's entire pot from training. Final
all-data views re-score the training cohort and are intentionally in-sample and
optimistic. Sigmoid values are not calibrated biological probabilities.</div>
{plot}
<div id="peek"><img alt="hovered cropped plant image"><div></div></div>
<script>
  const D = {payload};
  const graph = document.getElementById('viz');
  const peek = document.getElementById('peek');
  graph.on('plotly_hover', (event) => {{
    const data = event.points[0].customdata;
    if (!data) return;
    peek.querySelector('img').src = D.thumbnails[data[0]];
    peek.querySelector('div').textContent =
      'pot ' + data[3] + ' · ' + data[4] + ' · predicted: ' + data[7];
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
    return {"traces": len(fig.data), "selectable_views": len(states)}


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    cohort = build_50cm_symptom_cohort(metadata)
    oof_scores = pd.read_csv(args.oof_scores)
    predictions, classifier_manifest = build_cropped_prediction_table(
        cohort,
        oof_scores,
        args.embeddings,
        args.classifiers,
    )
    prediction_path = args.out / "prediction_scores.csv"
    predictions.to_csv(prediction_path, index=False)

    coordinates = prepare_projection_coordinates(cohort, pd.read_csv(args.coordinates))
    coordinates = coordinates[coordinates["representation"].eq("cropped")][
        ["image_id", "model_key", "method", "x", "y"]
    ]
    if coordinates.duplicated(["image_id", "model_key", "method"]).any():
        raise ValueError("duplicate cropped projection coordinate")
    if predictions.duplicated(["image_id", "model_key", "prediction_source"]).any():
        raise ValueError("duplicate cropped prediction row")
    plot_data = coordinates.merge(
        predictions,
        on=["image_id", "model_key"],
        validate="many_to_many",
    )
    expected_plot_rows = (
        len(cohort) * len(MODEL_KEYS) * len(METHODS) * len(PREDICTION_SOURCES)
    )
    if len(plot_data) != expected_plot_rows:
        raise ValueError(
            f"expected {expected_plot_rows} plotted rows, got {len(plot_data)}"
        )
    plot_data_path = args.out / "projection_coordinates_and_predictions.csv"
    plot_data.to_csv(plot_data_path, index=False)

    thumbnails = build_thumbnails(
        cohort,
        width=args.thumbnail_width,
        quality=args.thumbnail_quality,
    )
    figure_path = args.out / "cropped_50cm_prediction_projections.html"
    figure_details = build_figure(cohort, plot_data, thumbnails, figure_path)

    output_paths = sorted(
        path
        for path in args.out.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    prediction_counts = {}
    for source in SOURCE_ORDER:
        source_rows = predictions[predictions["prediction_source"].eq(source)]
        prediction_counts[source] = {
            model_key: {
                threshold: {
                    symptom: int(
                        source_rows.loc[
                            source_rows["model_key"].eq(model_key),
                            f"{symptom}_predicted_{threshold}",
                        ].sum()
                    )
                    for symptom in SYMPTOMS
                }
                for threshold in THRESHOLD_ORDER
            }
            for model_key in MODEL_KEYS
        }
    manifest = {
        "schema": "agdamage.cropped_50cm_prediction_projection",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort": {
            "rule": "every preferred cropped 50 cm image",
            "images": len(cohort),
            "pots": int(cohort["pot_id"].nunique()),
        },
        "prediction_sources": {
            "pot_held_out_oof": (
                "existing scores; every image scored by a model excluding its pot"
            ),
            "all_data_final_in_sample": (
                "saved final coefficients applied to their training cohort; optimistic"
            ),
        },
        "thresholds": {
            "logit0": "logit >= 0",
            "control95": (
                "logit >= the same arm/symptom's 95th percentile among OOF "
                "untreated-reference logits"
            ),
        },
        "arms": {
            "representation": "cropped",
            "models": list(MODEL_KEYS),
            "symptoms": list(SYMPTOMS),
            "methods": list(METHODS),
        },
        "prediction_counts": prediction_counts,
        "figure": figure_details,
        "final_classifier_manifest": classifier_manifest,
        "inputs": {
            "metadata": {
                "path": manifest_path(args.metadata),
                "sha256": file_digest(args.metadata),
            },
            "coordinates": {
                "path": manifest_path(args.coordinates),
                "sha256": file_digest(args.coordinates),
            },
            "oof_scores": {
                "path": manifest_path(args.oof_scores),
                "sha256": file_digest(args.oof_scores),
            },
            "classifiers": {
                "path": manifest_path(args.classifiers),
                "sha256": file_digest(args.classifiers),
            },
            "cropped_embeddings": {
                path.name: file_digest(path)
                for path in sorted(args.embeddings.glob("cropped__*.npz"))
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
    print(f"wrote cropped prediction projections to {figure_path}", flush=True)


if __name__ == "__main__":
    main()
