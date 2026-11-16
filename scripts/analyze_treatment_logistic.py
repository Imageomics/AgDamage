"""Matched 30/50 cm treated-vs-untreated projections and logistic probes."""

import argparse
import base64
import io
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from PIL import Image
from plotly.subplots import make_subplots

from agdamage.data import DATA_DIR
from agdamage.images import load_rgb
from agdamage.rekey import file_digest, manifest_path
from agdamage.treatment import (
    MODEL_KEYS,
    MODEL_NAMES,
    REPRESENTATIONS,
    arm_features,
    build_50cm_cohort,
    build_longitudinal_evaluation_plan,
    evaluate_logistic_trials,
    load_metadata,
    load_rekeyed_bundle,
    project,
    validate_evaluation_plan,
)

REPORT_DIR = Path("reports/treated_vs_untreated_logistic")
IMAGE_ROOTS = {
    "full": DATA_DIR / "Waterhemp_GreenHouse_Images",
    "cropped": DATA_DIR / "Waterhemp_GreenHouse_Images_cropped",
    "segmented": DATA_DIR / "Waterhemp_GreenHouse_Images_plant_only",
}
MODEL_COLORS = {
    "bioclip": "#0072B2",
    "bioclip-2": "#009E73",
    "bioclip-2.5": "#E69F00",
    "dinov3-vit7b16": "#CC79A7",
}
CLASS_STYLES = {
    False: ("untreated", "#0072B2"),
    True: ("treated", "#D55E00"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv")
    parser.add_argument(
        "--embeddings", type=Path, default=DATA_DIR / "embeddings" / "rekeyed"
    )
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 40, 60, 100, 140],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-projections", action="store_true")
    return parser.parse_args()


def make_thumbnail(path: Path, width: int = 224, quality: int = 72) -> str:
    image, _ = load_rgb(path)
    image.draft("RGB", (width * 2, width * 2))
    ratio = width / image.width
    image = image.resize(
        (width, max(1, round(image.height * ratio))), Image.Resampling.LANCZOS
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def thumbnails_for_arm(
    cohort: pd.DataFrame,
    height_cm: int,
    representation: str,
    cache: dict[tuple[str, str], str],
) -> list[str]:
    values = []
    root = IMAGE_ROOTS[representation]
    for relative_path in cohort[f"original_relative_path_{height_cm}"]:
        key = (representation, relative_path)
        if key not in cache:
            cache[key] = make_thumbnail(root / relative_path)
        values.append(cache[key])
    return values


def projection_figure(
    cohort: pd.DataFrame,
    height_cm: int,
    representation: str,
    coordinates: dict[str, dict[str, np.ndarray]],
    thumbnails: list[str],
    output_path: Path,
) -> None:
    methods = ("t-SNE", "UMAP")
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[MODEL_NAMES[key] for key in MODEL_KEYS] + [""] * 4,
        horizontal_spacing=0.045,
        vertical_spacing=0.10,
    )
    treated = cohort["herbicide_treated"].to_numpy(bool)
    point_index = np.arange(len(cohort))
    for row_index, method in enumerate(methods, start=1):
        for column_index, model_key in enumerate(MODEL_KEYS, start=1):
            xy = coordinates[model_key][method]
            for class_value, (class_name, color) in CLASS_STYLES.items():
                mask = treated == class_value
                customdata = np.column_stack(
                    [
                        point_index[mask],
                        cohort.loc[mask, "image_id_" + str(height_cm)],
                        cohort.loc[mask, "pot_label"],
                        cohort.loc[mask, "capture_date"],
                        cohort.loc[mask, "source_label_" + str(height_cm)],
                        cohort.loc[mask, "positive_symptoms_" + str(height_cm)],
                        cohort.loc[mask, "file_name_" + str(height_cm)],
                        cohort.loc[mask, "observation_id"],
                        cohort.loc[mask, "capture_role_" + str(height_cm)],
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=xy[mask, 0],
                        y=xy[mask, 1],
                        mode="markers",
                        name=class_name,
                        legendgroup=class_name,
                        showlegend=(row_index == 1 and column_index == 1),
                        marker={
                            "color": color,
                            "size": 7,
                            "opacity": 0.85,
                            "line": {"color": "white", "width": 0.6},
                        },
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[4]}</b><br>"
                            "pot: %{customdata[2]}<br>"
                            "date: %{customdata[3]}<br>"
                            f"height: {height_cm} cm<br>"
                            "positive symptoms: %{customdata[5]}<br>"
                            "capture role: %{customdata[8]}<br>"
                            "image: %{customdata[6]}<br>"
                            "UUID: %{customdata[1]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
    for row_index, method in enumerate(methods, start=1):
        fig.update_yaxes(title_text=method, row=row_index, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"{height_cm} cm {representation} images: treated vs untreated"
                "<br><sub>Matched pot×date cohort; cosine t-SNE and UMAP; hover for metadata</sub>"
            )
        },
        width=1680,
        height=900,
        margin={"l": 65, "r": 25, "t": 110, "b": 70},
        legend={"orientation": "h", "y": -0.08, "x": 0},
        hovermode="closest",
    )

    plot = fig.to_html(full_html=False, include_plotlyjs="inline", div_id="viz")
    payload = json.dumps({"thumbnails": thumbnails})
    output_path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>{height_cm} cm {representation} treatment projections</title>
<style>
  body {{ margin: 0; padding: 18px; background: #fff; color: #111;
         font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; }}
  #peek {{ position: fixed; display: none; z-index: 10; pointer-events: none;
           width: 236px; padding: 6px; border: 1px solid #ccc; border-radius: 7px;
           background: white; box-shadow: 0 6px 24px #0003; }}
  #peek img {{ display: block; width: 224px; max-height: 224px; object-fit: contain;
               background: #eee; border-radius: 4px; }}
  #peek div {{ margin: 5px 2px 1px; font-weight: 600; }}
</style>
{plot}
<div id="peek"><img alt="hovered plant image"><div></div></div>
<script>
  const D = {payload};
  const graph = document.getElementById('viz');
  const peek = document.getElementById('peek');
  graph.on('plotly_hover', (event) => {{
    const data = event.points[0].customdata;
    if (!data) return;
    peek.querySelector('img').src = D.thumbnails[data[0]];
    peek.querySelector('div').textContent = data[4] + ' · pot ' + data[2] + ' · ' + data[3];
    peek.style.display = 'block';
    const e = event.event, pad = 14, width = 250, height = 275;
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


def write_projections(
    cohort: pd.DataFrame,
    embedding_dir: Path,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    coordinate_rows = []
    thumbnail_cache = {}
    for height_cm in (50,):
        for representation, bundle_prefix in REPRESENTATIONS.items():
            coordinates = {}
            for model_key in MODEL_KEYS:
                print(
                    f"projecting {height_cm} cm / {representation} / {MODEL_NAMES[model_key]}",
                    flush=True,
                )
                bundle = load_rekeyed_bundle(
                    embedding_dir / f"{bundle_prefix}__{model_key}.npz"
                )
                features = arm_features(cohort, height_cm, bundle)
                coordinates[model_key] = project(features, seed)
                for method, xy in coordinates[model_key].items():
                    for index, cohort_row in enumerate(cohort.itertuples(index=False)):
                        coordinate_rows.append(
                            {
                                "height_cm": height_cm,
                                "representation": representation,
                                "model": MODEL_NAMES[model_key],
                                "model_key": model_key,
                                "method": method,
                                "observation_id": cohort_row.observation_id,
                                "image_id": getattr(
                                    cohort_row, f"image_id_{height_cm}"
                                ),
                                "pot_id": cohort_row.pot_id,
                                "capture_date": cohort_row.capture_date,
                                "herbicide_treated": bool(cohort_row.herbicide_treated),
                                "x": float(xy[index, 0]),
                                "y": float(xy[index, 1]),
                            }
                        )
            thumbs = thumbnails_for_arm(
                cohort, height_cm, representation, thumbnail_cache
            )
            projection_figure(
                cohort,
                height_cm,
                representation,
                coordinates,
                thumbs,
                output_dir / f"projections_{height_cm}cm_{representation}.html",
            )
    return pd.DataFrame(coordinate_rows)


def performance_figure(trials: pd.DataFrame, output_path: Path) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "agdamage-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for column_index, representation in enumerate(REPRESENTATIONS):
        axis = axes[column_index]
        subset = trials[trials["representation"] == representation]
        for model_key in MODEL_KEYS:
            model_rows = subset[subset["model_key"] == model_key]
            # Pool predictions conceptually across held-out pots by showing the
            # repeat mean over folds; intervals describe support-draw variation,
            # not the quantized performance of a single held-out control pot.
            repeat_scores = (
                model_rows.groupby(["support_images_per_class", "repeat"])[
                    "balanced_accuracy"
                ]
                .mean()
                .reset_index()
            )
            grouped = repeat_scores.groupby("support_images_per_class")[
                "balanced_accuracy"
            ]
            summary = grouped.agg(
                mean="mean",
                low=lambda values: values.quantile(0.10),
                high=lambda values: values.quantile(0.90),
            ).reset_index()
            x = summary["support_images_per_class"].to_numpy()
            axis.plot(
                x,
                summary["mean"],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[model_key],
                label=MODEL_NAMES[model_key],
            )
            axis.fill_between(
                x,
                summary["low"],
                summary["high"],
                color=MODEL_COLORS[model_key],
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(0.5, color="#777", linestyle="--", linewidth=1)
        axis.set_title(f"50 cm · {representation}")
        axis.set_xscale("symlog", linthresh=2)
        axis.set_xticks(sorted(trials["support_images_per_class"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(alpha=0.2)
        axis.set_xlabel("Support images per class")
        if column_index == 0:
            axis.set_ylabel("Balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "50 cm treated-vs-untreated logistic regression by representation",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def package_versions() -> dict:
    cache_dir = Path(tempfile.gettempdir()) / "agdamage-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    import plotly
    import umap

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "plotly": plotly.__version__,
        "umap_learn": umap.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    cohort = build_50cm_cohort(metadata)
    shots = tuple(sorted(set(args.shots)))
    plan = build_longitudinal_evaluation_plan(cohort, shots, args.repeats, args.seed)
    validate_evaluation_plan(cohort, plan)
    plan_path = args.out / "evaluation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"50 cm cohort: {len(cohort)} observations, {cohort['pot_id'].nunique()} pots; "
        f"treated={int(cohort['herbicide_treated'].sum())}, "
        f"untreated={int((~cohort['herbicide_treated']).sum())}",
        flush=True,
    )

    trials_path = args.out / "classifier_trials.csv"
    performance_path = args.out / "classifier_performance_vs_support.png"
    coordinates_path = args.out / "projection_coordinates.csv"
    if args.skip_classifier:
        if not trials_path.exists():
            raise SystemExit(f"cannot skip classifier; missing {trials_path}")
        trials = pd.read_csv(trials_path)
    else:
        trials = evaluate_logistic_trials(
            cohort, plan, args.embeddings, c_value=args.C, heights=(50,)
        )
        trials.to_csv(trials_path, index=False)
        performance_figure(trials, performance_path)

    if not args.skip_projections:
        coordinates = write_projections(cohort, args.embeddings, args.out, args.seed)
        coordinates.to_csv(coordinates_path, index=False)

    bundle_paths = sorted(args.embeddings.glob("*.npz"))
    outputs = sorted(
        path
        for path in args.out.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema": "agdamage.treated_vs_untreated_logistic_analysis",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "target": {
            "column": "herbicide_treated",
            "false": "source_label == healthy_untreated",
            "true": "source_label is a positive symptom label",
            "unit": "image at capture time",
        },
        "cohort": {
            "rule": "every preferred 50 cm image across all available dates",
            "observations": len(cohort),
            "pots": int(cohort["pot_id"].nunique()),
            "treated_observations": int(cohort["herbicide_treated"].sum()),
            "untreated_observations": int((~cohort["herbicide_treated"]).sum()),
            "dates": sorted(cohort["capture_date"].unique().tolist()),
        },
        "arms": {
            "heights_cm": [50],
            "representations": list(REPRESENTATIONS),
            "models": [MODEL_NAMES[key] for key in MODEL_KEYS],
        },
        "classifier": {
            "type": "sklearn.linear_model.LogisticRegression",
            "penalty": "L2",
            "C": args.C,
            "solver": "liblinear",
            "support_images_per_class": list(shots),
            "repeats": args.repeats,
            "folds": plan["n_folds"],
            "primary_metric": "balanced_accuracy",
        },
        "projections": {
            "methods": ["t-SNE", "UMAP"],
            "metric": "cosine",
            "seed": args.seed,
            "tsne_perplexity": 30,
            "umap_neighbors": 15,
            "umap_min_dist": 0.1,
        },
        "inputs": {
            "metadata": {
                "path": manifest_path(args.metadata),
                "sha256": file_digest(args.metadata),
            },
            "embedding_bundles": {
                path.name: file_digest(path) for path in bundle_paths
            },
        },
        "outputs": {path.name: file_digest(path) for path in outputs},
        "software": package_versions(),
    }
    manifest_output_path = args.out / "analysis_manifest.json"
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote analysis to {args.out}", flush=True)


if __name__ == "__main__":
    main()
