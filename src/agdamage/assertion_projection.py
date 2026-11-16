"""Data and geometry helpers for partial-label embedding projections."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agdamage.treatment import MODEL_KEYS, REPRESENTATIONS, SYMPTOMS

METHODS = ("t-SNE", "UMAP")
SYMPTOM_COLORS = {
    "chlorosis": "#E69F00",
    "necrosis": "#D55E00",
    "epinasty": "#0072B2",
    "foliar_distortion": "#009E73",
}
SYMPTOM_QUADRANTS = {
    "chlorosis": (90.0, 180.0),
    "necrosis": (0.0, 90.0),
    "epinasty": (180.0, 270.0),
    "foliar_distortion": (270.0, 360.0),
}


def prepare_projection_coordinates(
    cohort: pd.DataFrame, coordinates: pd.DataFrame
) -> pd.DataFrame:
    """Validate and enrich label-independent coordinates with assertions."""
    required = {
        "height_cm",
        "representation",
        "model",
        "model_key",
        "method",
        "observation_id",
        "image_id",
        "pot_id",
        "capture_date",
        "x",
        "y",
    }
    missing = required.difference(coordinates.columns)
    if missing:
        raise ValueError(f"projection coordinates missing columns: {sorted(missing)}")
    if set(coordinates["height_cm"]) != {50}:
        raise ValueError(
            "positive-assertion projection requires only 50 cm coordinates"
        )

    expected_ids = set(cohort["image_id_50"])
    expected_arms = {
        (representation, model_key, method)
        for representation in REPRESENTATIONS
        for model_key in MODEL_KEYS
        for method in METHODS
    }
    grouped = coordinates.groupby(["representation", "model_key", "method"], sort=False)
    if set(grouped.groups) != expected_arms:
        missing_arms = expected_arms.difference(grouped.groups)
        extra_arms = set(grouped.groups).difference(expected_arms)
        raise ValueError(
            f"coordinate arm mismatch; missing={sorted(missing_arms)}, "
            f"extra={sorted(extra_arms)}"
        )
    for arm, group in grouped:
        if len(group) != len(cohort) or set(group["image_id"]) != expected_ids:
            raise ValueError(f"coordinate identities do not match cohort for {arm}")

    metadata = cohort[
        [
            "image_id_50",
            "observation_id",
            "pot_id",
            "pot_label",
            "capture_date",
            "file_name",
            "original_relative_path",
            "source_label",
            "capture_role",
            *SYMPTOMS,
        ]
    ].rename(
        columns={
            "image_id_50": "image_id",
            "observation_id": "metadata_observation_id",
            "pot_id": "metadata_pot_id",
            "capture_date": "metadata_capture_date",
        }
    )
    enriched = coordinates.merge(metadata, on="image_id", validate="many_to_one")
    for coordinate_column, metadata_column in (
        ("observation_id", "metadata_observation_id"),
        ("pot_id", "metadata_pot_id"),
        ("capture_date", "metadata_capture_date"),
    ):
        if not enriched[coordinate_column].eq(enriched[metadata_column]).all():
            raise ValueError(f"coordinate/metadata conflict for {coordinate_column}")
        enriched = enriched.drop(columns=metadata_column)

    for symptom in SYMPTOMS:
        enriched[symptom] = enriched[symptom].eq("true")
    enriched["positive_assertions"] = enriched.apply(
        lambda row: (
            ", ".join(symptom for symptom in SYMPTOMS if bool(row[symptom]))
            or "none recorded"
        ),
        axis=1,
    )

    representation_order = {value: index for index, value in enumerate(REPRESENTATIONS)}
    model_order = {value: index for index, value in enumerate(MODEL_KEYS)}
    method_order = {value: index for index, value in enumerate(METHODS)}
    image_order = {
        image_id: index for index, image_id in enumerate(cohort["image_id_50"])
    }
    enriched["_representation_order"] = enriched["representation"].map(
        representation_order
    )
    enriched["_model_order"] = enriched["model_key"].map(model_order)
    enriched["_method_order"] = enriched["method"].map(method_order)
    enriched["_image_order"] = enriched["image_id"].map(image_order)
    enriched = enriched.sort_values(
        [
            "_representation_order",
            "_model_order",
            "_method_order",
            "_image_order",
        ],
        ignore_index=True,
    ).drop(
        columns=[
            "_representation_order",
            "_model_order",
            "_method_order",
            "_image_order",
        ]
    )
    return enriched


def normalize_panel_coordinates(xy: np.ndarray) -> np.ndarray:
    """Center one panel and preserve its aspect ratio inside a unit square."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (rows, 2)")
    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    scale = float(np.max(maximum - minimum))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("coordinates have no finite spread")
    center = (minimum + maximum) / 2
    return (xy - center) / scale


def circle_polygons(
    xy: np.ndarray, *, radius: float, arc_points: int = 16
) -> tuple[list[float | None], list[float | None]]:
    """Return Plotly-ready, None-separated circle polygons."""
    if radius <= 0 or arc_points < 4:
        raise ValueError("circle geometry requires positive radius and >=4 points")
    angles = np.linspace(0, 2 * np.pi, arc_points + 1)
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for center_x, center_y in np.asarray(xy):
        x_values.extend((center_x + radius * np.cos(angles)).tolist())
        y_values.extend((center_y + radius * np.sin(angles)).tolist())
        x_values.append(None)
        y_values.append(None)
    return x_values, y_values


def symptom_wedge_polygons(
    xy: np.ndarray,
    positive: np.ndarray,
    symptom: str,
    *,
    radius: float,
    arc_points: int = 5,
) -> tuple[list[float | None], list[float | None]]:
    """Return fixed-quadrant wedge polygons only for asserted-positive rows."""
    if symptom not in SYMPTOM_QUADRANTS:
        raise ValueError(f"unknown symptom: {symptom}")
    xy = np.asarray(xy, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    if len(xy) != len(positive):
        raise ValueError("coordinate and positive-mask lengths differ")
    if radius <= 0 or arc_points < 2:
        raise ValueError("wedge geometry requires positive radius and >=2 points")
    start, stop = SYMPTOM_QUADRANTS[symptom]
    angles = np.deg2rad(np.linspace(start, stop, arc_points))
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for center_x, center_y in xy[positive]:
        x_values.append(float(center_x))
        y_values.append(float(center_y))
        x_values.extend((center_x + radius * np.cos(angles)).tolist())
        y_values.extend((center_y + radius * np.sin(angles)).tolist())
        x_values.extend([float(center_x), None])
        y_values.extend([float(center_y), None])
    return x_values, y_values
