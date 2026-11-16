"""Score the 50 cm cohort with held-out and final symptom classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agdamage.symptoms import symptom_label_state
from agdamage.treatment import (
    MODEL_KEYS,
    MODEL_NAMES,
    SYMPTOMS,
    arm_features,
    load_rekeyed_bundle,
)

PREDICTION_SOURCES = {
    "pot_held_out_oof": "Pot-held-out OOF",
    "all_data_final_in_sample": "Final all-data (in-sample)",
}
THRESHOLDS = {
    "logit0": "logit >= 0",
    "control95": "above OOF control 95th percentile",
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def _reference_scores(oof_arm: pd.DataFrame, symptom: str) -> tuple[np.ndarray, float]:
    states = oof_arm[f"{symptom}_label_state"]
    reference = np.sort(
        oof_arm.loc[states.eq("control_reference"), f"{symptom}_logit"].to_numpy(float)
    )
    if len(reference) == 0:
        raise ValueError(f"{symptom}: no OOF control-reference scores")
    threshold = float(np.quantile(reference, 0.95, method="higher"))
    return reference, threshold


def _add_decisions(
    frame: pd.DataFrame,
    reference_by_symptom: dict[str, tuple[np.ndarray, float]],
) -> pd.DataFrame:
    frame = frame.copy()
    for symptom, (reference, threshold) in reference_by_symptom.items():
        logits = frame[f"{symptom}_logit"].to_numpy(float)
        frame[f"{symptom}_probability"] = _sigmoid(logits)
        frame[f"{symptom}_control_percentile"] = [
            np.searchsorted(reference, value, side="right") / len(reference)
            for value in logits
        ]
        frame[f"{symptom}_control95_logit_threshold"] = threshold
        frame[f"{symptom}_predicted_logit0"] = logits >= 0
        frame[f"{symptom}_predicted_control95"] = logits >= threshold
    return frame


def build_cropped_prediction_table(
    cohort: pd.DataFrame,
    oof_scores: pd.DataFrame,
    embedding_dir: Path,
    classifier_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Return matched OOF and all-data predictions for every cropped image."""
    expected_ids = set(cohort["image_id_50"])
    selected_oof = oof_scores[oof_scores["representation"].eq("cropped")].copy()
    expected_rows = len(cohort) * len(MODEL_KEYS)
    if len(selected_oof) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} cropped OOF rows, got {len(selected_oof)}"
        )
    for model_key, arm in selected_oof.groupby("model_key"):
        if model_key not in MODEL_KEYS or set(arm["image_id"]) != expected_ids:
            raise ValueError(f"OOF identities do not match cohort for {model_key}")
        if arm["image_id"].duplicated().any():
            raise ValueError(f"duplicate OOF image scores for {model_key}")

    common_columns = [
        "image_id",
        "file_name",
        "observation_id",
        "pot_id",
        "pot_label",
        "capture_date",
        "camera_height_cm",
        "representation",
        "model",
        "model_key",
        "source_label",
        "herbicide_treated",
    ]
    symptom_columns = [
        f"{symptom}_{suffix}"
        for symptom in SYMPTOMS
        for suffix in ("label_state", "logit")
    ]
    oof_output = selected_oof[[*common_columns, *symptom_columns]].copy()
    oof_output["prediction_source"] = "pot_held_out_oof"
    state_by_symptom = {
        symptom: symptom_label_state(cohort, symptom) for symptom in SYMPTOMS
    }

    with np.load(classifier_path, allow_pickle=False) as stored:
        classifier_arrays = {
            key: np.array(stored[key], copy=True)
            for key in stored.files
            if key != "manifest"
        }
        classifier_manifest = json.loads(str(stored["manifest"].item()))

    final_records = []
    oof_parts = []
    cohort_by_image = cohort.set_index("image_id_50")
    for model_key in MODEL_KEYS:
        arm_oof = selected_oof[selected_oof["model_key"].eq(model_key)].copy()
        reference_by_symptom = {
            symptom: _reference_scores(arm_oof, symptom) for symptom in SYMPTOMS
        }
        bundle = load_rekeyed_bundle(embedding_dir / f"cropped__{model_key}.npz")
        features = arm_features(cohort, 50, bundle)
        logits_by_symptom = {}
        for symptom in SYMPTOMS:
            prefix = f"cropped__{model_key}__{symptom}"
            coefficient = classifier_arrays[f"{prefix}__coef"]
            intercept = classifier_arrays[f"{prefix}__intercept"]
            if coefficient.shape != (1, features.shape[1]) or intercept.shape != (1,):
                raise ValueError(f"classifier shape mismatch for {prefix}")
            logits_by_symptom[symptom] = (features @ coefficient.T + intercept).ravel()

        for row_index, row in enumerate(cohort.itertuples(index=False)):
            record = {
                "image_id": row.image_id_50,
                "file_name": row.file_name,
                "observation_id": row.observation_id,
                "pot_id": row.pot_id,
                "pot_label": row.pot_label,
                "capture_date": row.capture_date,
                "camera_height_cm": 50,
                "representation": "cropped",
                "model": MODEL_NAMES[model_key],
                "model_key": model_key,
                "source_label": row.source_label,
                "herbicide_treated": bool(row.herbicide_treated),
                "prediction_source": "all_data_final_in_sample",
            }
            for symptom in SYMPTOMS:
                record[f"{symptom}_label_state"] = state_by_symptom[symptom].iloc[
                    row_index
                ]
                record[f"{symptom}_logit"] = float(
                    logits_by_symptom[symptom][row_index]
                )
            final_records.append(record)

        oof_parts.append(
            _add_decisions(
                oof_output[oof_output["model_key"].eq(model_key)],
                reference_by_symptom,
            )
        )
    oof_output = pd.concat(oof_parts, ignore_index=True)

    final_output = pd.DataFrame(final_records)
    final_parts = []
    for model_key in MODEL_KEYS:
        arm_oof = selected_oof[selected_oof["model_key"].eq(model_key)]
        reference_by_symptom = {
            symptom: _reference_scores(arm_oof, symptom) for symptom in SYMPTOMS
        }
        final_parts.append(
            _add_decisions(
                final_output[final_output["model_key"].eq(model_key)],
                reference_by_symptom,
            )
        )
    final_output = pd.concat(final_parts, ignore_index=True)

    output = pd.concat([oof_output, final_output], ignore_index=True, sort=False)
    output = output.sort_values(
        ["prediction_source", "model_key", "capture_date", "pot_label"],
        ignore_index=True,
    )
    if len(output) != 2 * expected_rows:
        raise ValueError("prediction table has an unexpected row count")
    for source, group in output.groupby("prediction_source"):
        if group.groupby("model_key")["image_id"].nunique().ne(len(cohort)).any():
            raise ValueError(f"{source}: predictions do not cover every image")
    if not set(output["image_id"]).issubset(cohort_by_image.index):
        raise ValueError("predictions contain an image outside the cohort")
    return output, classifier_manifest
