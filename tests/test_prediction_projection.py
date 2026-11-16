import json
from pathlib import Path

import numpy as np
import pandas as pd

from agdamage.prediction_projection import build_cropped_prediction_table
from agdamage.treatment import MODEL_KEYS, SYMPTOMS


def cohort():
    rows = []
    for index in range(2):
        rows.append(
            {
                "image_id_50": f"i{index}",
                "observation_id": f"o{index}",
                "pot_id": f"p{index}",
                "pot_label": f"{index:03d}",
                "capture_date": f"2026-05-{7 + index:02d}",
                "file_name": f"images/i{index}.jpg",
                "source_label": (
                    "healthy_untreated" if index == 0 else "chlorosis_only"
                ),
                "herbicide_treated": bool(index),
                "chlorosis": "true" if index else "",
                "necrosis": "",
                "epinasty": "",
                "foliar_distortion": "",
            }
        )
    return pd.DataFrame(rows)


def oof_scores():
    rows = []
    for model_key in MODEL_KEYS:
        for index, row in cohort().iterrows():
            record = {
                "image_id": row.image_id_50,
                "file_name": row.file_name,
                "observation_id": row.observation_id,
                "pot_id": row.pot_id,
                "pot_label": row.pot_label,
                "capture_date": row.capture_date,
                "camera_height_cm": 50,
                "representation": "cropped",
                "model": model_key,
                "model_key": model_key,
                "source_label": row.source_label,
                "herbicide_treated": row.herbicide_treated,
            }
            for symptom in SYMPTOMS:
                record[f"{symptom}_label_state"] = (
                    "positive"
                    if symptom == "chlorosis" and index == 1
                    else "control_reference"
                    if index == 0
                    else "unknown"
                )
                record[f"{symptom}_logit"] = float(index * 2 - 1)
            rows.append(record)
    return pd.DataFrame(rows)


def write_inputs(root: Path):
    embedding_dir = root / "embeddings"
    embedding_dir.mkdir()
    arrays = {}
    classifiers = []
    for model_key in MODEL_KEYS:
        manifest = {"model_key": model_key}
        np.savez_compressed(
            embedding_dir / f"cropped__{model_key}.npz",
            ids=np.asarray(["i0", "i1"]),
            features=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            manifest=np.asarray(json.dumps(manifest)),
        )
        for symptom in SYMPTOMS:
            prefix = f"cropped__{model_key}__{symptom}"
            arrays[f"{prefix}__coef"] = np.asarray([[1.0, 0.0]])
            arrays[f"{prefix}__intercept"] = np.asarray([0.0])
            classifiers.append(
                {
                    "key": prefix,
                    "representation": "cropped",
                    "model_key": model_key,
                    "symptom": symptom,
                }
            )
    arrays["manifest"] = np.asarray(
        json.dumps({"classifier": {}, "classifiers": classifiers})
    )
    classifier_path = root / "classifiers.npz"
    np.savez_compressed(classifier_path, **arrays)
    return embedding_dir, classifier_path


def test_builds_oof_and_final_predictions_with_both_thresholds(tmp_path):
    embedding_dir, classifier_path = write_inputs(tmp_path)
    result, _ = build_cropped_prediction_table(
        cohort(), oof_scores(), embedding_dir, classifier_path
    )
    assert len(result) == 2 * 2 * len(MODEL_KEYS)
    assert set(result["prediction_source"]) == {
        "pot_held_out_oof",
        "all_data_final_in_sample",
    }
    for symptom in SYMPTOMS:
        assert result[f"{symptom}_probability"].between(0, 1).all()
        assert result[f"{symptom}_control_percentile"].between(0, 1).all()
        assert result[f"{symptom}_control95_logit_threshold"].notna().all()
        assert result[f"{symptom}_predicted_logit0"].dtype == bool
        assert result[f"{symptom}_predicted_control95"].dtype == bool
