"""Positive-only symptom probes over precomputed image embeddings."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from agdamage.treatment import (
    MODEL_KEYS,
    MODEL_NAMES,
    REPRESENTATIONS,
    SYMPTOMS,
    _pot_diverse_order,
    arm_features,
    load_rekeyed_bundle,
    string_digest,
)

CONTROL_LABEL = "healthy_untreated"
LABEL_STATES = ("control_reference", "positive", "unknown")


def build_50cm_symptom_cohort(metadata: pd.DataFrame) -> pd.DataFrame:
    """Select one preferred 50 cm image per pot/date with partial labels."""
    selected = metadata[
        metadata["is_preferred_for_height"] & metadata["camera_height_cm"].eq(50)
    ].copy()
    if selected.empty:
        raise ValueError("no preferred 50 cm images")
    if selected["observation_id"].duplicated().any():
        raise ValueError("multiple preferred 50 cm images for one observation")
    if selected["image_id"].duplicated().any():
        raise ValueError("duplicate image ids in 50 cm cohort")

    keep = [
        "observation_id",
        "pot_id",
        "pot_label",
        "capture_date",
        "image_id",
        "file_name",
        "original_relative_path",
        "source_label",
        "capture_role",
        "camera_code_raw",
        "herbicide_treated",
        *SYMPTOMS,
    ]
    cohort = selected[keep].rename(columns={"image_id": "image_id_50"})
    return cohort.sort_values(["capture_date", "pot_label"], ignore_index=True)


def symptom_label_state(cohort: pd.DataFrame, symptom: str) -> pd.Series:
    """Return positive, control-reference, or unknown for one symptom head."""
    if symptom not in SYMPTOMS:
        raise ValueError(f"unknown symptom: {symptom}")
    positive = cohort[symptom].eq("true")
    control = cohort["source_label"].eq(CONTROL_LABEL)
    if (positive & control).any():
        raise ValueError(f"rows cannot be both {symptom}-positive and control")
    values = np.full(len(cohort), "unknown", dtype=object)
    values[control.to_numpy()] = "control_reference"
    values[positive.to_numpy()] = "positive"
    return pd.Series(values, index=cohort.index, name=f"{symptom}_label_state")


def _effective_shots(
    requested: Iterable[int], minimum_positive_pool: int, cap: int
) -> list[int]:
    if minimum_positive_pool < 1:
        raise ValueError("positive training pool is empty")
    maximum = min(minimum_positive_pool, cap)
    shots = sorted({int(value) for value in requested if 0 < int(value) <= maximum})
    if maximum not in shots:
        shots.append(maximum)
    return shots


def build_symptom_evaluation_plan(
    cohort: pd.DataFrame,
    *,
    requested_shots: tuple[int, ...],
    repeats: int,
    seed: int,
    n_folds: int = 5,
    max_shots: int = 60,
) -> dict:
    """Build symptom-specific, pot-held-out folds and nested supports."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")

    symptom_plans = {}
    all_pots = set(cohort["pot_id"])
    for symptom_index, symptom in enumerate(SYMPTOMS):
        states = symptom_label_state(cohort, symptom)
        known = states.ne("unknown")
        known_rows = cohort.loc[known].copy()
        truth = states.loc[known].eq("positive").astype(int)
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed + symptom_index,
        )
        split_records = []
        minimum_positive_pool = len(known_rows)
        for fold_index, (train_index, test_index) in enumerate(
            splitter.split(known_rows, truth, groups=known_rows["pot_id"])
        ):
            train = known_rows.iloc[train_index]
            test = known_rows.iloc[test_index]
            train_states = states.loc[train.index]
            test_states = states.loc[test.index]
            pools = {
                "control_reference": train.loc[
                    train_states.eq("control_reference"),
                    ["observation_id", "pot_id"],
                ],
                "positive": train.loc[
                    train_states.eq("positive"), ["observation_id", "pot_id"]
                ],
            }
            if test_states.nunique() != 2:
                raise ValueError(f"{symptom} fold {fold_index} lacks a test class")
            minimum_positive_pool = min(minimum_positive_pool, len(pools["positive"]))
            split_records.append((fold_index, train, test, pools, test_states))

        shots = _effective_shots(requested_shots, minimum_positive_pool, max_shots)
        folds = []
        for fold_index, _train, test, pools, test_states in split_records:
            test_pots = sorted(test["pot_id"].unique().tolist())
            # Every pot has an untreated baseline in this experiment, but use a
            # defensive complement so unknown images from a held-out pot inherit
            # that pot's fold for out-of-fold scoring.
            if not set(test_pots).issubset(all_pots):
                raise ValueError("fold contains an unknown pot")
            supports = []
            for repeat in range(repeats):
                repeat_rng = np.random.default_rng(
                    [seed, symptom_index, fold_index, repeat]
                )
                supports.append(
                    {
                        name: _pot_diverse_order(pool, repeat_rng)
                        for name, pool in pools.items()
                    }
                )
            folds.append(
                {
                    "fold": fold_index,
                    "test_pots": test_pots,
                    "test_known_observations": sorted(test["observation_id"].tolist()),
                    "test_class_counts": {
                        "control_reference": int(
                            test_states.eq("control_reference").sum()
                        ),
                        "positive": int(test_states.eq("positive").sum()),
                    },
                    "train_pool_sizes": {
                        name: len(pool) for name, pool in pools.items()
                    },
                    "train_pool_pots": {
                        name: int(pool["pot_id"].nunique())
                        for name, pool in pools.items()
                    },
                    "supports": supports,
                }
            )
        symptom_plans[symptom] = {
            "shots_per_class": shots,
            "known_class_counts": {
                "control_reference": int(states.eq("control_reference").sum()),
                "positive": int(states.eq("positive").sum()),
                "unknown": int(states.eq("unknown").sum()),
            },
            "folds": folds,
        }

    plan = {
        "schema": "agdamage.symptom_vs_untreated_50cm_evaluation_plan",
        "schema_version": 1,
        "seed": seed,
        "repeats": repeats,
        "n_folds": n_folds,
        "max_support_images_per_class": max_shots,
        "cohort_observation_ids_sha256": string_digest(cohort["observation_id"]),
        "cohort": [
            {
                "observation_id": row.observation_id,
                "pot_id": row.pot_id,
                "capture_date": row.capture_date,
                "image_id_50": row.image_id_50,
            }
            for row in cohort.itertuples(index=False)
        ],
        "symptoms": symptom_plans,
    }
    validate_symptom_evaluation_plan(cohort, plan)
    return plan


def validate_symptom_evaluation_plan(cohort: pd.DataFrame, plan: dict) -> None:
    """Reject changed identities, pot leakage, and invalid support labels."""
    expected_digest = string_digest(cohort["observation_id"])
    if plan["cohort_observation_ids_sha256"] != expected_digest:
        raise ValueError("evaluation plan does not match cohort identities/order")
    all_pots = set(cohort["pot_id"])
    pot_by_observation = dict(
        zip(cohort["observation_id"], cohort["pot_id"], strict=True)
    )
    for symptom in SYMPTOMS:
        states = symptom_label_state(cohort, symptom)
        state_by_observation = dict(zip(cohort["observation_id"], states, strict=True))
        known_observations = {
            observation
            for observation, state in state_by_observation.items()
            if state != "unknown"
        }
        test_pots_seen = set()
        test_known_seen = set()
        for fold in plan["symptoms"][symptom]["folds"]:
            test_pots = set(fold["test_pots"])
            test_known = set(fold["test_known_observations"])
            if test_pots_seen.intersection(test_pots):
                raise ValueError(f"{symptom}: a pot occurs in multiple test folds")
            if test_known_seen.intersection(test_known):
                raise ValueError(
                    f"{symptom}: an observation occurs in multiple test folds"
                )
            test_pots_seen.update(test_pots)
            test_known_seen.update(test_known)
            for support in fold["supports"]:
                for label in ("control_reference", "positive"):
                    observations = support[label]
                    if any(
                        state_by_observation.get(value) != label
                        for value in observations
                    ):
                        raise ValueError(f"{symptom}: mislabeled {label} support")
                    if any(
                        pot_by_observation[value] in test_pots for value in observations
                    ):
                        raise ValueError(f"{symptom}: support/test pot leakage")
        if test_pots_seen != all_pots:
            raise ValueError(f"{symptom}: folds do not partition all pots")
        if test_known_seen != known_observations:
            raise ValueError(f"{symptom}: folds do not partition known rows")


def _fit_classifier(
    features: np.ndarray,
    rows: np.ndarray,
    truth: np.ndarray,
    *,
    c_value: float,
    seed: int,
    class_weight: str | None = None,
) -> LogisticRegression:
    classifier = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
        class_weight=class_weight,
    )
    classifier.fit(features[rows], truth[rows])
    return classifier


def evaluate_symptom_trials(
    cohort: pd.DataFrame,
    plan: dict,
    embedding_dir,
    *,
    c_value: float = 1.0,
) -> pd.DataFrame:
    """Evaluate balanced few-shot symptom-vs-control logistic probes."""
    validate_symptom_evaluation_plan(cohort, plan)
    observation_row = {
        observation: row for row, observation in enumerate(cohort["observation_id"])
    }
    pot_by_observation = dict(
        zip(cohort["observation_id"], cohort["pot_id"], strict=True)
    )
    trial_rows = []
    for representation, bundle_prefix in REPRESENTATIONS.items():
        for model_key in MODEL_KEYS:
            bundle = load_rekeyed_bundle(
                embedding_dir / f"{bundle_prefix}__{model_key}.npz"
            )
            features = arm_features(cohort, 50, bundle)
            for symptom in SYMPTOMS:
                print(
                    f"classifying {symptom} / {representation} / "
                    f"{MODEL_NAMES[model_key]}",
                    flush=True,
                )
                states = symptom_label_state(cohort, symptom)
                truth = states.eq("positive").astype(int).to_numpy()
                symptom_plan = plan["symptoms"][symptom]
                for fold in symptom_plan["folds"]:
                    test_rows = np.asarray(
                        [
                            observation_row[value]
                            for value in fold["test_known_observations"]
                        ]
                    )
                    test_truth = truth[test_rows]
                    for repeat, support in enumerate(fold["supports"]):
                        for shots in symptom_plan["shots_per_class"]:
                            observations = (
                                support["control_reference"][:shots]
                                + support["positive"][:shots]
                            )
                            train_rows = np.asarray(
                                [observation_row[value] for value in observations]
                            )
                            classifier = _fit_classifier(
                                features,
                                train_rows,
                                truth,
                                c_value=c_value,
                                seed=plan["seed"],
                            )
                            logits = classifier.decision_function(features[test_rows])
                            probabilities = classifier.predict_proba(
                                features[test_rows]
                            )[:, 1]
                            predictions = logits >= 0
                            true_positive = int(((test_truth == 1) & predictions).sum())
                            false_positive = int(
                                ((test_truth == 0) & predictions).sum()
                            )
                            true_negative = int(
                                ((test_truth == 0) & ~predictions).sum()
                            )
                            false_negative = int(
                                ((test_truth == 1) & ~predictions).sum()
                            )
                            trial_rows.append(
                                {
                                    "symptom": symptom,
                                    "height_cm": 50,
                                    "representation": representation,
                                    "model": MODEL_NAMES[model_key],
                                    "model_key": model_key,
                                    "fold": fold["fold"],
                                    "repeat": repeat,
                                    "support_images_per_class": shots,
                                    "n_train": len(train_rows),
                                    "n_train_control_pots": len(
                                        {
                                            pot_by_observation[value]
                                            for value in support["control_reference"][
                                                :shots
                                            ]
                                        }
                                    ),
                                    "n_train_positive_pots": len(
                                        {
                                            pot_by_observation[value]
                                            for value in support["positive"][:shots]
                                        }
                                    ),
                                    "n_test": len(test_rows),
                                    "n_test_control_reference": int(
                                        (1 - test_truth).sum()
                                    ),
                                    "n_test_positive": int(test_truth.sum()),
                                    "true_positive": true_positive,
                                    "false_positive": false_positive,
                                    "true_negative": true_negative,
                                    "false_negative": false_negative,
                                    "balanced_accuracy": balanced_accuracy_score(
                                        test_truth, predictions
                                    ),
                                    "roc_auc": roc_auc_score(test_truth, probabilities),
                                    "average_precision": average_precision_score(
                                        test_truth, probabilities
                                    ),
                                    "f1": f1_score(
                                        test_truth,
                                        predictions,
                                        zero_division=0,
                                    ),
                                    "precision": precision_score(
                                        test_truth,
                                        predictions,
                                        zero_division=0,
                                    ),
                                    "recall": recall_score(
                                        test_truth,
                                        predictions,
                                        zero_division=0,
                                    ),
                                    "log_loss": log_loss(
                                        test_truth,
                                        probabilities,
                                        labels=[0, 1],
                                    ),
                                    "accuracy": accuracy_score(test_truth, predictions),
                                }
                            )
    return pd.DataFrame(trial_rows)


def score_all_images_out_of_fold(
    cohort: pd.DataFrame,
    plan: dict,
    embedding_dir,
    *,
    c_value: float = 1.0,
) -> pd.DataFrame:
    """Score every image with a model that excluded the image's entire pot."""
    validate_symptom_evaluation_plan(cohort, plan)
    rows = []
    for representation, bundle_prefix in REPRESENTATIONS.items():
        for model_key in MODEL_KEYS:
            bundle = load_rekeyed_bundle(
                embedding_dir / f"{bundle_prefix}__{model_key}.npz"
            )
            features = arm_features(cohort, 50, bundle)
            records = [
                {
                    "image_id": row.image_id_50,
                    "file_name": row.file_name,
                    "observation_id": row.observation_id,
                    "pot_id": row.pot_id,
                    "pot_label": row.pot_label,
                    "capture_date": row.capture_date,
                    "camera_height_cm": 50,
                    "representation": representation,
                    "model": MODEL_NAMES[model_key],
                    "model_key": model_key,
                    "source_label": row.source_label,
                    "herbicide_treated": bool(row.herbicide_treated),
                }
                for row in cohort.itertuples(index=False)
            ]
            for symptom in SYMPTOMS:
                states = symptom_label_state(cohort, symptom)
                truth = states.eq("positive").astype(int).to_numpy()
                score_seen = np.zeros(len(cohort), dtype=bool)
                for fold in plan["symptoms"][symptom]["folds"]:
                    test_mask = cohort["pot_id"].isin(fold["test_pots"]).to_numpy()
                    train_mask = (~test_mask) & states.ne("unknown").to_numpy()
                    train_rows = np.flatnonzero(train_mask)
                    test_rows = np.flatnonzero(test_mask)
                    classifier = _fit_classifier(
                        features,
                        train_rows,
                        truth,
                        c_value=c_value,
                        seed=plan["seed"],
                        class_weight="balanced",
                    )
                    logits = classifier.decision_function(features[test_rows])
                    probabilities = classifier.predict_proba(features[test_rows])[:, 1]
                    for row_index, logit, probability in zip(
                        test_rows, logits, probabilities, strict=True
                    ):
                        records[row_index][f"{symptom}_fold"] = fold["fold"]
                        records[row_index][f"{symptom}_label_state"] = states.iloc[
                            row_index
                        ]
                        records[row_index][f"{symptom}_logit"] = float(logit)
                        records[row_index][f"{symptom}_probability"] = float(
                            probability
                        )
                    score_seen[test_rows] = True
                if not score_seen.all():
                    raise ValueError(
                        f"{symptom}: not every image received an OOF score"
                    )

                control_logits = np.sort(
                    np.asarray(
                        [
                            record[f"{symptom}_logit"]
                            for record in records
                            if record[f"{symptom}_label_state"] == "control_reference"
                        ]
                    )
                )
                threshold = float(np.quantile(control_logits, 0.95, method="higher"))
                for record in records:
                    logit = record[f"{symptom}_logit"]
                    record[f"{symptom}_control_percentile"] = float(
                        np.searchsorted(control_logits, logit, side="right")
                        / len(control_logits)
                    )
                    record[f"{symptom}_above_control_95pct"] = bool(logit >= threshold)
            rows.extend(records)
    scores = pd.DataFrame(rows)
    expected = len(cohort) * len(REPRESENTATIONS) * len(MODEL_KEYS)
    if len(scores) != expected:
        raise ValueError(f"expected {expected} score rows, got {len(scores)}")
    return scores


def fit_final_classifiers(
    cohort: pd.DataFrame,
    embedding_dir,
    *,
    c_value: float = 1.0,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit all-known-data heads and return portable coefficients plus metadata."""
    arrays = {}
    classifiers = []
    for representation, bundle_prefix in REPRESENTATIONS.items():
        for model_key in MODEL_KEYS:
            bundle = load_rekeyed_bundle(
                embedding_dir / f"{bundle_prefix}__{model_key}.npz"
            )
            features = arm_features(cohort, 50, bundle)
            for symptom in SYMPTOMS:
                states = symptom_label_state(cohort, symptom)
                rows = np.flatnonzero(states.ne("unknown").to_numpy())
                truth = states.eq("positive").astype(int).to_numpy()
                classifier = _fit_classifier(
                    features,
                    rows,
                    truth,
                    c_value=c_value,
                    seed=seed,
                    class_weight="balanced",
                )
                prefix = f"{representation}__{model_key}__{symptom}"
                arrays[f"{prefix}__coef"] = classifier.coef_.astype(np.float32)
                arrays[f"{prefix}__intercept"] = classifier.intercept_.astype(
                    np.float32
                )
                classifiers.append(
                    {
                        "key": prefix,
                        "representation": representation,
                        "model_key": model_key,
                        "symptom": symptom,
                        "feature_dimension": int(features.shape[1]),
                        "n_control_reference": int(
                            states.eq("control_reference").sum()
                        ),
                        "n_positive": int(states.eq("positive").sum()),
                    }
                )
    manifest = {
        "schema": "agdamage.final_50cm_symptom_logistic_coefficients",
        "schema_version": 1,
        "classifier": {
            "type": "sklearn.linear_model.LogisticRegression",
            "penalty": "L2",
            "C": c_value,
            "solver": "liblinear",
            "class_weight": "balanced",
            "logit_formula": "features @ coef.T + intercept",
        },
        "classifiers": classifiers,
    }
    return arrays, manifest
