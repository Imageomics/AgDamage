"""Matched-cohort treatment classification over precomputed image embeddings."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

MODEL_KEYS = ("bioclip", "bioclip-2", "bioclip-2.5", "dinov3-vit7b16")
MODEL_NAMES = {
    "bioclip": "BioCLIP",
    "bioclip-2": "BioCLIP 2",
    "bioclip-2.5": "BioCLIP 2.5",
    "dinov3-vit7b16": "DINOv3 ViT-7B/16",
}
REPRESENTATIONS = {
    "full": "original",
    "cropped": "cropped",
    "segmented": "plant_only",
}
SYMPTOMS = ("chlorosis", "necrosis", "epinasty", "foliar_distortion")


def string_digest(values) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the flat CSV without converting blanks into false symptom labels."""
    metadata = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "observation_id",
        "pot_id",
        "pot_label",
        "capture_date",
        "camera_height_cm",
        "image_id",
        "file_name",
        "original_relative_path",
        "source_label",
        "capture_role",
        "camera_code_raw",
        "is_preferred_for_height",
        "herbicide_treated",
        *SYMPTOMS,
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")
    for column in ("is_preferred_for_height", "herbicide_treated"):
        unexpected = set(metadata[column]).difference({"true", "false"})
        if unexpected:
            raise ValueError(f"{column} contains non-Boolean values: {unexpected}")
        metadata[column] = metadata[column].eq("true")
    metadata["camera_height_cm"] = metadata["camera_height_cm"].astype(int)
    return metadata


def _positive_symptoms(row: pd.Series) -> str:
    values = [name for name in SYMPTOMS if row[name] == "true"]
    return ", ".join(values) if values else "none recorded"


def build_matched_cohort(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per pot/date having preferred captures at both 30 and 50 cm."""
    preferred = metadata[metadata["is_preferred_for_height"]].copy()
    duplicates = preferred.duplicated(
        ["observation_id", "camera_height_cm"], keep=False
    )
    if duplicates.any():
        keys = preferred.loc[duplicates, ["observation_id", "camera_height_cm"]]
        raise ValueError(
            f"multiple preferred images for an observation/height:\n{keys}"
        )

    records = []
    for observation_id, group in preferred.groupby("observation_id", sort=True):
        by_height = {int(row.camera_height_cm): row for _, row in group.iterrows()}
        if not {30, 50}.issubset(by_height):
            continue
        labels = {bool(by_height[height].herbicide_treated) for height in (30, 50)}
        if len(labels) != 1:
            raise ValueError(
                f"treatment-label conflict between heights: {observation_id}"
            )
        first = by_height[30]
        record = {
            "observation_id": observation_id,
            "pot_id": first.pot_id,
            "pot_label": first.pot_label,
            "capture_date": first.capture_date,
            "herbicide_treated": labels.pop(),
        }
        for height in (30, 50):
            row = by_height[height]
            for column in (
                "image_id",
                "file_name",
                "original_relative_path",
                "source_label",
                "capture_role",
                "camera_code_raw",
            ):
                record[f"{column}_{height}"] = row[column]
            record[f"positive_symptoms_{height}"] = _positive_symptoms(row)
        records.append(record)

    cohort = pd.DataFrame(records).sort_values(
        ["capture_date", "pot_label"], ignore_index=True
    )
    if cohort.empty:
        raise ValueError("no matched 30/50 cm observations")
    for date, group in cohort.groupby("capture_date"):
        if set(group["herbicide_treated"]) != {False, True}:
            raise ValueError(f"matched date lacks both treatment classes: {date}")
    if cohort["observation_id"].duplicated().any():
        raise ValueError("matched cohort contains duplicate observation ids")
    return cohort


def build_50cm_cohort(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per preferred 50 cm image, retaining every available date."""
    selected = metadata[
        metadata["is_preferred_for_height"] & metadata["camera_height_cm"].eq(50)
    ].copy()
    if selected["observation_id"].duplicated().any():
        raise ValueError("multiple preferred 50 cm images for one observation")
    records = []
    for row in selected.itertuples(index=False):
        record = {
            "observation_id": row.observation_id,
            "pot_id": row.pot_id,
            "pot_label": row.pot_label,
            "capture_date": row.capture_date,
            "herbicide_treated": bool(row.herbicide_treated),
        }
        for column in (
            "image_id",
            "file_name",
            "original_relative_path",
            "source_label",
            "capture_role",
            "camera_code_raw",
        ):
            record[f"{column}_50"] = getattr(row, column)
        symptom_values = [
            symptom for symptom in SYMPTOMS if getattr(row, symptom) == "true"
        ]
        record["positive_symptoms_50"] = (
            ", ".join(symptom_values) if symptom_values else "none recorded"
        )
        records.append(record)
    cohort = pd.DataFrame(records).sort_values(
        ["capture_date", "pot_label"], ignore_index=True
    )
    if cohort.empty:
        raise ValueError("no preferred 50 cm images")
    return cohort


def _pot_diverse_order(pool: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Order observations in rounds so each pot appears before reuse."""
    by_pot = {}
    for pot_id, group in pool.groupby("pot_id"):
        values = group["observation_id"].tolist()
        by_pot[pot_id] = list(np.asarray(values)[rng.permutation(len(values))])
    ordered = []
    while by_pot:
        pots = list(by_pot)
        pots = list(np.asarray(pots)[rng.permutation(len(pots))])
        for pot_id in pots:
            ordered.append(by_pot[pot_id].pop())
            if not by_pot[pot_id]:
                del by_pot[pot_id]
    return ordered


def build_longitudinal_evaluation_plan(
    cohort: pd.DataFrame,
    shots: tuple[int, ...],
    repeats: int,
    seed: int,
) -> dict:
    """Hold out whole pots while allowing their labels to change over time."""
    ever_treated = cohort.groupby("pot_id")["herbicide_treated"].any()
    untreated_pots = sorted(ever_treated.index[~ever_treated].tolist())
    treated_pots = sorted(ever_treated.index[ever_treated].tolist())
    if len(untreated_pots) < 2:
        raise ValueError("at least two never-treated pots are required")
    n_folds = len(untreated_pots)
    if len(treated_pots) % n_folds:
        raise ValueError("treated pots do not divide evenly across control-pot folds")

    rng = np.random.default_rng(seed)
    untreated_pots = list(
        np.asarray(untreated_pots)[rng.permutation(len(untreated_pots))]
    )
    treated_pots = list(np.asarray(treated_pots)[rng.permutation(len(treated_pots))])
    treated_per_fold = len(treated_pots) // n_folds
    max_shots = max(shots)
    folds = []
    for fold_index in range(n_folds):
        test_pots = {
            untreated_pots[fold_index],
            *treated_pots[
                fold_index * treated_per_fold : (fold_index + 1) * treated_per_fold
            ],
        }
        test_mask = cohort["pot_id"].isin(test_pots)
        train = cohort[~test_mask]
        test = cohort[test_mask]
        class_pools = {
            "untreated": train[~train["herbicide_treated"]],
            "treated": train[train["herbicide_treated"]],
        }
        short = {
            name: len(pool)
            for name, pool in class_pools.items()
            if len(pool) < max_shots
        }
        if short:
            raise ValueError(f"fold {fold_index} has insufficient support: {short}")
        supports = []
        for repeat in range(repeats):
            repeat_rng = np.random.default_rng([seed, fold_index, repeat])
            supports.append(
                {
                    name: _pot_diverse_order(pool, repeat_rng)
                    for name, pool in class_pools.items()
                }
            )
        folds.append(
            {
                "fold": fold_index,
                "test_pots": sorted(test_pots),
                "test_observations": sorted(test["observation_id"].tolist()),
                "test_class_counts": {
                    "untreated": int((~test["herbicide_treated"]).sum()),
                    "treated": int(test["herbicide_treated"].sum()),
                },
                "train_pool_sizes": {
                    name: len(pool) for name, pool in class_pools.items()
                },
                "train_pool_pots": {
                    name: int(pool["pot_id"].nunique())
                    for name, pool in class_pools.items()
                },
                "supports": supports,
            }
        )

    cohort_records = [
        {
            "observation_id": row.observation_id,
            "pot_id": row.pot_id,
            "capture_date": row.capture_date,
            "herbicide_treated": bool(row.herbicide_treated),
            "image_id_50": row.image_id_50,
        }
        for row in cohort.itertuples(index=False)
    ]
    return {
        "schema": "agdamage.treated_vs_untreated_50cm_evaluation_plan",
        "schema_version": 1,
        "seed": seed,
        "shots_per_class": list(shots),
        "repeats": repeats,
        "n_folds": n_folds,
        "cohort_observation_ids_sha256": string_digest(cohort["observation_id"]),
        "cohort": cohort_records,
        "folds": folds,
    }


def build_stratified_group_evaluation_plan(
    cohort: pd.DataFrame,
    shots: tuple[int, ...],
    repeats: int,
    seed: int,
    *,
    n_folds: int = 5,
) -> dict:
    """Match the symptom-probe strategy with stratified pot-held-out folds."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    shots = tuple(sorted({int(value) for value in shots if int(value) > 0}))
    if not shots:
        raise ValueError("at least one positive support count is required")

    truth = cohort["herbicide_treated"].astype(int)
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    folds = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(cohort, truth, groups=cohort["pot_id"])
    ):
        train = cohort.iloc[train_index]
        test = cohort.iloc[test_index]
        pools = {
            "untreated": train[~train["herbicide_treated"]][
                ["observation_id", "pot_id"]
            ],
            "treated": train[train["herbicide_treated"]][["observation_id", "pot_id"]],
        }
        short = {
            name: len(pool) for name, pool in pools.items() if len(pool) < max(shots)
        }
        if short:
            raise ValueError(f"fold {fold_index} has insufficient support: {short}")
        if test["herbicide_treated"].nunique() != 2:
            raise ValueError(f"fold {fold_index} lacks a test class")
        supports = []
        for repeat in range(repeats):
            repeat_rng = np.random.default_rng([seed, fold_index, repeat])
            supports.append(
                {
                    name: _pot_diverse_order(pool, repeat_rng)
                    for name, pool in pools.items()
                }
            )
        folds.append(
            {
                "fold": fold_index,
                "test_pots": sorted(test["pot_id"].unique().tolist()),
                "test_observations": sorted(test["observation_id"].tolist()),
                "test_class_counts": {
                    "untreated": int((~test["herbicide_treated"]).sum()),
                    "treated": int(test["herbicide_treated"].sum()),
                },
                "train_pool_sizes": {name: len(pool) for name, pool in pools.items()},
                "train_pool_pots": {
                    name: int(pool["pot_id"].nunique()) for name, pool in pools.items()
                },
                "supports": supports,
            }
        )

    plan = {
        "schema": "agdamage.treated_vs_untreated_50cm_matched_strategy_plan",
        "schema_version": 1,
        "seed": seed,
        "shots_per_class": list(shots),
        "repeats": repeats,
        "n_folds": n_folds,
        "cohort_observation_ids_sha256": string_digest(cohort["observation_id"]),
        "cohort": [
            {
                "observation_id": row.observation_id,
                "pot_id": row.pot_id,
                "capture_date": row.capture_date,
                "herbicide_treated": bool(row.herbicide_treated),
                "image_id_50": row.image_id_50,
            }
            for row in cohort.itertuples(index=False)
        ],
        "folds": folds,
    }
    validate_evaluation_plan(cohort, plan)
    return plan


def build_evaluation_plan(
    cohort: pd.DataFrame,
    shots: tuple[int, ...],
    repeats: int,
    seed: int,
) -> dict:
    """Build six pot-held-out folds with nested, arm-independent supports."""
    pot_classes = cohort.groupby("pot_id")["herbicide_treated"].nunique()
    if (pot_classes != 1).any():
        raise ValueError("a matched-cohort pot changes treatment class")
    class_by_pot = cohort.groupby("pot_id")["herbicide_treated"].first()
    untreated_pots = sorted(class_by_pot.index[~class_by_pot].tolist())
    treated_pots = sorted(class_by_pot.index[class_by_pot].tolist())
    if len(untreated_pots) < 2:
        raise ValueError("at least two untreated pots are required")

    n_folds = len(untreated_pots)
    if len(treated_pots) % n_folds:
        raise ValueError("treated pots do not divide evenly across control-pot folds")
    rng = np.random.default_rng(seed)
    untreated_pots = list(
        np.asarray(untreated_pots)[rng.permutation(len(untreated_pots))]
    )
    treated_pots = list(np.asarray(treated_pots)[rng.permutation(len(treated_pots))])
    treated_per_fold = len(treated_pots) // n_folds
    max_shots = max(shots)
    folds = []

    for fold_index in range(n_folds):
        test_pots = {
            untreated_pots[fold_index],
            *treated_pots[
                fold_index * treated_per_fold : (fold_index + 1) * treated_per_fold
            ],
        }
        test_mask = cohort["pot_id"].isin(test_pots)
        train = cohort[~test_mask]
        test = cohort[test_mask]
        pools = {
            "untreated": sorted(
                train.loc[~train["herbicide_treated"], "observation_id"].tolist()
            ),
            "treated": sorted(
                train.loc[train["herbicide_treated"], "observation_id"].tolist()
            ),
        }
        short = {
            name: len(pool) for name, pool in pools.items() if len(pool) < max_shots
        }
        if short:
            raise ValueError(f"fold {fold_index} has insufficient support: {short}")
        supports = []
        for repeat in range(repeats):
            repeat_rng = np.random.default_rng([seed, fold_index, repeat])
            supports.append(
                {
                    name: list(
                        np.asarray(pool)[repeat_rng.permutation(len(pool))[:max_shots]]
                    )
                    for name, pool in pools.items()
                }
            )
        folds.append(
            {
                "fold": fold_index,
                "test_pots": sorted(test_pots),
                "test_observations": sorted(test["observation_id"].tolist()),
                "test_class_counts": {
                    "untreated": int((~test["herbicide_treated"]).sum()),
                    "treated": int(test["herbicide_treated"].sum()),
                },
                "train_pool_sizes": {name: len(pool) for name, pool in pools.items()},
                "supports": supports,
            }
        )

    cohort_records = [
        {
            "observation_id": row.observation_id,
            "pot_id": row.pot_id,
            "capture_date": row.capture_date,
            "herbicide_treated": bool(row.herbicide_treated),
            "image_id_30": row.image_id_30,
            "image_id_50": row.image_id_50,
        }
        for row in cohort.itertuples(index=False)
    ]
    return {
        "schema": "agdamage.treated_vs_untreated_evaluation_plan",
        "schema_version": 1,
        "seed": seed,
        "shots_per_class": list(shots),
        "repeats": repeats,
        "n_folds": n_folds,
        "cohort_observation_ids_sha256": string_digest(cohort["observation_id"]),
        "cohort": cohort_records,
        "folds": folds,
    }


def validate_evaluation_plan(cohort: pd.DataFrame, plan: dict) -> None:
    """Reject leakage, missing observations, and changed cohort ordering."""
    if plan["cohort_observation_ids_sha256"] != string_digest(cohort["observation_id"]):
        raise ValueError("evaluation plan does not match cohort identities/order")
    all_observations = set(cohort["observation_id"])
    test_seen = set()
    test_pots_seen = set()
    for fold in plan["folds"]:
        test = set(fold["test_observations"])
        if test_seen.intersection(test):
            raise ValueError("an observation appears in more than one test fold")
        test_seen.update(test)
        test_pots = set(fold["test_pots"])
        if test_pots_seen.intersection(test_pots):
            raise ValueError("a pot appears in more than one test fold")
        test_pots_seen.update(test_pots)
        for support in fold["supports"]:
            chosen = set(support["treated"]) | set(support["untreated"])
            if chosen.intersection(test):
                raise ValueError("support and test observations overlap")
            if not chosen.issubset(all_observations):
                raise ValueError("support contains an unknown observation")
    if test_seen != all_observations:
        raise ValueError("test folds do not partition the cohort")
    if test_pots_seen != set(cohort["pot_id"]):
        raise ValueError("test folds do not partition the pots")


def load_rekeyed_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {
            "ids": stored["ids"].astype(str),
            "features": np.array(stored["features"], copy=True),
            "manifest": json.loads(str(stored["manifest"].item())),
        }


def arm_features(
    cohort: pd.DataFrame,
    height_cm: int,
    bundle: dict,
) -> np.ndarray:
    """Select one embedding row per cohort observation in cohort order."""
    index = {image_id: row for row, image_id in enumerate(bundle["ids"])}
    ids = cohort[f"image_id_{height_cm}"].tolist()
    missing = [image_id for image_id in ids if image_id not in index]
    if missing:
        raise ValueError(f"embedding bundle missing {len(missing)} cohort images")
    return bundle["features"][[index[image_id] for image_id in ids]]


def evaluate_logistic_trials(
    cohort: pd.DataFrame,
    plan: dict,
    embedding_dir: Path,
    *,
    c_value: float = 1.0,
    heights: tuple[int, ...] = (30, 50),
) -> pd.DataFrame:
    """Evaluate every height/representation/model with shared support draws."""
    validate_evaluation_plan(cohort, plan)
    observation_row = {
        observation_id: row
        for row, observation_id in enumerate(cohort["observation_id"].tolist())
    }
    truth = cohort["herbicide_treated"].astype(int).to_numpy()
    pot_by_observation = dict(
        zip(cohort["observation_id"], cohort["pot_id"], strict=True)
    )
    trial_rows = []

    for height_cm in heights:
        for representation, bundle_prefix in REPRESENTATIONS.items():
            for model_key in MODEL_KEYS:
                bundle_path = embedding_dir / f"{bundle_prefix}__{model_key}.npz"
                bundle = load_rekeyed_bundle(bundle_path)
                features = arm_features(cohort, height_cm, bundle)
                print(
                    f"classifying {height_cm} cm / {representation} / {MODEL_NAMES[model_key]}",
                    flush=True,
                )
                for fold in plan["folds"]:
                    test_rows = np.asarray(
                        [observation_row[value] for value in fold["test_observations"]]
                    )
                    test_x = features[test_rows]
                    test_y = truth[test_rows]
                    for repeat, support in enumerate(fold["supports"]):
                        for shots in plan["shots_per_class"]:
                            support_observations = (
                                support["untreated"][:shots]
                                + support["treated"][:shots]
                            )
                            train_rows = np.asarray(
                                [
                                    observation_row[value]
                                    for value in support_observations
                                ]
                            )
                            train_y = truth[train_rows]
                            classifier = LogisticRegression(
                                C=c_value,
                                solver="liblinear",
                                max_iter=2000,
                                random_state=plan["seed"],
                            )
                            classifier.fit(features[train_rows], train_y)
                            probability = classifier.predict_proba(test_x)[:, 1]
                            prediction = probability >= 0.5
                            true_positive = int(((test_y == 1) & prediction).sum())
                            false_positive = int(((test_y == 0) & prediction).sum())
                            true_negative = int(((test_y == 0) & ~prediction).sum())
                            false_negative = int(((test_y == 1) & ~prediction).sum())
                            trial_rows.append(
                                {
                                    "height_cm": height_cm,
                                    "representation": representation,
                                    "model": MODEL_NAMES[model_key],
                                    "model_key": model_key,
                                    "fold": fold["fold"],
                                    "repeat": repeat,
                                    "support_images_per_class": shots,
                                    "n_train": len(train_rows),
                                    "n_train_pots": len(
                                        {
                                            pot_by_observation[value]
                                            for value in support_observations
                                        }
                                    ),
                                    "n_test": len(test_rows),
                                    "n_test_treated": int(test_y.sum()),
                                    "n_test_untreated": int((1 - test_y).sum()),
                                    "true_positive": true_positive,
                                    "false_positive": false_positive,
                                    "true_negative": true_negative,
                                    "false_negative": false_negative,
                                    "balanced_accuracy": balanced_accuracy_score(
                                        test_y, prediction
                                    ),
                                    "roc_auc": roc_auc_score(test_y, probability),
                                    "average_precision": average_precision_score(
                                        test_y, probability
                                    ),
                                    "f1": f1_score(test_y, prediction, zero_division=0),
                                    "precision": precision_score(
                                        test_y, prediction, zero_division=0
                                    ),
                                    "recall": recall_score(
                                        test_y, prediction, zero_division=0
                                    ),
                                    "accuracy": accuracy_score(test_y, prediction),
                                }
                            )
    return pd.DataFrame(trial_rows)


def project(features: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Compute deterministic cosine t-SNE and UMAP coordinates."""
    import umap

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=30,
        max_iter=1000,
        random_state=seed,
    ).fit_transform(features)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
        n_jobs=1,
        low_memory=True,
    )
    return {"t-SNE": tsne, "UMAP": reducer.fit_transform(features)}
