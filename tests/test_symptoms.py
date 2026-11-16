import numpy as np
import pandas as pd

from agdamage.symptoms import (
    build_50cm_symptom_cohort,
    build_symptom_evaluation_plan,
    symptom_label_state,
    validate_symptom_evaluation_plan,
)


def metadata_row(pot_number, date_number, source_label):
    symptom = source_label if source_label != "healthy_untreated" else ""
    pot = f"p{pot_number:03d}"
    observation = f"{pot}-d{date_number}"
    return {
        "observation_id": observation,
        "pot_id": pot,
        "pot_label": f"{pot_number:03d}",
        "capture_date": f"2026-05-{7 + date_number:02d}",
        "camera_height_cm": 50,
        "image_id": f"{observation}-image",
        "file_name": f"images/{observation}.jpg",
        "original_relative_path": f"{source_label}/{observation}.jpg",
        "source_label": source_label,
        "capture_role": "standard",
        "camera_code_raw": "IGPTRIAL1",
        "is_preferred_for_height": True,
        "herbicide_treated": source_label != "healthy_untreated",
        "chlorosis": "true" if symptom == "chlorosis" else "",
        "necrosis": "true" if symptom == "necrosis" else "",
        "epinasty": "true" if symptom == "epinasty" else "",
        "foliar_distortion": ("true" if symptom == "foliar_distortion" else ""),
    }


def synthetic_metadata():
    rows = []
    symptoms = ("chlorosis", "necrosis", "epinasty", "foliar_distortion")
    for pot_number in range(20):
        rows.append(metadata_row(pot_number, 0, "healthy_untreated"))
        rows.append(metadata_row(pot_number, 1, symptoms[pot_number % 4]))
    return pd.DataFrame(rows)


def test_partial_label_states_do_not_invent_negatives():
    cohort = build_50cm_symptom_cohort(synthetic_metadata())
    chlorosis = symptom_label_state(cohort, "chlorosis")
    labels = pd.DataFrame({"source": cohort["source_label"], "state": chlorosis})
    assert set(labels.loc[labels.source.eq("healthy_untreated"), "state"]) == {
        "control_reference"
    }
    assert set(labels.loc[labels.source.eq("chlorosis"), "state"]) == {"positive"}
    assert set(labels.loc[labels.source.eq("necrosis"), "state"]) == {"unknown"}


def test_symptom_plan_holds_out_pots_and_uses_smallest_positive_endpoint():
    cohort = build_50cm_symptom_cohort(synthetic_metadata())
    plan = build_symptom_evaluation_plan(
        cohort,
        requested_shots=(1, 2, 5, 10, 20, 40, 60),
        repeats=2,
        seed=42,
        n_folds=5,
        max_shots=60,
    )
    validate_symptom_evaluation_plan(cohort, plan)
    for symptom_plan in plan["symptoms"].values():
        minimum_positive_pool = min(
            fold["train_pool_sizes"]["positive"] for fold in symptom_plan["folds"]
        )
        assert symptom_plan["shots_per_class"][-1] == minimum_positive_pool
        for fold in symptom_plan["folds"]:
            test_pots = set(fold["test_pots"])
            support = fold["supports"][0]
            support_observations = support["control_reference"] + support["positive"]
            pot_lookup = cohort.set_index("observation_id")["pot_id"]
            assert not test_pots.intersection(pot_lookup.loc[support_observations])


def test_effective_supports_are_nested():
    cohort = build_50cm_symptom_cohort(synthetic_metadata())
    plan = build_symptom_evaluation_plan(
        cohort,
        requested_shots=(1, 2),
        repeats=1,
        seed=7,
        n_folds=5,
        max_shots=2,
    )
    support = plan["symptoms"]["chlorosis"]["folds"][0]["supports"][0]
    assert np.asarray(support["positive"][:1]).tolist() == support["positive"][:2][:1]
