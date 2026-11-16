import pandas as pd

from agdamage.treatment import (
    build_50cm_cohort,
    build_evaluation_plan,
    build_longitudinal_evaluation_plan,
    build_matched_cohort,
    build_stratified_group_evaluation_plan,
    validate_evaluation_plan,
)


def metadata_row(observation, pot, height, treated, preferred=True):
    return {
        "observation_id": observation,
        "pot_id": pot,
        "pot_label": pot[-3:],
        "capture_date": "2026-05-11",
        "camera_height_cm": height,
        "image_id": f"{observation}-{height}",
        "file_name": f"images/{observation}-{height}.jpg",
        "original_relative_path": f"label/{observation}-{height}.jpg",
        "source_label": "chlorosis_only" if treated else "healthy_untreated",
        "capture_role": "standard",
        "camera_code_raw": "IGPTRIAL1H30" if height == 30 else "IGPTRIAL1",
        "is_preferred_for_height": preferred,
        "herbicide_treated": treated,
        "chlorosis": "true" if treated else "",
        "necrosis": "",
        "epinasty": "",
        "foliar_distortion": "",
    }


def test_matched_cohort_prefers_one_image_per_height():
    rows = [
        metadata_row("o1", "p001", 30, True),
        metadata_row("o1", "p001", 50, True),
        metadata_row("o2", "p002", 30, False),
        metadata_row("o2", "p002", 50, False),
        metadata_row("o3", "p003", 50, True),
        metadata_row("o1", "p001", 30, True, preferred=False),
    ]
    cohort = build_matched_cohort(pd.DataFrame(rows))
    assert cohort["observation_id"].tolist() == ["o1", "o2"]
    assert cohort["image_id_30"].tolist() == ["o1-30", "o2-30"]


def test_evaluation_plan_holds_out_whole_pots_and_nests_supports():
    records = []
    for treated, prefix in [(False, "control"), (True, "treated")]:
        for pot_number in range(2 if not treated else 4):
            pot = f"{prefix}{pot_number}"
            for date_number in range(3):
                records.append(
                    {
                        "observation_id": f"{pot}-d{date_number}",
                        "pot_id": pot,
                        "pot_label": pot,
                        "capture_date": f"2026-05-{11 + date_number:02d}",
                        "herbicide_treated": treated,
                        "image_id_30": f"{pot}-d{date_number}-30",
                        "image_id_50": f"{pot}-d{date_number}-50",
                    }
                )
    cohort = pd.DataFrame(records)
    plan = build_evaluation_plan(cohort, shots=(1, 2, 3), repeats=2, seed=42)
    validate_evaluation_plan(cohort, plan)
    assert len(plan["folds"]) == 2
    for fold in plan["folds"]:
        assert fold["train_pool_sizes"]["untreated"] == 3
        for support in fold["supports"]:
            assert support["untreated"][:1] == support["untreated"][:2][:1]


def test_50cm_cohort_and_longitudinal_plan_keep_pots_whole():
    rows = []
    for pot_number in range(6):
        pot = f"p{pot_number:03d}"
        is_control = pot_number < 2
        for date_number in range(3):
            treated = not is_control and date_number > 0
            row = metadata_row(f"{pot}-d{date_number}", pot, 50, treated)
            row["capture_date"] = f"2026-05-{7 + date_number:02d}"
            rows.append(row)
    cohort = build_50cm_cohort(pd.DataFrame(rows))
    plan = build_longitudinal_evaluation_plan(
        cohort, shots=(1, 2, 3), repeats=2, seed=42
    )
    validate_evaluation_plan(cohort, plan)
    assert len(cohort) == 18
    assert len(plan["folds"]) == 2
    for fold in plan["folds"]:
        test_pots = set(fold["test_pots"])
        test_observations = set(fold["test_observations"])
        assert (
            set(cohort.loc[cohort.pot_id.isin(test_pots), "observation_id"])
            == test_observations
        )
        support = fold["supports"][0]
        first_untreated_pots = {
            cohort.set_index("observation_id").loc[value, "pot_id"]
            for value in support["untreated"][:3]
        }
        assert len(first_untreated_pots) == 3


def test_stratified_group_plan_matches_symptom_strategy():
    rows = []
    for pot_number in range(10):
        pot = f"p{pot_number:03d}"
        for date_number in range(2):
            treated = date_number == 1
            row = metadata_row(f"{pot}-d{date_number}", pot, 50, treated)
            row["capture_date"] = f"2026-05-{7 + date_number:02d}"
            rows.append(row)
    cohort = build_50cm_cohort(pd.DataFrame(rows))
    plan = build_stratified_group_evaluation_plan(
        cohort,
        shots=(1, 2, 5),
        repeats=2,
        seed=42,
        n_folds=5,
    )
    validate_evaluation_plan(cohort, plan)
    assert len(plan["folds"]) == 5
    for fold in plan["folds"]:
        assert fold["test_class_counts"] == {"untreated": 2, "treated": 2}
        test_pots = set(fold["test_pots"])
        support = fold["supports"][0]
        support_observations = support["untreated"] + support["treated"]
        pot_lookup = cohort.set_index("observation_id")["pot_id"]
        assert not test_pots.intersection(pot_lookup.loc[support_observations])
