import csv

import pytest

from agdamage.metadata import (
    build_metadata_rows,
    copy_flat_images,
    parse_filename,
    symptom_values,
    write_metadata_csv,
)


def image_name(camera: str, pot: str, image_id: str) -> str:
    return f"2026-05-11_{camera}_{pot}_1_rgb_{image_id}.jpg"


@pytest.mark.parametrize(
    ("camera", "height", "role"),
    [
        ("IGPTRIAL1", 50, "standard"),
        ("IGPTRIAL1H30", 30, "standard"),
        ("IGPTRIAL1H30C", 30, "retake"),
    ],
)
def test_parse_filename_camera_variants(camera, height, role):
    parsed = parse_filename(
        image_name(camera, "204", "ed974693-50a7-4d0d-b052-a6809c5eca80")
    )
    assert parsed.capture_date.isoformat() == "2026-05-11"
    assert parsed.camera_height_cm == height
    assert parsed.capture_role == role
    assert parsed.pot_label == "204"
    assert parsed.constant_token == 1
    assert parsed.modality == "rgb"


def test_parse_filename_rejects_nonconstant_token():
    name = "2026-05-11_IGPTRIAL1_204_2_rgb_ed974693-50a7-4d0d-b052-a6809c5eca80.jpg"
    with pytest.raises(ValueError, match="constant filename token 1"):
        parse_filename(name)


def test_symptom_mapping_keeps_unrecorded_labels_unknown():
    assert symptom_values("epinasty") == {
        "chlorosis": None,
        "necrosis": None,
        "epinasty": True,
        "foliar_distortion": None,
    }
    assert symptom_values("healthy_untreated") == {
        "chlorosis": None,
        "necrosis": None,
        "epinasty": None,
        "foliar_distortion": None,
    }
    assert symptom_values("chlorosis_only")["necrosis"] is None
    assert symptom_values("necrosis_only")["chlorosis"] is None


def test_flat_metadata_links_h30c_retake_and_writes_nullable_booleans(tmp_path):
    source = tmp_path / "source"
    image_ids = {
        "h30": "11111111-1111-4111-8111-111111111111",
        "h30c": "22222222-2222-4222-8222-222222222222",
        "h50": "33333333-3333-4333-8333-333333333333",
    }
    for label, camera, key in [
        ("chlorosis_only", "IGPTRIAL1H30", "h30"),
        ("epinasty", "IGPTRIAL1H30C", "h30c"),
        ("healthy_untreated", "IGPTRIAL1", "h50"),
    ]:
        path = source / label / image_name(camera, "204", image_ids[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.new("RGB", (8, 6), "green").save(path)

    rows = build_metadata_rows(source)
    by_id = {row["image_id"]: row for row in rows}
    h30 = by_id[image_ids["h30"]]
    h30c = by_id[image_ids["h30c"]]
    assert not h30["is_preferred_for_height"]
    assert h30["superseded_by_image_id"] == image_ids["h30c"]
    assert h30c["is_preferred_for_height"]
    assert h30c["retake_of_image_id"] == image_ids["h30"]

    output = tmp_path / "flat"
    copy_flat_images(rows, output)
    write_metadata_csv(rows, output / "metadata.csv")
    assert len(list((output / "images").iterdir())) == 3

    with (output / "metadata.csv").open(newline="") as handle:
        csv_rows = {row["image_id"]: row for row in csv.DictReader(handle)}
    assert csv_rows[image_ids["h30"]]["chlorosis"] == "true"
    assert csv_rows[image_ids["h30"]]["necrosis"] == ""
    assert csv_rows[image_ids["h30"]]["epinasty"] == ""
    assert csv_rows[image_ids["h50"]]["foliar_distortion"] == ""
    assert csv_rows[image_ids["h30"]]["herbicide_treated"] == "true"
    assert csv_rows[image_ids["h50"]]["herbicide_treated"] == "false"
    assert csv_rows[image_ids["h50"]]["herbicide"] == ""


def test_flattening_rejects_duplicate_filenames(tmp_path):
    source = tmp_path / "source"
    name = image_name("IGPTRIAL1", "204", "33333333-3333-4333-8333-333333333333")
    from PIL import Image

    for label in ("healthy_untreated", "epinasty"):
        path = source / label / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "green").save(path)
    with pytest.raises(ValueError, match="duplicate filename"):
        build_metadata_rows(source)
