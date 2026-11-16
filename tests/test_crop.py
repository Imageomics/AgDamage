from pathlib import Path

import pytest
from PIL import Image

from agdamage.crop import (
    CropSettings,
    Detection,
    centered_square_box,
    choose_central_detection,
    clip_box,
    crop_dataset,
    evenly_sample_per_class,
    padded_square_box,
    review_reasons,
)


def test_clip_box_rejects_box_entirely_outside_image():
    with pytest.raises(ValueError, match="empty box"):
        clip_box((1300, 10, 1400, 100), width=1200, height=900)


def test_padded_square_box_is_square_and_contains_subject():
    crop = padded_square_box((400, 300, 600, 700), 1200, 900, padding=0.10)
    assert crop == (260, 260, 740, 740)
    assert crop[2] - crop[0] == crop[3] - crop[1]
    assert crop[0] <= 400 and crop[1] <= 300
    assert crop[2] >= 600 and crop[3] >= 700


def test_padded_square_box_shifts_at_image_edge():
    crop = padded_square_box((0, 10, 200, 210), 1200, 900, padding=0.10)
    assert crop == (0, 0, 240, 240)


def test_centered_square_box_uses_short_dimension():
    assert centered_square_box(1200, 900, 0.30) == (465, 315, 735, 585)


def test_central_detection_beats_slightly_more_confident_edge_candidate():
    selected = choose_central_detection(
        [
            Detection((450, 300, 750, 600), 0.82),
            Detection((0, 0, 250, 250), 0.95),
        ],
        width=1200,
        height=900,
    )
    assert selected is not None
    assert selected.detection.score == pytest.approx(0.82)
    assert selected.contains_center


def test_review_reasons_flag_shadow_like_geometry():
    selected = choose_central_detection(
        [Detection((0, 0, 100, 100), 0.45)], width=1200, height=900
    )
    reasons = review_reasons(selected, 1200, 900, CropSettings())
    assert reasons == ["low_confidence", "off_center"]


def test_even_sample_per_class():
    paths = [Path(f"root/a/{index}.jpg") for index in range(8)]
    paths += [Path(f"root/b/{index}.jpg") for index in range(4)]
    sampled = evenly_sample_per_class(paths, 2)
    assert sampled == [
        Path("root/a/2.jpg"),
        Path("root/a/6.jpg"),
        Path("root/b/1.jpg"),
        Path("root/b/3.jpg"),
    ]


def test_crop_dataset_materializes_mirrored_image_and_manifests(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "healthy" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "green").save(source_path)

    class FakeDetector:
        def detect(self, image, prompt):
            assert image.size == (100, 80)
            assert prompt == "potted plant"
            return [Detection((30, 20, 70, 60), 0.9)]

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "crops"
    manifest = crop_dataset(
        source_root,
        output_root,
        [source_path],
        FakeDetector(),
        "potted plant",
        CropSettings(),
    )

    with Image.open(output_root / "healthy" / "image.jpg") as crop:
        assert crop.size == (48, 48)
    assert (output_root / "crops.csv").is_file()
    assert (output_root / "manifest.json").is_file()
    assert manifest["summary"] == {"ok": 1}


def test_crop_dataset_uses_broader_fallback_for_weak_primary(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "necrosis" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "gray").save(source_path)

    class FakeDetector:
        def detect(self, image, prompt):
            score = 0.4 if prompt == "potted plant" else 0.8
            return [Detection((30, 20, 70, 60), score)]

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "crops"
    crop_dataset(
        source_root,
        output_root,
        [source_path],
        FakeDetector(),
        "potted plant",
        CropSettings(),
        fallback_prompt="plant",
    )

    import csv

    with (output_root / "crops.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["prompt_used"] == "plant"
    assert row["prompts_tried"] == "potted plant;plant"
    assert row["score"] == "0.8"


def test_crop_dataset_uses_pot_as_last_resort(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "necrosis" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "gray").save(source_path)

    class FakeDetector:
        def detect(self, image, prompt):
            if prompt == "flower pot":
                return [Detection((30, 20, 70, 60), 0.9)]
            return []

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "crops"
    crop_dataset(
        source_root,
        output_root,
        [source_path],
        FakeDetector(),
        "potted plant",
        CropSettings(),
        fallback_prompt="plant",
        last_resort_prompt="flower pot",
    )

    import csv

    with (output_root / "crops.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "ok"
    assert row["prompt_used"] == "flower pot"
    assert row["prompts_tried"] == "potted plant;plant;flower pot"


def test_crop_dataset_uses_reviewable_center_fallback(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "necrosis" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "gray").save(source_path)

    class EmptyDetector:
        def detect(self, image, prompt):
            return []

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "crops"
    crop_dataset(
        source_root,
        output_root,
        [source_path],
        EmptyDetector(),
        "potted plant",
        CropSettings(center_fallback_fraction=0.30),
        fallback_prompt="plant",
        last_resort_prompt="flower pot",
    )

    import csv

    with (output_root / "crops.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    with Image.open(output_root / "necrosis" / "image.jpg") as crop:
        assert crop.size == (24, 24)
    assert row["status"] == "review"
    assert row["crop_method"] == "center_fallback"
    assert row["review_reasons"] == "center_fallback"
    assert row["prompts_tried"] == "potted plant;plant;flower pot"
