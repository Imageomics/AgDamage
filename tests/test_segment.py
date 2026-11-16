import csv

import pytest
from PIL import Image, ImageDraw

from agdamage.crop import Detection, SegmentedDetection
from agdamage.segment import (
    SegmentSettings,
    border_median_color,
    choose_central_segmentation,
    constrain_mask_to_detection,
    segment_dataset,
)


def rectangle_mask(size=(100, 80), box=(30, 20, 70, 60)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def test_border_median_color_ignores_central_subject():
    image = Image.new("RGB", (100, 80), (140, 150, 160))
    ImageDraw.Draw(image).rectangle((25, 15, 75, 65), fill=(20, 180, 30))
    assert border_median_color(image) == (140, 150, 160)


def test_choose_central_segmentation_ignores_empty_high_score_mask():
    chosen = choose_central_segmentation(
        [
            SegmentedDetection(
                Detection((30, 20, 70, 60), 0.99), Image.new("L", (100, 80), 0)
            ),
            SegmentedDetection(Detection((30, 20, 70, 60), 0.80), rectangle_mask()),
        ],
        100,
        80,
    )
    assert chosen is not None
    assert chosen[1].detection.score == pytest.approx(0.80)


def test_constrain_mask_removes_distant_fragment_but_preserves_subject():
    mask = rectangle_mask()
    mask.putpixel((50, 2), 255)
    constrained = constrain_mask_to_detection(
        mask, (30, 20, 70, 60), width=100, height=80, padding=0.02
    )
    assert constrained.getpixel((50, 2)) == 0
    assert constrained.getbbox() == (30, 20, 71, 61)


def test_segment_dataset_saves_same_id_cutout_and_full_resolution_mask(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "healthy" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (100, 80), (150, 150, 150))
    ImageDraw.Draw(image).rectangle((30, 20, 70, 60), fill=(20, 180, 30))
    image.save(source_path)

    class FakeSegmenter:
        def segment(self, image, prompt):
            assert image.size == (100, 80)
            assert prompt == "plant"
            return [
                SegmentedDetection(Detection((30, 20, 70, 60), 0.9), rectangle_mask())
            ]

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "segments"
    mask_root = tmp_path / "masks"
    manifest = segment_dataset(
        source_root,
        output_root,
        mask_root,
        [source_path],
        FakeSegmenter(),
        "plant",
        SegmentSettings(),
    )

    with Image.open(output_root / "healthy" / "image.jpg") as segment:
        assert segment.size == (50, 50)
    with Image.open(mask_root / "healthy" / "image.png") as mask:
        assert mask.size == (100, 80)
        assert mask.getbbox() == (30, 20, 71, 61)
    assert manifest["summary"] == {"ok": 1}
    assert manifest["segment"]["resize"] is None

    with (output_root / "segments.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["id"] == "healthy/image.jpg"
    assert row["mask_id"] == "healthy/image.png"
    assert row["prompt_used"] == "plant"


def test_segment_dataset_uses_dead_plant_fallback_for_weak_primary(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "necrosis" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), (150, 150, 150)).save(source_path)

    class FakeSegmenter:
        def segment(self, image, prompt):
            score = 0.4 if prompt == "plant" else 0.8
            return [
                SegmentedDetection(Detection((30, 20, 70, 60), score), rectangle_mask())
            ]

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "segments"
    segment_dataset(
        source_root,
        output_root,
        tmp_path / "masks",
        [source_path],
        FakeSegmenter(),
        "plant",
        SegmentSettings(),
        fallback_prompt="dead plant",
    )
    with (output_root / "segments.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["prompt_used"] == "dead plant"
    assert row["prompts_tried"] == "plant;dead plant"
    assert row["score"] == "0.8"


def test_segment_dataset_materializes_reviewable_empty_mask(tmp_path):
    source_root = tmp_path / "source"
    source_path = source_root / "necrosis" / "image.jpg"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), (150, 150, 150)).save(source_path)

    class EmptySegmenter:
        def segment(self, image, prompt):
            return []

        def provenance(self):
            return {"model_id": "fake"}

    output_root = tmp_path / "segments"
    mask_root = tmp_path / "masks"
    segment_dataset(
        source_root,
        output_root,
        mask_root,
        [source_path],
        EmptySegmenter(),
        "plant",
        SegmentSettings(center_fallback_fraction=0.30),
        fallback_prompt="dead plant",
    )

    with Image.open(output_root / "necrosis" / "image.jpg") as segment:
        assert segment.size == (24, 24)
    with Image.open(mask_root / "necrosis" / "image.png") as mask:
        assert mask.size == (100, 80)
        assert mask.getbbox() is None
    with (output_root / "segments.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "review"
    assert row["review_reasons"] == "no_plant_mask"
    assert row["prompts_tried"] == "plant;dead plant"
