"""Full-resolution plant masks and native-pixel masked crops."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from agdamage.crop import (
    SegmentedDetection,
    SelectedDetection,
    box_area,
    centered_square_box,
    choose_central_detection,
    clip_box,
    evenly_sample_per_class,
    padded_square_box,
)
from agdamage.images import list_images, load_rgb


@dataclass(frozen=True)
class SegmentSettings:
    padding: float = 0.10
    mask_box_padding: float = 0.02
    review_score: float = 0.50
    max_center_distance: float = 0.15
    min_mask_area_fraction: float = 0.0005
    max_mask_area_fraction: float = 0.70
    center_fallback_fraction: float = 0.30
    jpeg_quality: int = 95


MANIFEST_FIELDS = [
    "id",
    "mask_id",
    "label",
    "status",
    "review_reasons",
    "prompt_used",
    "prompts_tried",
    "score",
    "selection_score",
    "detection_count",
    "source_width",
    "source_height",
    "detected_left",
    "detected_top",
    "detected_right",
    "detected_bottom",
    "mask_left",
    "mask_top",
    "mask_right",
    "mask_bottom",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "mask_area_fraction",
    "mask_crop_coverage",
    "crop_area_fraction",
    "center_distance",
    "contains_center",
    "background_red",
    "background_green",
    "background_blue",
    "truncated",
    "output_size_bytes",
    "mask_size_bytes",
]


def border_median_color(
    image: Image.Image, sample_size: int = 96
) -> tuple[int, int, int]:
    """Estimate a robust, shadow-resistant background color from the perimeter."""
    sampled = image.resize((sample_size, sample_size), Image.Resampling.BOX)
    pixels = np.asarray(sampled)
    edge = max(1, sample_size // 16)
    perimeter = np.concatenate(
        [
            pixels[:edge].reshape(-1, 3),
            pixels[-edge:].reshape(-1, 3),
            pixels[edge:-edge, :edge].reshape(-1, 3),
            pixels[edge:-edge, -edge:].reshape(-1, 3),
        ]
    )
    return tuple(int(value) for value in np.median(perimeter, axis=0))


def constrain_mask_to_detection(
    mask: Image.Image,
    box: Sequence[float],
    width: int,
    height: int,
    padding: float,
) -> Image.Image:
    """Remove isolated mask pixels well outside the model's semantic box."""
    if padding < 0:
        raise ValueError("mask box padding must be non-negative")
    left, top, right, bottom = clip_box(box, width, height)
    margin = int(np.ceil(max(right - left, bottom - top) * padding))
    keep_box = (
        max(0, int(np.floor(left)) - margin),
        max(0, int(np.floor(top)) - margin),
        min(width, int(np.ceil(right)) + margin),
        min(height, int(np.ceil(bottom)) + margin),
    )
    constrained = Image.new("L", (width, height), 0)
    constrained.paste(mask.crop(keep_box), keep_box[:2])
    return constrained


def choose_central_segmentation(
    candidates: Sequence[SegmentedDetection],
    width: int,
    height: int,
    mask_box_padding: float = 0.02,
) -> tuple[SegmentedDetection, SelectedDetection] | None:
    """Choose the best central candidate, ignoring predictions with empty masks."""
    ranked = []
    for candidate in candidates:
        mask = constrain_mask_to_detection(
            candidate.mask,
            candidate.detection.box,
            width,
            height,
            mask_box_padding,
        )
        if mask.getbbox() is None:
            continue
        selected = choose_central_detection([candidate.detection], width, height)
        if selected is not None:
            ranked.append(
                (
                    SegmentedDetection(detection=candidate.detection, mask=mask),
                    selected,
                )
            )
    return max(ranked, key=lambda item: item[1].selection_score, default=None)


def save_mask(mask: Image.Image, path: Path) -> None:
    """Atomically save a full-resolution, lossless one-bit mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    mask.convert("1").save(temporary, format="PNG", compress_level=9)
    temporary.replace(path)


def save_segment(
    image: Image.Image,
    mask: Image.Image,
    crop_box: Sequence[int],
    background: tuple[int, int, int],
    path: Path,
    jpeg_quality: int,
) -> None:
    """Save a native-pixel crop with every non-plant pixel made uniform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", image.size, background)
    canvas.paste(image, mask=mask)
    segment = canvas.crop(tuple(crop_box))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        segment.save(
            temporary,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
        )
    elif suffix == ".png":
        segment.save(temporary, format="PNG", compress_level=6)
    else:
        segment.save(temporary, format=image.format or "PNG")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _review_reasons(
    selected: SelectedDetection,
    mask_fraction: float,
    settings: SegmentSettings,
) -> list[str]:
    reasons = []
    if selected.detection.score < settings.review_score:
        reasons.append("low_confidence")
    if selected.center_distance > settings.max_center_distance:
        reasons.append("off_center")
    if mask_fraction < settings.min_mask_area_fraction:
        reasons.append("very_small_mask")
    if mask_fraction > settings.max_mask_area_fraction:
        reasons.append("very_large_mask")
    return reasons


def segment_dataset(
    source_root: Path,
    output_root: Path,
    mask_root: Path,
    paths: Sequence[Path],
    segmenter,
    prompt: str,
    settings: SegmentSettings,
    fallback_prompt: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Materialize plant-only native crops and full-source-resolution masks."""
    started = time.perf_counter()
    rows = []
    output_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(paths, 1):
        image_id = str(source_path.relative_to(source_root))
        mask_id = str(Path(image_id).with_suffix(".png"))
        output_path = output_root / image_id
        mask_path = mask_root / mask_id
        existing = [path for path in (output_path, mask_path) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{existing[0]} exists; pass --overwrite to replace segments"
            )

        image, was_truncated = load_rgb(source_path)
        background = border_median_color(image)
        candidates = segmenter.segment(image, prompt)
        chosen = choose_central_segmentation(
            candidates,
            image.width,
            image.height,
            settings.mask_box_padding,
        )
        prompt_used = prompt
        prompts_tried = [prompt]
        detection_count = len(candidates)

        if fallback_prompt and (
            chosen is None or chosen[1].detection.score < settings.review_score
        ):
            fallback_candidates = segmenter.segment(image, fallback_prompt)
            fallback_chosen = choose_central_segmentation(
                fallback_candidates,
                image.width,
                image.height,
                settings.mask_box_padding,
            )
            prompts_tried.append(fallback_prompt)
            detection_count += len(fallback_candidates)
            if fallback_chosen is not None and (
                chosen is None
                or fallback_chosen[1].selection_score > chosen[1].selection_score
            ):
                chosen = fallback_chosen
                prompt_used = fallback_prompt

        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(
            id=image_id,
            mask_id=mask_id,
            label=source_path.parent.name,
            prompt_used=prompt_used if chosen else "",
            prompts_tried=";".join(prompts_tried),
            detection_count=detection_count,
            source_width=image.width,
            source_height=image.height,
            background_red=background[0],
            background_green=background[1],
            background_blue=background[2],
            truncated=was_truncated,
        )

        if chosen is None:
            mask = Image.new("L", image.size, 0)
            crop_box = centered_square_box(
                image.width, image.height, settings.center_fallback_fraction
            )
            save_mask(mask, mask_path)
            save_segment(
                image,
                mask,
                crop_box,
                background,
                output_path,
                settings.jpeg_quality,
            )
            crop_fraction = box_area(crop_box) / (image.width * image.height)
            row.update(
                status="review",
                review_reasons="no_plant_mask",
                crop_left=crop_box[0],
                crop_top=crop_box[1],
                crop_right=crop_box[2],
                crop_bottom=crop_box[3],
                mask_area_fraction=0.0,
                mask_crop_coverage=0.0,
                crop_area_fraction=round(crop_fraction, 6),
                output_size_bytes=output_path.stat().st_size,
                mask_size_bytes=mask_path.stat().st_size,
            )
            rows.append(row)
            print(
                f"{index:,}/{len(paths):,} review score=n/a mask=empty  {image_id}",
                flush=True,
            )
            continue

        candidate, selected = chosen
        mask = candidate.mask.convert("L").point(lambda value: 255 if value else 0)
        if mask.size != image.size:
            raise ValueError(
                f"mask size {mask.size} does not match image {image.size}: {image_id}"
            )
        mask_box = mask.getbbox()
        if mask_box is None:
            raise AssertionError(
                "empty masks were expected to be filtered during selection"
            )
        crop_box = padded_square_box(
            mask_box, image.width, image.height, settings.padding
        )
        save_mask(mask, mask_path)
        save_segment(
            image,
            mask,
            crop_box,
            background,
            output_path,
            settings.jpeg_quality,
        )

        detected_box = selected.detection.box
        mask_pixels = int(np.count_nonzero(np.asarray(mask)))
        mask_fraction = mask_pixels / (image.width * image.height)
        crop_area = box_area(crop_box)
        reasons = _review_reasons(selected, mask_fraction, settings)
        row.update(
            status="review" if reasons else "ok",
            review_reasons=";".join(reasons),
            score=round(selected.detection.score, 6),
            selection_score=round(selected.selection_score, 6),
            detected_left=round(detected_box[0], 2),
            detected_top=round(detected_box[1], 2),
            detected_right=round(detected_box[2], 2),
            detected_bottom=round(detected_box[3], 2),
            mask_left=mask_box[0],
            mask_top=mask_box[1],
            mask_right=mask_box[2],
            mask_bottom=mask_box[3],
            crop_left=crop_box[0],
            crop_top=crop_box[1],
            crop_right=crop_box[2],
            crop_bottom=crop_box[3],
            mask_area_fraction=round(mask_fraction, 6),
            mask_crop_coverage=round(mask_pixels / crop_area, 6),
            crop_area_fraction=round(crop_area / (image.width * image.height), 6),
            center_distance=round(selected.center_distance, 6),
            contains_center=selected.contains_center,
            output_size_bytes=output_path.stat().st_size,
            mask_size_bytes=mask_path.stat().st_size,
        )
        rows.append(row)
        print(
            f"{index:,}/{len(paths):,} {row['status']:<6} "
            f"score={selected.detection.score:.3f} "
            f"mask={mask_fraction:.3%} crop={crop_box[2] - crop_box[0]}px  "
            f"{image_id}",
            flush=True,
        )

    write_csv(output_root / "segments.csv", rows)
    summary = dict(Counter(row["status"] for row in rows))
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "mask_root": str(mask_root.resolve()),
        "rows_file": "segments.csv",
        "images": len(rows),
        "summary": summary,
        "prompts": {"primary": prompt, "fallback": fallback_prompt},
        "selection": "text segmentation ranked by confidence and image-center geometry",
        "segment": {
            **asdict(settings),
            "mask_resolution": "source",
            "crop_shape": "square",
            "resize": None,
            "background": "median RGB of downsampled source perimeter",
            "mask_cleanup": "clip to padded semantic detection box",
        },
        "detector": segmenter.provenance(),
        "python": platform.python_version(),
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def source_images(root: Path, sample_per_class: int | None = None) -> list[Path]:
    paths = list_images(root)
    if sample_per_class is not None:
        paths = evenly_sample_per_class(paths, sample_per_class)
    return paths
