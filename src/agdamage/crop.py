"""Detection-backed, full-resolution crops for embedding experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from agdamage.images import list_images, load_rgb
from agdamage.runtime import select_device

SAM3_MODEL_ID = "facebook/sam3"
# Pin the checkpoint used for experiments instead of following a moving main branch.
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


@dataclass(frozen=True)
class Detection:
    """One detector result in source-image xyxy coordinates."""

    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class SegmentedDetection:
    """A detector result paired with its full-source-resolution binary mask."""

    detection: Detection
    mask: Image.Image


@dataclass(frozen=True)
class SelectedDetection:
    detection: Detection
    selection_score: float
    center_distance: float
    contains_center: bool


@dataclass(frozen=True)
class CropSettings:
    padding: float = 0.10
    review_score: float = 0.50
    min_area_fraction: float = 0.002
    max_area_fraction: float = 0.70
    center_fallback_fraction: float = 0.30
    jpeg_quality: int = 95


MANIFEST_FIELDS = [
    "id",
    "label",
    "status",
    "crop_method",
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
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "detected_area_fraction",
    "crop_area_fraction",
    "center_distance",
    "contains_center",
    "truncated",
    "output_size_bytes",
]


def clip_box(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Clip an xyxy box to image bounds and reject empty boxes."""
    left, top, right, bottom = (float(value) for value in box)
    left, right = sorted(
        (min(max(0.0, left), float(width)), min(max(0.0, right), float(width)))
    )
    top, bottom = sorted(
        (min(max(0.0, top), float(height)), min(max(0.0, bottom), float(height)))
    )
    if right <= left or bottom <= top:
        raise ValueError(f"empty box after clipping: {box}")
    return left, top, right, bottom


def box_area(box: Sequence[float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def center_distance(box: Sequence[float], width: int, height: int) -> float:
    """Distance between box and image centers, normalized by image dimensions."""
    left, top, right, bottom = box
    x = (left + right) / 2
    y = (top + bottom) / 2
    return math.hypot((x - width / 2) / width, (y - height / 2) / height)


def contains_image_center(box: Sequence[float], width: int, height: int) -> bool:
    left, top, right, bottom = box
    return left <= width / 2 <= right and top <= height / 2 <= bottom


def choose_central_detection(
    detections: Iterable[Detection], width: int, height: int
) -> SelectedDetection | None:
    """Choose a confident detection while strongly preferring the central subject.

    Text prompting does most of the semantic work. This geometry score protects the
    greenhouse images from off-center rulers, camera hardware, and strong shadows.
    """
    ranked = []
    for detection in detections:
        box = clip_box(detection.box, width, height)
        distance = center_distance(box, width, height)
        contains = contains_image_center(box, width, height)
        # Center containment is deliberately worth more than a modest confidence
        # difference; distance breaks ties for boxes near, but not over, the center.
        rank = float(detection.score) + (0.30 if contains else 0.0) - 0.40 * distance
        ranked.append(
            SelectedDetection(
                Detection(box=box, score=float(detection.score)),
                selection_score=rank,
                center_distance=distance,
                contains_center=contains,
            )
        )
    return max(ranked, key=lambda item: item.selection_score, default=None)


def padded_square_box(
    box: Sequence[float], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    """Return an in-bounds square crop around a box, without resizing pixels."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    left, top, right, bottom = clip_box(box, width, height)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    side = math.ceil(max(right - left, bottom - top) * (1 + 2 * padding))
    side = max(1, min(side, width, height))

    crop_left = round(center_x - side / 2)
    crop_top = round(center_y - side / 2)
    crop_left = min(max(0, crop_left), width - side)
    crop_top = min(max(0, crop_top), height - side)
    return crop_left, crop_top, crop_left + side, crop_top + side


def centered_square_box(
    width: int, height: int, fraction: float
) -> tuple[int, int, int, int]:
    """Return a centered square whose side is a fraction of the short dimension."""
    if not 0 < fraction <= 1:
        raise ValueError(
            "center fallback fraction must be greater than zero and at most one"
        )
    side = max(1, round(min(width, height) * fraction))
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def review_reasons(
    selected: SelectedDetection,
    width: int,
    height: int,
    settings: CropSettings,
) -> list[str]:
    fraction = box_area(selected.detection.box) / (width * height)
    reasons = []
    if selected.detection.score < settings.review_score:
        reasons.append("low_confidence")
    if not selected.contains_center:
        reasons.append("off_center")
    if fraction < settings.min_area_fraction:
        reasons.append("very_small")
    if fraction > settings.max_area_fraction:
        reasons.append("very_large")
    return reasons


def evenly_sample_per_class(paths: Sequence[Path], count: int) -> list[Path]:
    """Deterministically sample across each class's sorted acquisition sequence."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.parent.name].append(path)

    selected = []
    for class_name in sorted(groups):
        group = sorted(groups[class_name])
        if count >= len(group):
            selected.extend(group)
            continue
        indices = [
            min(len(group) - 1, math.floor((i + 0.5) * len(group) / count))
            for i in range(count)
        ]
        selected.extend(group[index] for index in indices)
    return sorted(selected)


class Sam3Detector:
    """SAM 3 text-prompted detector with original-coordinate results."""

    def __init__(
        self,
        model_id: str = SAM3_MODEL_ID,
        revision: str = SAM3_REVISION,
        threshold: float = 0.30,
        mask_threshold: float = 0.50,
    ):
        import torch
        import transformers
        from transformers import Sam3Model, Sam3Processor

        self.model_id = model_id
        self.revision = revision
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.device = select_device()
        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )
        self.torch = torch
        self.transformers_version = transformers.__version__

        print(
            f"loading {model_id}@{revision[:12]} on {self.device} ({self.dtype})",
            flush=True,
        )
        self.processor = Sam3Processor.from_pretrained(
            model_id, revision=revision, token=True
        )
        self.model = Sam3Model.from_pretrained(
            model_id,
            revision=revision,
            token=True,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def _instances(self, image: Image.Image, prompt: str):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        del outputs, inputs
        return results

    def detect(self, image: Image.Image, prompt: str) -> list[Detection]:
        results = self._instances(image, prompt)
        boxes = results["boxes"].detach().float().cpu().tolist()
        scores = results["scores"].detach().float().cpu().tolist()
        del results
        return [
            Detection(tuple(box), score)
            for box, score in zip(boxes, scores, strict=True)
        ]

    def segment(self, image: Image.Image, prompt: str) -> list[SegmentedDetection]:
        """Return aligned detections and binary masks at the source dimensions."""
        results = self._instances(image, prompt)
        boxes = results["boxes"].detach().float().cpu().tolist()
        scores = results["scores"].detach().float().cpu().tolist()
        masks = results["masks"].detach().to(self.torch.uint8).cpu().numpy()
        del results
        return [
            SegmentedDetection(
                detection=Detection(tuple(box), score),
                mask=Image.fromarray(mask * 255),
            )
            for box, score, mask in zip(boxes, scores, masks, strict=True)
        ]

    def provenance(self) -> dict:
        gpu = None
        if self.device.type == "cuda":
            gpu = self.torch.cuda.get_device_name(0)
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "threshold": self.threshold,
            "mask_threshold": self.mask_threshold,
            "device": self.device.type,
            "dtype": str(self.dtype).removeprefix("torch."),
            "gpu": gpu,
            "torch": self.torch.__version__,
            "transformers": self.transformers_version,
        }


def save_crop(
    image: Image.Image, box: Sequence[int], path: Path, jpeg_quality: int
) -> None:
    """Atomically save a crop, retaining the source format and relative filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cropped = image.crop(tuple(box))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cropped.save(
            temporary,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
        )
    elif suffix == ".png":
        cropped.save(temporary, format="PNG", compress_level=6)
    else:
        cropped.save(temporary, format=image.format or "PNG")
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def crop_dataset(
    source_root: Path,
    output_root: Path,
    paths: Sequence[Path],
    detector: Sam3Detector,
    prompt: str,
    settings: CropSettings,
    fallback_prompt: str | None = None,
    last_resort_prompt: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Detect and materialize crops; return a complete run manifest."""
    started = time.perf_counter()
    rows = []
    output_root.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(paths, 1):
        image_id = str(source_path.relative_to(source_root))
        output_path = output_root / image_id
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} exists; pass --overwrite to replace crops"
            )

        image, was_truncated = load_rgb(source_path)
        detections = detector.detect(image, prompt)
        selected = choose_central_detection(detections, image.width, image.height)
        prompt_used = prompt
        prompts_tried = [prompt]
        detection_count = len(detections)

        # A broader concept is useful for sparse or almost entirely necrotic plants.
        # It only costs another inference when the primary result is absent or weak.
        if fallback_prompt and (
            selected is None or selected.detection.score < settings.review_score
        ):
            fallback_detections = detector.detect(image, fallback_prompt)
            fallback_selected = choose_central_detection(
                fallback_detections, image.width, image.height
            )
            prompts_tried.append(fallback_prompt)
            detection_count += len(fallback_detections)
            if fallback_selected is not None and (
                selected is None
                or fallback_selected.selection_score > selected.selection_score
            ):
                selected = fallback_selected
                prompt_used = fallback_prompt

        # When herbicide damage has left almost no plant tissue, the pot is the
        # only reliable representation of the central experimental subject.
        if selected is None and last_resort_prompt:
            last_resort_detections = detector.detect(image, last_resort_prompt)
            selected = choose_central_detection(
                last_resort_detections, image.width, image.height
            )
            prompts_tried.append(last_resort_prompt)
            detection_count += len(last_resort_detections)
            if selected is not None:
                prompt_used = last_resort_prompt

        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(
            id=image_id,
            label=source_path.parent.name,
            prompt_used=prompt_used if selected else "",
            prompts_tried=";".join(prompts_tried),
            detection_count=detection_count,
            source_width=image.width,
            source_height=image.height,
            truncated=was_truncated,
        )

        if selected is None:
            if settings.center_fallback_fraction <= 0:
                row.update(
                    status="failed",
                    crop_method="none",
                    review_reasons="no_detection",
                )
                rows.append(row)
                print(f"{index:,}/{len(paths):,} failed  {image_id}", flush=True)
                continue

            crop_box = centered_square_box(
                image.width, image.height, settings.center_fallback_fraction
            )
            save_crop(image, crop_box, output_path, settings.jpeg_quality)
            crop_fraction = box_area(crop_box) / (image.width * image.height)
            row.update(
                status="review",
                crop_method="center_fallback",
                review_reasons="center_fallback",
                crop_left=crop_box[0],
                crop_top=crop_box[1],
                crop_right=crop_box[2],
                crop_bottom=crop_box[3],
                crop_area_fraction=round(crop_fraction, 6),
                center_distance=0.0,
                contains_center=True,
                output_size_bytes=output_path.stat().st_size,
            )
            rows.append(row)
            print(
                f"{index:,}/{len(paths):,} review score=n/a "
                f"crop={crop_box[2] - crop_box[0]}px  {image_id}",
                flush=True,
            )
            continue

        detected_box = selected.detection.box
        crop_box = padded_square_box(
            detected_box, image.width, image.height, settings.padding
        )
        reasons = review_reasons(selected, image.width, image.height, settings)
        save_crop(image, crop_box, output_path, settings.jpeg_quality)

        detected_fraction = box_area(detected_box) / (image.width * image.height)
        crop_fraction = box_area(crop_box) / (image.width * image.height)
        row.update(
            status="review" if reasons else "ok",
            crop_method="detection",
            review_reasons=";".join(reasons),
            score=round(selected.detection.score, 6),
            selection_score=round(selected.selection_score, 6),
            detected_left=round(detected_box[0], 2),
            detected_top=round(detected_box[1], 2),
            detected_right=round(detected_box[2], 2),
            detected_bottom=round(detected_box[3], 2),
            crop_left=crop_box[0],
            crop_top=crop_box[1],
            crop_right=crop_box[2],
            crop_bottom=crop_box[3],
            detected_area_fraction=round(detected_fraction, 6),
            crop_area_fraction=round(crop_fraction, 6),
            center_distance=round(selected.center_distance, 6),
            contains_center=selected.contains_center,
            output_size_bytes=output_path.stat().st_size,
        )
        rows.append(row)
        print(
            f"{index:,}/{len(paths):,} {row['status']:<6} "
            f"score={selected.detection.score:.3f} crop={crop_box[2] - crop_box[0]}px  {image_id}",
            flush=True,
        )

    write_csv(output_root / "crops.csv", rows)
    summary = dict(Counter(row["status"] for row in rows))
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "rows_file": "crops.csv",
        "images": len(rows),
        "summary": summary,
        "prompts": {
            "primary": prompt,
            "fallback": fallback_prompt,
            "last_resort": last_resort_prompt,
        },
        "selection": "text detection ranked by confidence and image-center geometry",
        "crop": {**asdict(settings), "shape": "square", "resize": None},
        "detector": detector.provenance(),
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
