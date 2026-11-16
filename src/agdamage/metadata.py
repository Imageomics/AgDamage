"""Metadata and flat-release helpers for the greenhouse image dataset."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image

from agdamage.images import list_images

SYMPTOMS = (
    "chlorosis",
    "necrosis",
    "epinasty",
    "foliar_distortion",
)

LABEL_MAP: dict[str, tuple[bool | None, ...]] = {
    "chlorosis_necrosis": (True, True, None, None),
    "chlorosis_only": (True, None, None, None),
    "epinasty": (None, None, True, None),
    "foliar_distortion": (None, None, None, True),
    "healthy_untreated": (None, None, None, None),
    "necrosis_only": (None, True, None, None),
}

CAMERA_MAP = {
    "IGPTRIAL1": (50, "standard"),
    "IGPTRIAL1H30": (30, "standard"),
    "IGPTRIAL1H30C": (30, "retake"),
}

METADATA_COLUMNS = (
    "file_name",
    "image_id",
    "sha256",
    "original_filename",
    "original_relative_path",
    "source_label",
    "capture_date",
    "timepoint_index",
    "camera_code_raw",
    "camera_height_cm",
    "pot_label",
    "constant_token",
    "modality",
    "file_extension",
    "width_px",
    "height_px",
    "size_bytes",
    "experiment_id",
    "pot_id",
    "observation_id",
    "capture_role",
    "retake_of_image_id",
    "superseded_by_image_id",
    "is_preferred_for_height",
    "herbicide_treated",
    "herbicide",
    "dosage",
    "group",
    *SYMPTOMS,
    "label_source",
    "label_version",
)

_FILENAME_PATTERN = re.compile(
    r"^(?P<capture_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<camera_code>IGPTRIAL1(?:H30C|H30)?)_"
    r"(?P<pot_label>\d{3})_"
    r"(?P<constant_token>\d+)_"
    r"(?P<modality>[A-Za-z0-9]+)_"
    r"(?P<image_id>[0-9a-fA-F-]{36})\."
    r"(?P<file_extension>jpe?g)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFilename:
    """Fields encoded in one source image filename."""

    capture_date: date
    camera_code_raw: str
    camera_height_cm: int
    pot_label: str
    constant_token: int
    modality: str
    image_id: str
    file_extension: str
    capture_role: str


def parse_filename(path: Path | str) -> ParsedFilename:
    """Parse and validate the fixed greenhouse image filename grammar."""
    name = Path(path).name
    match = _FILENAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unrecognized greenhouse image filename: {name}")

    values = match.groupdict()
    capture_date = date.fromisoformat(values["capture_date"])
    camera_code = values["camera_code"].upper()
    height_cm, capture_role = CAMERA_MAP[camera_code]
    constant_token = int(values["constant_token"])
    if constant_token != 1:
        raise ValueError(f"expected constant filename token 1 in {name}")

    modality = values["modality"].lower()
    if modality != "rgb":
        raise ValueError(f"expected rgb modality in {name}")

    image_id = str(uuid.UUID(values["image_id"]))
    return ParsedFilename(
        capture_date=capture_date,
        camera_code_raw=camera_code,
        camera_height_cm=height_cm,
        pot_label=values["pot_label"],
        constant_token=constant_token,
        modality=modality,
        image_id=image_id,
        file_extension=values["file_extension"].lower(),
        capture_role=capture_role,
    )


def symptom_values(source_label: str) -> dict[str, bool | None]:
    """Translate a legacy folder label into four partial symptom labels."""
    try:
        values = LABEL_MAP[source_label]
    except KeyError as error:
        raise ValueError(f"unrecognized source label: {source_label}") from error
    return dict(zip(SYMPTOMS, values, strict=True))


def _assign_retake_links(rows: list[dict]) -> None:
    by_observation: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["camera_height_cm"] == 30:
            key = (row["capture_date"], row["pot_label"])
            by_observation[key][row["capture_role"]].append(row)

    for key, variants in by_observation.items():
        standards = variants["standard"]
        retakes = variants["retake"]
        if len(standards) > 1 or len(retakes) > 1:
            raise ValueError(f"ambiguous 30 cm captures for date/pot {key}")
        if not retakes:
            continue

        retake = retakes[0]
        retake["is_preferred_for_height"] = True
        if standards:
            standard = standards[0]
            standard["is_preferred_for_height"] = False
            standard["superseded_by_image_id"] = retake["image_id"]
            retake["retake_of_image_id"] = standard["image_id"]


def build_metadata_rows(
    source_root: Path | str,
    experiment_id: str = "waterhemp_greenhouse_2026",
) -> list[dict]:
    """Build deterministic metadata rows without copying or modifying images."""
    source_root = Path(source_root)
    paths = list_images(source_root)
    if not paths:
        raise ValueError(f"no images found under {source_root}")

    parsed = [(path, parse_filename(path)) for path in paths]
    capture_dates = sorted({fields.capture_date for _, fields in parsed})
    timepoint_by_date = {value: index for index, value in enumerate(capture_dates)}

    rows = []
    filenames: set[str] = set()
    image_ids: set[str] = set()
    for path, fields in parsed:
        relative_path = path.relative_to(source_root)
        if len(relative_path.parts) != 2:
            raise ValueError(
                f"expected <label>/<image> source path, got {relative_path}"
            )
        source_label = relative_path.parts[0]
        if path.name in filenames:
            raise ValueError(f"duplicate filename cannot be flattened: {path.name}")
        if fields.image_id in image_ids:
            raise ValueError(f"duplicate image UUID: {fields.image_id}")
        filenames.add(path.name)
        image_ids.add(fields.image_id)

        with Image.open(path) as image:
            width, height = image.size

        capture_date = fields.capture_date.isoformat()
        pot_id = f"{experiment_id}_p{fields.pot_label}"
        row = {
            "file_name": f"images/{path.name}",
            "image_id": fields.image_id,
            "sha256": "",
            "original_filename": path.name,
            "original_relative_path": relative_path.as_posix(),
            "source_label": source_label,
            "capture_date": capture_date,
            "timepoint_index": timepoint_by_date[fields.capture_date],
            "camera_code_raw": fields.camera_code_raw,
            "camera_height_cm": fields.camera_height_cm,
            "pot_label": fields.pot_label,
            "constant_token": fields.constant_token,
            "modality": fields.modality,
            "file_extension": fields.file_extension,
            "width_px": width,
            "height_px": height,
            "size_bytes": path.stat().st_size,
            "experiment_id": experiment_id,
            "pot_id": pot_id,
            "observation_id": f"{pot_id}_{capture_date}",
            "capture_role": fields.capture_role,
            "retake_of_image_id": "",
            "superseded_by_image_id": "",
            "is_preferred_for_height": True,
            "herbicide_treated": source_label != "healthy_untreated",
            "herbicide": "",
            "dosage": "",
            "group": "",
            **symptom_values(source_label),
            "label_source": "source_folder",
            "label_version": "positive_only_legacy_folder_v2",
            "_source_path": path,
        }
        rows.append(row)

    _assign_retake_links(rows)
    return rows


def _copy_and_hash(source: Path, destination: Path) -> str:
    """Copy one file atomically while calculating its SHA-256 digest."""
    if destination.exists():
        source_digest = _sha256(source)
        if source_digest != _sha256(destination):
            raise ValueError(f"existing flat image differs from source: {destination}")
        return source_digest

    partial = destination.with_name(f".{destination.name}.partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_handle, partial.open("xb") as output_handle:
            while chunk := input_handle.read(8 * 1024 * 1024):
                output_handle.write(chunk)
                digest.update(chunk)
        shutil.copystat(source, partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_flat_images(
    rows: list[dict],
    output_root: Path | str,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Copy every row's source image into ``<output_root>/images``."""
    output_root = Path(output_root)
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    for index, row in enumerate(rows, 1):
        destination = output_root / row["file_name"]
        row["sha256"] = _copy_and_hash(row["_source_path"], destination)
        if progress is not None:
            progress(index, total)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_metadata_csv(rows: list[dict], output_path: Path | str) -> None:
    """Write the public metadata columns with blanks for unknown values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial")
    try:
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: _csv_value(row[column]) for column in METADATA_COLUMNS}
                )
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
