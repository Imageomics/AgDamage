"""Losslessly re-key embedding bundles against the flat image manifest."""

import csv
import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

REQUIRED_METADATA_COLUMNS = {
    "file_name",
    "image_id",
    "original_relative_path",
    "source_label",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def manifest_path(path: Path | str) -> str:
    """Represent repository-local manifest paths relative to the project root."""
    value = Path(path)
    try:
        return value.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return value.as_posix()


def ids_digest(ids) -> str:
    """Hash an ordered sequence of string identifiers."""
    digest = sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order values."""
    values = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata_index(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Index flat metadata rows by the class-relative legacy image path."""
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_METADATA_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"metadata is missing required columns: {names}")
        rows = list(reader)

    index = {}
    image_ids = set()
    file_names = set()
    for row in rows:
        legacy_id = row["original_relative_path"]
        image_id = row["image_id"]
        file_name = row["file_name"]
        if not legacy_id or not image_id or not file_name:
            raise ValueError("metadata identity columns cannot be blank")
        if legacy_id in index:
            raise ValueError(f"duplicate original_relative_path: {legacy_id}")
        if image_id in image_ids:
            raise ValueError(f"duplicate image_id: {image_id}")
        if file_name in file_names:
            raise ValueError(f"duplicate file_name: {file_name}")
        if not file_name.startswith("images/"):
            raise ValueError(f"file_name is not under images/: {file_name}")
        index[legacy_id] = row
        image_ids.add(image_id)
        file_names.add(file_name)
    return index


def rekey_bundle(
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    representation: str,
    *,
    overwrite: bool = False,
) -> dict:
    """Copy one bundle with UUID IDs while preserving feature values exactly."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass overwrite=True to replace it"
        )

    metadata = load_metadata_index(metadata_path)
    with np.load(source_path, allow_pickle=False) as stored:
        legacy_ids = np.asarray(stored["ids"], dtype=np.str_)
        legacy_labels = np.asarray(stored["labels"], dtype=np.str_)
        features = np.array(stored["features"], copy=True)
        source_manifest = json.loads(str(stored["manifest"].item()))

    if features.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, got shape {features.shape}")
    if len(legacy_ids) != len(legacy_labels) or len(legacy_ids) != len(features):
        raise ValueError("source ids, labels, and feature rows have different lengths")
    if len(set(legacy_ids.tolist())) != len(legacy_ids):
        raise ValueError("source bundle contains duplicate ids")

    source_id_set = set(legacy_ids.tolist())
    missing = source_id_set.difference(metadata)
    extra = set(metadata).difference(source_id_set)
    if missing or extra:
        raise ValueError(
            "bundle and metadata identities differ: "
            f"{len(missing)} missing from metadata, {len(extra)} missing from bundle"
        )

    rows = [metadata[legacy_id] for legacy_id in legacy_ids]
    image_ids = np.asarray([row["image_id"] for row in rows], dtype=np.str_)
    file_names = np.asarray([row["file_name"] for row in rows], dtype=np.str_)
    metadata_labels = np.asarray([row["source_label"] for row in rows], dtype=np.str_)
    if not np.array_equal(legacy_labels, metadata_labels):
        raise ValueError(
            "source bundle labels do not match metadata source_label values"
        )

    source_ids_sha256 = ids_digest(legacy_ids)
    recorded_source_digest = source_manifest.get("ids_sha256")
    if recorded_source_digest and source_ids_sha256 != recorded_source_digest:
        raise ValueError("source bundle ids do not match its manifest checksum")

    features_sha256 = array_digest(features)
    manifest = {
        "schema": "agdamage.embedding_bundle",
        "schema_version": 2,
        "rows": len(image_ids),
        "dimension": int(features.shape[1]),
        "model": source_manifest.get("model"),
        "normalized": source_manifest.get("normalized"),
        "representation": representation,
        "ids_sha256": ids_digest(image_ids),
        "features_sha256": features_sha256,
        "identity": {
            "ids": "metadata.csv:image_id",
            "file_names": "metadata.csv:file_name",
            "original_relative_paths": "metadata.csv:original_relative_path",
            "source_labels": "metadata.csv:source_label (provenance only)",
        },
        "derivation": {
            "operation": "rekey_only_no_embedding_recomputation",
            "source_bundle": source_path.name,
            "source_bundle_sha256": file_digest(source_path),
            "source_ids_sha256": source_ids_sha256,
            "metadata_file": metadata_path.name,
            "metadata_sha256": file_digest(metadata_path),
            "feature_values_unchanged": True,
        },
        "source_manifest": source_manifest,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                ids=image_ids,
                file_names=file_names,
                original_relative_paths=legacy_ids,
                source_labels=legacy_labels,
                features=features,
                manifest=np.asarray(
                    json.dumps(manifest, sort_keys=True), dtype=np.str_
                ),
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest


def verify_rekeyed_bundle(source_path: Path, derived_path: Path) -> None:
    """Assert that re-keying changed identity metadata but not feature values."""
    with np.load(source_path, allow_pickle=False) as source:
        source_features = source["features"]
        source_ids = source["ids"]
        source_labels = source["labels"]
        with np.load(derived_path, allow_pickle=False) as derived:
            if not np.array_equal(source_features, derived["features"]):
                raise ValueError(f"feature values changed in {derived_path}")
            if not np.array_equal(source_ids, derived["original_relative_paths"]):
                raise ValueError(f"legacy ids changed in {derived_path}")
            if not np.array_equal(source_labels, derived["source_labels"]):
                raise ValueError(f"source labels changed in {derived_path}")
            manifest = json.loads(str(derived["manifest"].item()))
            if manifest["features_sha256"] != array_digest(derived["features"]):
                raise ValueError(f"feature checksum mismatch in {derived_path}")
            if manifest["ids_sha256"] != ids_digest(derived["ids"]):
                raise ValueError(f"id checksum mismatch in {derived_path}")
