import csv
import json
from pathlib import Path

import numpy as np
import pytest

from agdamage.rekey import (
    ids_digest,
    manifest_path,
    rekey_bundle,
    verify_rekeyed_bundle,
)


def test_manifest_path_is_relative_for_repository_files():
    assert manifest_path(Path(__file__)) == "tests/test_rekey.py"


def write_fixture(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    fieldnames = ["file_name", "image_id", "original_relative_path", "source_label"]
    rows = [
        {
            "file_name": "images/a.jpg",
            "image_id": "11111111-1111-4111-8111-111111111111",
            "original_relative_path": "chlorosis_only/a.jpg",
            "source_label": "chlorosis_only",
        },
        {
            "file_name": "images/b.jpg",
            "image_id": "22222222-2222-4222-8222-222222222222",
            "original_relative_path": "epinasty/b.jpg",
            "source_label": "epinasty",
        },
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    source_path = tmp_path / "source.npz"
    ids = np.asarray([row["original_relative_path"] for row in rows], dtype=np.str_)
    labels = np.asarray([row["source_label"] for row in rows], dtype=np.str_)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    manifest = {
        "ids_sha256": ids_digest(ids),
        "model": {"key": "test"},
        "normalized": True,
    }
    np.savez_compressed(
        source_path,
        ids=ids,
        labels=labels,
        features=features,
        manifest=np.asarray(json.dumps(manifest), dtype=np.str_),
    )
    return metadata_path, source_path, rows, features


def test_rekey_bundle_preserves_features_and_writes_flat_identity(tmp_path):
    metadata_path, source_path, rows, features = write_fixture(tmp_path)
    output_path = tmp_path / "rekeyed.npz"

    manifest = rekey_bundle(source_path, output_path, metadata_path, "original")
    verify_rekeyed_bundle(source_path, output_path)

    with np.load(output_path, allow_pickle=False) as stored:
        assert stored.files == [
            "ids",
            "file_names",
            "original_relative_paths",
            "source_labels",
            "features",
            "manifest",
        ]
        assert stored["ids"].tolist() == [row["image_id"] for row in rows]
        assert stored["file_names"].tolist() == [row["file_name"] for row in rows]
        assert np.array_equal(stored["features"], features)
    assert manifest["representation"] == "original"
    assert manifest["derivation"]["feature_values_unchanged"] is True


def test_rekey_bundle_does_not_overwrite_by_default(tmp_path):
    metadata_path, source_path, _, _ = write_fixture(tmp_path)
    output_path = tmp_path / "rekeyed.npz"
    rekey_bundle(source_path, output_path, metadata_path, "original")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        rekey_bundle(source_path, output_path, metadata_path, "original")
