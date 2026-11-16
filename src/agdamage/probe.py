"""Few-shot sampling plan and linear-SVM probing over embedding bundles."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def ids_digest(ids) -> str:
    digest = sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_plan(ids, labels, shots, repeats, test_fraction, seed):
    """Draw one sampling plan, recorded by image id rather than row position.

    Keyed by id so the very same images can be pulled from any bundle - every
    model, and the full and cropped variants alike - regardless of row order.

    The test split is drawn once and reused everywhere. Support sets are
    nested within a repeat (the 1-shot image is the first of the 5, and so on),
    which removes sampling noise from the shot-to-shot comparison.
    """
    ids = [str(i) for i in ids]
    labels = np.asarray(labels)
    max_shots = max(shots)

    train_ids, test_ids = train_test_split(
        ids, test_size=test_fraction, random_state=seed, stratify=labels
    )
    by_class = {}
    lookup = dict(zip(ids, labels, strict=False))
    for image_id in train_ids:
        by_class.setdefault(str(lookup[image_id]), []).append(image_id)

    short = {c: len(v) for c, v in by_class.items() if len(v) < max_shots}
    if short:
        raise ValueError(f"train pool smaller than {max_shots} shots for {short}")

    supports = []
    for repeat in range(repeats):
        rng = np.random.default_rng([seed, repeat])
        supports.append(
            {
                class_name: [pool[i] for i in rng.permutation(len(pool))[:max_shots]]
                for class_name, pool in sorted(by_class.items())
            }
        )

    return {
        "seed": seed,
        "shots": list(shots),
        "repeats": repeats,
        "test_fraction": test_fraction,
        "classes": sorted(by_class),
        "ids_sha256": ids_digest(ids),
        "test": test_ids,
        "train_pool_sizes": {c: len(v) for c, v in sorted(by_class.items())},
        "supports": supports,
    }


def load_plan(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def save_plan(plan: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan))


def row_index(ids) -> dict[str, int]:
    return {str(v): i for i, v in enumerate(ids)}


def check_plan(plan: dict, ids) -> None:
    """Refuse to evaluate a bundle that cannot supply every planned image."""
    index = row_index(ids)
    needed = set(plan["test"])
    for support in plan["supports"]:
        for chosen in support.values():
            needed.update(chosen)
    missing = needed - index.keys()
    if missing:
        raise ValueError(
            f"bundle is missing {len(missing)} planned images, e.g. {sorted(missing)[:3]}"
        )


def support_rows(plan: dict, repeat: int, shots: int, index: dict[str, int]):
    """Row positions and labels for one repeat's k-shot support set."""
    rows, labels = [], []
    for class_name, chosen in plan["supports"][repeat].items():
        for image_id in chosen[:shots]:
            rows.append(index[image_id])
            labels.append(class_name)
    return np.array(rows), np.array(labels)
