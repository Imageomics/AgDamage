"""Create UUID-keyed embedding bundles for the flat image dataset."""

import argparse
from pathlib import Path

from agdamage.data import DATA_DIR
from agdamage.rekey import rekey_bundle, verify_rekeyed_bundle

REPRESENTATIONS = {
    "Waterhemp_GreenHouse_Images": "original",
    "Waterhemp_GreenHouse_Images_cropped": "cropped",
    "Waterhemp_GreenHouse_Images_plant_only": "plant_only",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DATA_DIR / "embeddings")
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "embeddings" / "rekeyed",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = []
    for source in sorted(args.embeddings.glob("*.npz")):
        try:
            dataset, model = source.stem.rsplit("__", 1)
            representation = REPRESENTATIONS[dataset]
        except (KeyError, ValueError):
            print(f"skipping unrecognized bundle: {source.name}", flush=True)
            continue
        jobs.append(
            (source, args.out / f"{representation}__{model}.npz", representation)
        )

    if not jobs:
        raise SystemExit(f"No recognized embedding bundles found in {args.embeddings}")

    for index, (source, output, representation) in enumerate(jobs, start=1):
        manifest = rekey_bundle(
            source,
            output,
            args.metadata,
            representation,
            overwrite=args.overwrite,
        )
        verify_rekeyed_bundle(source, output)
        print(
            f"[{index}/{len(jobs)}] {source.name} -> {output.name} "
            f"({manifest['rows']}x{manifest['dimension']}, verified)",
            flush=True,
        )


if __name__ == "__main__":
    main()
