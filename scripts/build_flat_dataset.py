"""Copy greenhouse images to data/images and write data/metadata.csv."""

import argparse
from pathlib import Path

from agdamage.data import DATA_DIR
from agdamage.metadata import (
    build_metadata_rows,
    copy_flat_images,
    write_metadata_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "Waterhemp_GreenHouse_Images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Release root containing images/ and metadata.csv (default: project data/)",
    )
    parser.add_argument(
        "--experiment-id",
        default="waterhemp_greenhouse_2026",
    )
    args = parser.parse_args()

    rows = build_metadata_rows(args.source, experiment_id=args.experiment_id)
    print(f"parsed {len(rows):,} images from {args.source}", flush=True)

    def report(index: int, total: int) -> None:
        if index % 25 == 0 or index == total:
            print(f"  copied {index:,}/{total:,}", flush=True)

    copy_flat_images(rows, args.out, progress=report)
    metadata_path = args.out / "metadata.csv"
    write_metadata_csv(rows, metadata_path)
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
