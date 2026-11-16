"""Audit an image directory for corruption and decode problems."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageFile

from agdamage.data import DATA_DIR
from agdamage.images import list_images

# JPEG start/end markers; a JPEG missing EOI was cut off mid-write.
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def inspect(path: Path) -> dict:
    """Decode one image every way that matters and record what broke."""
    row = {
        "size_bytes": path.stat().st_size,
        "strict_error": "",
        "tolerant_error": "",
        "verify_error": "",
        "truncated": False,
        "missing_eoi": False,
        "width": "",
        "height": "",
        "mode": "",
        "format": "",
    }

    head = path.read_bytes()
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        row["missing_eoi"] = not (
            head.startswith(JPEG_SOI) and head.rstrip(b"\x00").endswith(JPEG_EOI)
        )

    # verify() catches structural damage without a full decode.
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        row["verify_error"] = f"{type(error).__name__}: {error}"

    # Strict full decode: what the embedding job does by default.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
            row.update(
                width=image.width,
                height=image.height,
                mode=image.mode,
                format=image.format,
            )
    except Exception as error:
        row["strict_error"] = f"{type(error).__name__}: {error}"

    # Tolerant decode: does the partial-decode fallback rescue it?
    if row["strict_error"]:
        row["truncated"] = True
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
                row.update(
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                    format=image.format,
                )
        except Exception as error:
            row["tolerant_error"] = f"{type(error).__name__}: {error}"
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=DATA_DIR / "Waterhemp_GreenHouse_Images"
    )
    parser.add_argument("--out", type=Path, default=Path("reports/image_audit.csv"))
    args = parser.parse_args()

    paths = list_images(args.images)
    print(f"auditing {len(paths):,} images under {args.images}\n")

    rows = []
    for index, path in enumerate(paths, 1):
        row = {"id": str(path.relative_to(args.images)), "label": path.parent.name}
        row.update(inspect(path))
        rows.append(row)
        if index % 100 == 0:
            print(f"  {index:,}/{len(paths):,}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    problems = [
        r for r in rows if r["strict_error"] or r["verify_error"] or r["missing_eoi"]
    ]
    unusable = [r for r in rows if r["tolerant_error"]]
    sizes = {(r["width"], r["height"]) for r in rows if r["width"]}
    modes = {r["mode"] for r in rows if r["mode"]}

    print(f"\n{'=' * 70}")
    print(f"clean            : {len(rows) - len(problems):,}/{len(rows):,}")
    print(f"problematic      : {len(problems)}")
    print(f"unusable         : {len(unusable)}")
    print(f"distinct sizes   : {sorted(sizes)}")
    print(f"distinct modes   : {sorted(modes)}")
    print(f"{'=' * 70}")
    for row in problems:
        print(f"\n{row['id']}  ({row['size_bytes']:,} bytes, label={row['label']})")
        print(f"  missing JPEG EOI : {row['missing_eoi']}")
        print(f"  verify()         : {row['verify_error'] or 'ok'}")
        print(f"  strict decode    : {row['strict_error'] or 'ok'}")
        print(f"  tolerant decode  : {row['tolerant_error'] or 'ok'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
