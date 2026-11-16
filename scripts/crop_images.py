"""Detect the central subject and save full-resolution square crops to disk."""

import argparse
import json
from pathlib import Path

from agdamage.crop import (
    SAM3_MODEL_ID,
    SAM3_REVISION,
    CropSettings,
    Sam3Detector,
    crop_dataset,
    source_images,
)
from agdamage.data import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=DATA_DIR / "Waterhemp_GreenHouse_Images"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "Waterhemp_GreenHouse_Images_cropped",
    )
    parser.add_argument("--prompt", default="potted plant")
    parser.add_argument(
        "--fallback-prompt",
        default="plant",
        help="Broader prompt tried only after a missing or low-confidence result",
    )
    parser.add_argument(
        "--last-resort-prompt",
        default="flower pot",
        help="Prompt tried only when neither plant prompt returns a detection",
    )
    parser.add_argument("--model-id", default=SAM3_MODEL_ID)
    parser.add_argument("--revision", default=SAM3_REVISION)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--review-score", type=float, default=0.50)
    parser.add_argument("--min-area-fraction", type=float, default=0.002)
    parser.add_argument("--max-area-fraction", type=float, default=0.70)
    parser.add_argument(
        "--center-fallback-fraction",
        type=float,
        default=0.30,
        help=(
            "Centered square side as a fraction of the short image dimension when "
            "all prompts fail; set to 0 to leave those images failed"
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--sample-per-class",
        type=int,
        help="Deterministic pilot sample from each immediate class directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.review_score <= 1:
        raise SystemExit("thresholds must be between zero and one")
    if not 0 <= args.center_fallback_fraction <= 1:
        raise SystemExit("center fallback fraction must be between zero and one")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("JPEG quality must be between 1 and 100")

    paths = source_images(args.images, args.sample_per_class)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")
    print(f"cropping {len(paths):,} images from {args.images}", flush=True)

    detector = Sam3Detector(args.model_id, args.revision, args.threshold)
    manifest = crop_dataset(
        source_root=args.images,
        output_root=args.out,
        paths=paths,
        detector=detector,
        prompt=args.prompt,
        settings=CropSettings(
            padding=args.padding,
            review_score=args.review_score,
            min_area_fraction=args.min_area_fraction,
            max_area_fraction=args.max_area_fraction,
            center_fallback_fraction=args.center_fallback_fraction,
            jpeg_quality=args.jpeg_quality,
        ),
        fallback_prompt=args.fallback_prompt or None,
        last_resort_prompt=args.last_resort_prompt or None,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["summary"], sort_keys=True), flush=True)
    if manifest["summary"].get("failed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
