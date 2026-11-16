"""Save full-resolution plant masks and native-pixel plant-only crops."""

import argparse
import json
from pathlib import Path

from agdamage.crop import SAM3_MODEL_ID, SAM3_REVISION, Sam3Detector
from agdamage.data import DATA_DIR
from agdamage.segment import SegmentSettings, segment_dataset, source_images


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=DATA_DIR / "Waterhemp_GreenHouse_Images"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "Waterhemp_GreenHouse_Images_plant_only",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=DATA_DIR / "Waterhemp_GreenHouse_Images_plant_masks",
    )
    parser.add_argument("--prompt", default="plant")
    parser.add_argument(
        "--fallback-prompt",
        default="dead plant",
        help="Prompt tried only after a missing or low-confidence plant mask",
    )
    parser.add_argument("--model-id", default=SAM3_MODEL_ID)
    parser.add_argument("--revision", default=SAM3_REVISION)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--mask-box-padding", type=float, default=0.02)
    parser.add_argument("--review-score", type=float, default=0.50)
    parser.add_argument("--max-center-distance", type=float, default=0.15)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.0005)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.70)
    parser.add_argument("--center-fallback-fraction", type=float, default=0.30)
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
    unit_values = {
        "threshold": args.threshold,
        "mask threshold": args.mask_threshold,
        "review score": args.review_score,
        "maximum center distance": args.max_center_distance,
        "minimum mask area fraction": args.min_mask_area_fraction,
        "maximum mask area fraction": args.max_mask_area_fraction,
        "center fallback fraction": args.center_fallback_fraction,
    }
    invalid = [name for name, value in unit_values.items() if not 0 <= value <= 1]
    if invalid:
        raise SystemExit(f"values must be between zero and one: {', '.join(invalid)}")
    if args.center_fallback_fraction == 0:
        raise SystemExit("center fallback fraction must be greater than zero")
    if args.padding < 0:
        raise SystemExit("padding must be non-negative")
    if args.mask_box_padding < 0:
        raise SystemExit("mask box padding must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("JPEG quality must be between 1 and 100")

    paths = source_images(args.images, args.sample_per_class)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")
    print(f"segmenting {len(paths):,} images from {args.images}", flush=True)

    segmenter = Sam3Detector(
        args.model_id,
        args.revision,
        args.threshold,
        args.mask_threshold,
    )
    manifest = segment_dataset(
        source_root=args.images,
        output_root=args.out,
        mask_root=args.masks,
        paths=paths,
        segmenter=segmenter,
        prompt=args.prompt,
        fallback_prompt=args.fallback_prompt or None,
        settings=SegmentSettings(
            padding=args.padding,
            mask_box_padding=args.mask_box_padding,
            review_score=args.review_score,
            max_center_distance=args.max_center_distance,
            min_mask_area_fraction=args.min_mask_area_fraction,
            max_mask_area_fraction=args.max_mask_area_fraction,
            center_fallback_fraction=args.center_fallback_fraction,
            jpeg_quality=args.jpeg_quality,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
