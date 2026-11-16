"""Compute BioCLIP embeddings for one model over a directory of images."""

import argparse
import platform
import time
from pathlib import Path

import torch

from agdamage.data import DATA_DIR
from agdamage.embed import (
    MODEL_SPECS,
    embed_images,
    list_images,
    load_encoder,
    save_bundle,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--images",
        type=Path,
        default=DATA_DIR / "Waterhemp_GreenHouse_Images",
        help="Directory searched recursively for images",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "embeddings")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    # A 7B tower wants a smaller batch than a ViT-B; specs carry their own default.
    batch_size = args.batch_size or spec.get("default_batch_size", 32)
    precision = args.precision or spec.get("default_precision", "fp32")
    out_path = args.out_dir / f"{args.images.name}__{args.model}.npz"

    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to recompute")
        return

    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")

    # Class label is the immediate parent directory (imagefolder layout).
    ids = [str(p.relative_to(args.images)) for p in paths]
    labels = [p.parent.name for p in paths]

    device = select_device()
    print(
        f"{spec['name']} | {len(paths):,} images | {device.type} | {precision} | batch {batch_size}",
        flush=True,
    )

    started = time.perf_counter()
    encoder = load_encoder(args.model, device, precision)
    features, truncated = embed_images(encoder, paths, batch_size)
    elapsed = time.perf_counter() - started

    save_bundle(
        out_path,
        ids,
        labels,
        features,
        {
            "model": {
                "key": args.model,
                "name": spec["name"],
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "tower": "visual",
            },
            "precision": precision,
            "backend": spec["backend"],
            "normalized": True,
            "images": str(args.images),
            "batch_size": batch_size,
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "seconds": round(elapsed, 1),
            # Truncated in the published dataset; decoded partially, not dropped.
            "truncated_images": [str(p.relative_to(args.images)) for p in truncated],
        },
    )
    if truncated:
        print(f"warning: {len(truncated)} truncated image(s) decoded partially:")
        for path in truncated:
            print(f"  {path.relative_to(args.images)}")
    print(
        f"wrote {out_path} ({features.shape[0]}x{features.shape[1]}) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
