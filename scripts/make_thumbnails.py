"""Build a cache of small inline thumbnails, one per image.

Hovering a point should feel instant, so the figure embeds these as data URIs
rather than pulling 3 MB originals from the Hub on every hover.
"""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

from agdamage.data import DATA_DIR
from agdamage.images import list_images, load_rgb


def thumbnail(path: Path, width: int, quality: int) -> str:
    image, _ = load_rgb(path)
    # draft() lets libjpeg downscale during decode — far cheaper than a full
    # 4032x3040 decode followed by a resize.
    image.draft("RGB", (width * 2, width * 2))
    ratio = width / image.width
    image = image.resize((width, max(1, round(image.height * ratio))), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=False)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=DATA_DIR / "Waterhemp_GreenHouse_Images"
    )
    parser.add_argument("--out", type=Path, default=DATA_DIR / "thumbnails.json")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--quality", type=int, default=72)
    args = parser.parse_args()

    paths = list_images(args.images)
    cache = {}
    for index, path in enumerate(paths, 1):
        cache[str(path.relative_to(args.images))] = thumbnail(
            path, args.width, args.quality
        )
        if index % 100 == 0:
            print(f"  {index:,}/{len(paths):,}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cache))
    total = sum(len(v) for v in cache.values())
    print(
        f"\n{len(cache):,} thumbnails, {total / 1e6:.1f} MB of data URIs -> {args.out}"
    )


if __name__ == "__main__":
    main()
