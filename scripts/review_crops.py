"""Create a contact sheet of source boxes and materialized crops."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agdamage.images import load_rgb


def fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, float]:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#252525")
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas, copy.width / image.width, copy.height / image.height


def scaled_box(row, prefix, scale_x, scale_y, offset_x, offset_y):
    return tuple(
        round(float(row[f"{prefix}_{name}"]) * scale + offset)
        for name, scale, offset in zip(
            ("left", "top", "right", "bottom"),
            (scale_x, scale_y, scale_x, scale_y),
            (offset_x, offset_y, offset_x, offset_y),
            strict=True,
        )
    )


def evenly_pick(rows, count):
    if count >= len(rows):
        return rows
    return [rows[(index * len(rows)) // count] for index in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/crop_review.jpg"))
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    metadata = json.loads(args.manifest.read_text())
    source_root = Path(metadata["source_root"])
    output_root = Path(metadata["output_root"])
    with (args.manifest.parent / metadata["rows_file"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Always include every flagged result, then fill the sheet with a
    # deterministic spread of ordinary cases across the sorted dataset.
    flagged = sorted(
        (row for row in rows if row["status"] != "ok"), key=lambda row: row["id"]
    )
    ordinary = sorted(
        (row for row in rows if row["status"] == "ok"), key=lambda row: row["id"]
    )
    ordinary_slots = max(0, args.max_images - len(flagged))
    rows = flagged + evenly_pick(ordinary, ordinary_slots)
    if not rows:
        raise SystemExit("manifest contains no materialized crops")

    tile_width, tile_height = 520, 500
    columns = min(args.columns, len(rows))
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, sheet_rows * tile_height), "#171717"
    )
    font = ImageFont.load_default(size=15)

    for index, row in enumerate(rows):
        source, _ = load_rgb(source_root / row["id"])
        source_panel, scale_x, scale_y = fit(source, (500, 340))
        if row["status"] == "failed":
            crop_panel = Image.new("RGB", (500, 115), "#252525")
            ImageDraw.Draw(crop_panel).text(
                (210, 48), "NO CROP", fill="#ff756b", font=font
            )
        else:
            crop, _ = load_rgb(output_root / row["id"])
            crop_panel, _, _ = fit(crop, (500, 115))
        source_draw = ImageDraw.Draw(source_panel)
        offset_x = (500 - round(source.width * scale_x)) // 2
        offset_y = (340 - round(source.height * scale_y)) // 2

        if row["status"] != "failed":
            if row.get("crop_method", "detection") == "detection":
                source_draw.rectangle(
                    scaled_box(row, "detected", scale_x, scale_y, offset_x, offset_y),
                    outline="#20df80",
                    width=4,
                )
            source_draw.rectangle(
                scaled_box(row, "crop", scale_x, scale_y, offset_x, offset_y),
                outline="#ffd23f",
                width=4,
            )
        x = (index % columns) * tile_width + 10
        y = (index // columns) * tile_height + 10
        sheet.paste(source_panel, (x, y + 32))
        sheet.paste(crop_panel, (x, y + 380))
        draw = ImageDraw.Draw(sheet)
        score = f"{float(row['score']):.3f}" if row["score"] else "n/a"
        method = row.get("crop_method", "detection")
        title = f"{row['status']}  score={score}  {method}  {row['label']}"
        draw.text((x, y + 5), title, fill="white", font=font)
        draw.text((x, y + 355), Path(row["id"]).name[:58], fill="#c8c8c8", font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92, optimize=True)
    print(f"wrote {args.out} ({len(rows)} images)")


if __name__ == "__main__":
    main()
