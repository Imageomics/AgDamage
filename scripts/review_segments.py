"""Create a contact sheet of full-resolution masks and plant-only cutouts."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agdamage.images import load_rgb


def fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, int, int]:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#252525")
    offset = ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2)
    canvas.paste(copy, offset)
    return canvas, offset[0], offset[1]


def evenly_pick(rows, count):
    if count >= len(rows):
        return rows
    return [rows[(index * len(rows)) // count] for index in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/segment_review.jpg"))
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    metadata = json.loads(args.manifest.read_text())
    source_root = Path(metadata["source_root"])
    output_root = Path(metadata["output_root"])
    mask_root = Path(metadata["mask_root"])
    with (args.manifest.parent / metadata["rows_file"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    flagged = sorted(
        (row for row in rows if row["status"] != "ok"), key=lambda row: row["id"]
    )
    ordinary = sorted(
        (row for row in rows if row["status"] == "ok"), key=lambda row: row["id"]
    )
    rows = flagged + evenly_pick(ordinary, max(0, args.max_images - len(flagged)))
    if not rows:
        raise SystemExit("manifest contains no segments")

    tile_width, tile_height = 520, 500
    columns = min(args.columns, len(rows))
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, sheet_rows * tile_height), "#171717"
    )
    font = ImageFont.load_default(size=15)

    for index, row in enumerate(rows):
        source, _ = load_rgb(source_root / row["id"])
        mask = Image.open(mask_root / row["mask_id"]).convert("L")
        source_panel, offset_x, offset_y = fit(source, (500, 340))
        displayed_width = round(
            source.width * min(500 / source.width, 340 / source.height)
        )
        displayed_height = round(
            source.height * min(500 / source.width, 340 / source.height)
        )
        displayed_mask = mask.resize(
            (displayed_width, displayed_height), Image.Resampling.NEAREST
        )
        canvas_mask = Image.new("L", source_panel.size, 0)
        canvas_mask.paste(displayed_mask, (offset_x, offset_y))
        tint = Image.new("RGB", source_panel.size, "#20df80")
        tinted = Image.blend(source_panel, tint, 0.32)
        source_panel.paste(tinted, mask=canvas_mask)

        crop, _ = load_rgb(output_root / row["id"])
        crop_panel, _, _ = fit(crop, (500, 115))
        source_draw = ImageDraw.Draw(source_panel)
        scale = min(500 / source.width, 340 / source.height)
        crop_box = tuple(
            round(float(row[f"crop_{name}"]) * scale + offset)
            for name, offset in zip(
                ("left", "top", "right", "bottom"),
                (offset_x, offset_y, offset_x, offset_y),
                strict=True,
            )
        )
        source_draw.rectangle(crop_box, outline="#ffd23f", width=4)

        x = (index % columns) * tile_width + 10
        y = (index // columns) * tile_height + 10
        sheet.paste(source_panel, (x, y + 32))
        sheet.paste(crop_panel, (x, y + 380))
        draw = ImageDraw.Draw(sheet)
        score = f"{float(row['score']):.3f}" if row["score"] else "n/a"
        title = f"{row['status']}  score={score}  {row['prompt_used'] or 'empty'}"
        draw.text((x, y + 5), title, fill="white", font=font)
        draw.text((x, y + 355), Path(row["id"]).name[:58], fill="#c8c8c8", font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92, optimize=True)
    print(f"wrote {args.out} ({len(rows)} images)")


if __name__ == "__main__":
    main()
