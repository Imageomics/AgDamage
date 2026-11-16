"""Image discovery and decoding shared by independent pipeline stages."""

from pathlib import Path

from PIL import Image, ImageFile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(root: Path | str) -> list[Path]:
    """Return every supported image under ``root`` in stable order."""
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb(path: Path) -> tuple[Image.Image, bool]:
    """Decode RGB, tolerating and reporting source files truncated in transit."""
    try:
        return Image.open(path).convert("RGB"), False
    except OSError:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return Image.open(path).convert("RGB"), True
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
