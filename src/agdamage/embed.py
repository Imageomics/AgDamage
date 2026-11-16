"""Image embeddings from the BioCLIP model family."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import snapshot_download

# Re-export these helpers for compatibility with existing embedding scripts while
# keeping their implementation independent of OpenCLIP.
from agdamage.images import list_images as list_images  # noqa: PLC0414
from agdamage.images import load_rgb as load_rgb  # noqa: PLC0414
from agdamage.runtime import select_device as select_device  # noqa: PLC0414

# Checkpoints pinned to the revisions used by the Sage 2026 BioCLIP lesson, so
# embeddings computed here stay comparable with that material.
MODEL_SPECS = {
    "bioclip": {
        "name": "BioCLIP",
        "repo_id": "imageomics/bioclip",
        "revision": "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
        "backend": "open_clip",
        "weight_file": "open_clip_pytorch_model.bin",
        "dimension": 512,
    },
    "bioclip-2": {
        "name": "BioCLIP 2",
        "repo_id": "imageomics/bioclip-2",
        "revision": "2957b322090f9cb17ae72c71981c7218a28d81e0",
        "backend": "open_clip",
        "weight_file": "open_clip_model.safetensors",
        "dimension": 768,
    },
    "bioclip-2.5": {
        "name": "BioCLIP 2.5",
        "repo_id": "imageomics/bioclip-2.5-vith14",
        "revision": "191d741545e4c741cdef4b22c6eb69c945c1e592",
        "backend": "open_clip",
        "weight_file": "open_clip_model.safetensors",
        "dimension": 1024,
    },
    # Self-supervised ViT rather than a CLIP tower: no text side, and the image
    # embedding is the CLS token instead of a contrastive projection.
    "dinov3-vit7b16": {
        "name": "DINOv3 ViT-7B/16",
        "repo_id": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "revision": "b80367753773648a6793235ab9c65cdbb029506f",
        "backend": "transformers",
        "dimension": 4096,
        "default_precision": "bf16",
        "default_batch_size": 16,
    },
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Encoder:
    """One image tower behind a single interface, whatever library loaded it.

    `preprocess` turns a PIL image into a model-ready tensor; `encode` turns a
    stacked batch into L2-normalized embeddings on the CPU.
    """

    def __init__(self, preprocess, encode, precision: str):
        self.preprocess = preprocess
        self.encode = encode
        self.precision = precision


def _load_open_clip(spec, device, precision) -> Encoder:
    snapshot = snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        allow_patterns=["open_clip_config.json", spec["weight_file"]],
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"local-dir:{snapshot}", device=device, precision=precision
    )
    model.eval()

    def encode(batch):
        return model.encode_image(batch.to(device), normalize=True).float().cpu()

    return Encoder(preprocess, encode, precision)


def _load_transformers(spec, device, precision) -> Encoder:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        spec["repo_id"], revision=spec["revision"]
    )
    model = AutoModel.from_pretrained(
        spec["repo_id"], revision=spec["revision"], dtype=DTYPES[precision]
    ).to(device)
    model.eval()

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    def encode(batch):
        out = model(pixel_values=batch.to(device, DTYPES[precision]))
        # CLS token is DINOv3's image-level descriptor; normalize so every
        # bundle is comparable under cosine distance.
        features = out.pooler_output.float()
        return torch.nn.functional.normalize(features, dim=-1).cpu()

    return Encoder(preprocess, encode, precision)


def load_encoder(
    key: str, device: torch.device, precision: str | None = None
) -> Encoder:
    spec = MODEL_SPECS[key]
    precision = precision or spec.get("default_precision", "fp32")
    if spec["backend"] == "open_clip":
        return _load_open_clip(spec, device, precision)
    return _load_transformers(spec, device, precision)


def embed_images(
    encoder: Encoder,
    paths: list[Path],
    batch_size: int = 32,
) -> tuple[np.ndarray, list[Path]]:
    """L2-normalized image embeddings in the order given, plus truncated files."""
    batches = []
    truncated = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = []
            for path in chunk:
                image, was_truncated = load_rgb(path)
                if was_truncated:
                    truncated.append(path)
                images.append(encoder.preprocess(image))
            batches.append(encoder.encode(torch.stack(images)))
            print(f"{min(start + batch_size, len(paths)):,}/{len(paths):,}", flush=True)
    return torch.cat(batches).numpy(), truncated


def ids_digest(ids) -> str:
    digest = sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_bundle(path: Path | str, ids, labels, features: np.ndarray, manifest: dict):
    """Write ids, labels, embeddings, and provenance to a single npz."""
    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Embeddings contain non-finite values")
    ids = np.asarray(ids, dtype=np.str_)

    record = dict(manifest)
    record.update(
        {
            "rows": len(ids),
            "dimension": int(features.shape[1]),
            "ids_sha256": ids_digest(ids),
        }
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=ids,
        labels=np.asarray(labels, dtype=np.str_),
        features=features,
        manifest=np.asarray(json.dumps(record, sort_keys=True), dtype=np.str_),
    )


def load_bundle(path: Path | str) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {
            "ids": stored["ids"],
            "labels": stored["labels"],
            "features": stored["features"],
            "manifest": json.loads(str(stored["manifest"].item())),
        }
