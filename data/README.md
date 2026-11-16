# Local data layout

The contents of this directory are generated and intentionally excluded from Git. The canonical source is [punlalit19/Waterhemp_GreenHouse_Images](https://huggingface.co/datasets/punlalit19/Waterhemp_GreenHouse_Images), downloaded by `agdamage.data.fetch_dataset()`.

Expected local layout:

```text
data/
├── Waterhemp_GreenHouse_Images/             source ImageFolder snapshot
├── Waterhemp_GreenHouse_Images_cropped/     padded central-subject crops
├── Waterhemp_GreenHouse_Images_plant_only/  plant-only cutouts
├── Waterhemp_GreenHouse_Images_plant_masks/ full-resolution lossless masks
├── images/                                  flat byte-identical image copies
├── metadata.csv                             one row per source image
└── embeddings/
    ├── *.npz                                original path-keyed bundles
    └── rekeyed/*.npz                        UUID-keyed derivative bundles
```

Run `python scripts/build_flat_dataset.py` after fetching the source to create `images/` and `metadata.csv`.

## Metadata identities

- `image_id`: UUID parsed from the original filename; stable image key.
- `pot_id`: stable experimental-unit key across dates and camera heights.
- `observation_id`: one pot at one capture date.
- `file_name`: path within the flat release (`images/<filename>.jpg`).
- `original_relative_path`: path within the source ImageFolder snapshot.

The original filename is decomposed into capture date, raw camera code, camera height, three-digit pot label, constant token, modality, UUID, and file extension. `IGPTRIAL1H30C` is a 30 cm retake. If a corresponding `IGPTRIAL1H30` image exists, the retake is marked preferred and both rows are linked by `retake_of_image_id`/`superseded_by_image_id`.

## Partial symptom fields

`chlorosis`, `necrosis`, `epinasty`, and `foliar_distortion` are tri-state in practice:

- `true`: explicitly asserted by the legacy source folder;
- blank: unknown or not assessed;
- `false`: reserved for a future explicit negative annotation and not inferred by the current builder.

The folder `healthy_untreated` means untreated at that image-time, not verified healthy. The `herbicide`, `dosage`, and `group` columns remain blank because those assignments are not available in the source release.

Every row includes an image SHA-256 digest, so a flat copy can be checked against its source independently of paths.
