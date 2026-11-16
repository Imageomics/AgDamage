# AgDamage

AgDamage is a FloraPalooza/Imageomics workflow for organizing and probing a longitudinal greenhouse image experiment on herbicide injury in waterhemp. It turns the source image-folder release into pot- and time-aware metadata, materializes optional plant crops and masks, computes frozen vision-model embeddings, and evaluates pot-held-out linear classifiers.

The repository contains code and compact report summaries. Images, embeddings, interactive HTML, and large per-trial tables are generated locally and are not stored in Git.

## Data source

The canonical image source is the public Hugging Face dataset [punlalit19/Waterhemp_GreenHouse_Images](https://huggingface.co/datasets/punlalit19/Waterhemp_GreenHouse_Images). The default repository ID is defined once as `agdamage.data.WATERHEMP_GREENHOUSE`; `fetch_dataset()` downloads that dataset, not a similarly named third-party collection. The Hub currently describes the release as an ImageFolder dataset under CC0-1.0.

The analyzed release contains 622 images of 78 pots at 382 pot-by-date observations across six dates. Photographs were made at nominal camera heights of 30 and 50 cm. The filename grammar is:

```text
<date>_<camera-code>_<pot-label>_1_rgb_<image-uuid>.jpg
```

`IGPTRIAL1` denotes 50 cm, `IGPTRIAL1H30` denotes 30 cm, and `IGPTRIAL1H30C` denotes a 30 cm retake. When both 30 cm versions exist for a pot and date, the `H30C` image is preferred and the original remains linked in the metadata for provenance.

### Label semantics

The source folder names are legacy annotations, not mutually exclusive image classes. The metadata exposes four positive-only symptom fields: `chlorosis`, `necrosis`, `epinasty`, and `foliar_distortion`.

`true` means that the source label explicitly asserted the symptom. A blank means unknown/not recorded; it must not be converted to `false`. In particular, the source folder `healthy_untreated` identifies an untreated image-time. It does not establish that the plant was visibly healthy or free of any symptom. Likewise, a folder such as `chlorosis_only` supplies a positive chlorosis statement but is not treated here as a complete negative annotation for the other three fields.

Every pot was photographed repeatedly, and the same pot can appear untreated at a pre-treatment timepoint and treated later. Analyses therefore split by `pot_id`, never randomly by image. Irrigation stopped for all pots after the treatment event, so treatment status is also confounded with calendar time, growth, and water stress. The supplied classifiers are exploratory linear probes, not causal herbicide-injury detectors or calibrated diagnoses.

See [data/README.md](data/README.md) for the flat-release layout and column semantics.

## Installation

Python 3.11 or newer is required. From a fresh clone:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
```

The DINOv3 and SAM 3 stages are GPU-oriented and download large, pinned model checkpoints. SAM 3 is gated on Hugging Face: accept the model terms and make a Hugging Face token available before running crop or segmentation jobs. The Slurm files are templates for OSC-style clusters; pass the appropriate account and partition at submission time.

## Workflow

The maintained workflow keeps metadata, preprocessing, and embeddings as separate reproducible stages:

1. Fetch the original class-folder dataset.
2. Build the flat image release and one-row-per-image metadata CSV.
3. Materialize full-resolution crops and plant-only derivatives.
4. Compute embeddings from each representation and rekey them by image UUID.
5. Run pot-held-out treatment and partial-label symptom analyses.

### Fetch data

Download the exact dataset linked above into `data/Waterhemp_GreenHouse_Images/`:

```bash
python -c 'from agdamage.data import fetch_dataset; print(fetch_dataset())'
```

`fetch_dataset()` follows the current Hub revision unless `revision=` is provided. For a frozen run, first record the current revision with `agdamage.data.dataset_revision()` and pass that SHA to `fetch_dataset`.

Build the flat copy and metadata table:

```bash
python scripts/build_flat_dataset.py
```

This writes `data/images/<original-filename>.jpg` and `data/metadata.csv`. Images are copied without changing their bytes; the CSV retains the original folder path, UUID, SHA-256 digest, pot, date, height, retake links, treatment state, partial symptom assertions, and blank herbicide/dosage/group fields.

### Crop preprocessing

Cropping uses text-prompted SAM 3 segmentation. It selects the candidate most consistent with the center of the image, expands it to a padded square, and saves the crop without resizing. A broader `plant` prompt is tried only when the primary `potted plant` result is absent or weak, helping with severely damaged specimens. If neither plant prompt finds anything, `flower pot` is used as a last resort; this captures the experimental subject even when necrosis has left almost no vegetation. If every prompt fails, a configurable centered square is saved and explicitly marked `center_fallback` for review rather than dropping the image. The square is important: it prevents the embedding model's own square center-crop from discarding part of the subject.

The output mirrors the source class folders and filenames, so IDs remain aligned between original and cropped embedding bundles. `crops.csv` records both the raw detection and final crop coordinates; `manifest.json` pins the detector checkpoint and all crop settings. Low-confidence, off-center, implausibly sized, and geometric fallback crops are saved with `status=review`. The fallback is an explicit crop, not a silent reuse of the full image.

The [Meta SAM 3 checkpoint](https://huggingface.co/facebook/sam3) is gated. Request access by agreeing to its Hugging Face terms once, and make an HF token available before running:

```bash
# Small GPU pilot: two images from each class.
sbatch --cluster=<cluster> --account=<account> --partition=debug \
  slurm/crop_ag_images.sh --sample-per-class 2 \
  --out data/Waterhemp_GreenHouse_Images_cropped_pilot

# After the job, make a source-box/crop contact sheet.
.venv/bin/python scripts/review_crops.py \
  --manifest data/Waterhemp_GreenHouse_Images_cropped_pilot/manifest.json \
  --out reports/crop_review.jpg

# Full crop run after reviewing the pilot.
sbatch --cluster=<cluster> --account=<account> --partition=debug slurm/crop_ag_images.sh
```

Green boxes in the review sheet are model detections; yellow boxes are the padded square crops. Crisp shadows are not used as a foreground heuristic: the text model proposes plant instances, and center geometry is used only to choose among them.

### Plant-only segmentation

Segmentation reuses the pinned SAM 3 checkpoint but is a separate on-disk stage. For every source image it saves a lossless binary mask at the original 4032×3040 resolution under `data/Waterhemp_GreenHouse_Images_plant_masks/`. It also uses the mask bounds to save a padded square cutout under `data/Waterhemp_GreenHouse_Images_plant_only/`. Cutouts retain native pixels and the original relative JPEG IDs; non-plant pixels are replaced with the median perimeter color so shadows and table texture cannot reach the embedding model. Tiny mask fragments outside the model's padded semantic detection box are removed before the native crop is calculated, preventing isolated prediction noise from making the plant small again.

The primary text prompt is `plant`, with `dead plant` tried only for missing or weak results. Pot prompts are deliberately excluded. If no visible plant can be segmented, the lossless mask is empty and a neutral cutout is saved with `status=review`, preserving row alignment without representing the pot as plant.

```bash
# Stratified pilot: two images per damage class.
sbatch --cluster=<cluster> --account=<account> --partition=debug \
  slurm/segment_ag_images.sh --sample-per-class 2 \
  --out data/Waterhemp_GreenHouse_Images_plant_only_pilot \
  --masks data/Waterhemp_GreenHouse_Images_plant_masks_pilot

# Inspect the mask overlay and embedding-ready cutout together.
.venv/bin/python scripts/review_segments.py \
  --manifest data/Waterhemp_GreenHouse_Images_plant_only_pilot/manifest.json \
  --out reports/segment_review.jpg

# Full materialization after reviewing the pilot.
sbatch --cluster=<cluster> --account=<account> --partition=debug slurm/segment_ag_images.sh
```

### Embeddings

All configured embedding models accept the same image-root argument. The output filename includes the root directory name, keeping original and cropped experiments separate.

```bash
# One model locally (GPU when available).
.venv/bin/python scripts/compute_embeddings.py \
  --model bioclip-2.5 \
  --images data/Waterhemp_GreenHouse_Images_cropped

# Or submit the existing model array and forward the image argument.
sbatch --cluster=<cluster> --account=<account> --partition=debug slurm/embed_ag_images.sh \
  --images data/Waterhemp_GreenHouse_Images_cropped

# Cache cropped hover thumbnails and build a report beside the original one.
.venv/bin/python scripts/make_thumbnails.py \
  --images data/Waterhemp_GreenHouse_Images_cropped \
  --out data/thumbnails_cropped.json
sbatch --cluster=<cluster> --account=<account> --partition=debug slurm/plot_ag_embeddings.sh \
  --dataset Waterhemp_GreenHouse_Images_cropped \
  --thumbnails data/thumbnails_cropped.json \
  --out reports/embedding_projections_cropped.html \
  --title "Embeddings of cropped waterhemp greenhouse images"
```

Four pinned frozen encoders are configured in `agdamage.embed.MODEL_SPECS`: BioCLIP, BioCLIP 2, BioCLIP 2.5, and DINOv3 ViT-7B/16. Repeat embedding for the full, cropped, and plant-only roots. The bundles initially use the original relative path as their row ID. After `data/metadata.csv` exists, create verified derivative copies keyed by the image UUID used in the flat release:

```bash
python scripts/rekey_embeddings.py
```

Rekeying changes identifiers and provenance only; it verifies that feature values are unchanged. Current analyses read `data/embeddings/rekeyed/{full,cropped,plant_only}__<model>.npz`.

## Analyses

The maintained analysis entry points use preferred 50 cm images and compare the full, cropped, and segmented representations independently. All evaluation splits hold out complete pots.

| Analysis | Command | Main report directory |
|---|---|---|
| Treated versus untreated, original longitudinal strategy and projections | `python scripts/analyze_treatment_logistic.py` | `reports/treated_vs_untreated_logistic/` |
| Treated versus untreated, strategy matched to symptom probes | `python scripts/analyze_treatment_logistic_matched.py` | `reports/treated_vs_untreated_logistic_matched/` |
| Four positive-symptom versus untreated-reference heads | `python scripts/analyze_symptom_logistic.py` | `reports/symptom_vs_untreated_logistic/` |
| Positive-assertion projection glyphs | `python scripts/plot_positive_assertion_projections.py` | `reports/positive_assertion_projections/` |
| Cropped-image OOF and all-data prediction projections | `python scripts/plot_cropped_prediction_projections.py` | `reports/cropped_50cm_prediction_projections/` |

The symptom heads optimize binary logistic loss (cross-entropy/log loss) and report F1 at the fixed logit-zero boundary. For an existing cohort image, use the pot-held-out out-of-fold scores; use the saved all-data classifiers only for genuinely new pots. Sigmoid outputs are not calibrated biological probabilities because the training classes were balanced and the reference class was untreated rather than verified symptom-negative.

Each analysis directory has its own README with the exact cohort, split, threshold, and interpretation. [reports/README.md](reports/README.md) indexes the report products and explains which large outputs are regenerated locally.

## Tests and code quality

```bash
pytest
ruff check .
ruff format --check .
```

Tests cover filename parsing, flat metadata and retake rules, crops and masks, embedding rekeying, pot-held-out evaluation plans, partial-label behavior, and prediction-table construction.

## Repository layout

```text
src/agdamage/   reusable data, preprocessing, embedding, and analysis code
scripts/        command-line entry points
slurm/          site-adjustable batch templates
tests/          unit tests
data/           local/generated data (ignored except for its README)
reports/        analysis documentation and compact, reviewable summaries
```

## AI disclosure

This repository was built using Codex with GPT-5.6-Sol.
