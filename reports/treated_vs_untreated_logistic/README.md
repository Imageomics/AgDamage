# 50 cm treated-versus-untreated linear probe

This analysis asks whether precomputed embeddings of 50 cm images linearly separate herbicide-treated plants from untreated plants. Treatment is an image-time label: images in the legacy `healthy_untreated` folder are untreated, while images carrying a positive symptom folder label are treated. The target makes no claim that untreated plants are healthy and supplies no negative symptom annotations.

## Cohort

Every preferred 50 cm image is included. This retains the pre-treatment baselines as well as later untreated controls and treated observations. The full, cropped, and segmented/plant-only representations are evaluated independently with BioCLIP, BioCLIP 2, BioCLIP 2.5, and DINOv3 ViT-7B/16 embeddings.

The same pot can be untreated at an early capture and treated later. Complete pots—not individual images—are assigned to folds, preventing baseline images and later observations of one pot from crossing between training and test.

## Evaluation

Six folds each hold out one never-treated control pot and a disjoint subset of pots that later received herbicide. Nested support draws contain equal numbers of treated and untreated images. Sampling proceeds in rounds over pots so that new support images come from new pots before additional dates from a previously sampled pot are used.

Each support set trains an L2-regularized logistic regression (`C=1`, liblinear solver). Balanced accuracy is the primary metric; ROC-AUC, average precision, and ordinary accuracy are retained in `classifier_trials.csv`. The plot first averages held-out-fold performance within each repeat, then shows the mean and 10th–90th percentile over support repeats. This avoids presenting the three-observation specificity of one held-out control pot as a smooth confidence interval.

## Projections

Each projection file is a 4-by-2 interactive grid: four embedding models by t-SNE and UMAP. Both methods use cosine distance and fixed random seeds. Points are colored by image-time treatment status. Hovering shows the corresponding image and its pot, date, source label, positive symptom statements, capture role, path, and UUID. Exact coordinates are retained in `projection_coordinates.csv`.

## Interpretation

Most untreated images occur at the early baseline dates, whereas treated images occur later. The classifier can therefore use plant age, plant size, date-associated imaging differences, or water stress in addition to visual effects of herbicide treatment. The result measures separation of the recorded image-time groups; it is not a causal estimate of herbicide injury.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/analyze_treatment_logistic.py
```

`evaluation_plan.json` records folds and support selections. `analysis_manifest.json` records input checksums, parameters, output checksums, and software versions.
