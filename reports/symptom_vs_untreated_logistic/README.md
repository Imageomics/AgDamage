# Positive symptom versus untreated-reference linear probes

This analysis fits four independent binary logistic classifiers: chlorosis, necrosis, epinasty, and foliar distortion. Each classifier uses only explicit positive statements for its symptom. Images carrying another treated symptom label are unknown for that classifier and are excluded from its training and evaluation data; they are never converted into negative examples.

## Reference class

The comparison class is the legacy `healthy_untreated` group. Here this name is interpreted only as **untreated at the time of imaging**, not as a verified absence of visible symptoms. The classifiers therefore measure similarity to a recorded symptom group relative to an untreated reference group. They do not establish symptom presence or absence, and the four outputs are not mutually exclusive.

## Cohort and model arms

Only preferred 50 cm images are used. Full, cropped, and segmented/plant-only representations are analyzed independently with BioCLIP, BioCLIP 2, BioCLIP 2.5, and DINOv3 ViT-7B/16 embeddings.

## Pot-held-out evaluation

Each symptom has its own five-fold split because the positive pots differ by symptom. Complete pots are held out: no timepoint or representation of a test pot is used to train that fold. Within each training fold, support sets contain equal numbers of explicit-positive and untreated-reference images. Sampling visits distinct pots before reusing a pot at another date.

Learning curves stop at 60 support images per class or at the smallest positive training pool across folds, whichever is smaller. The exact endpoint therefore differs by symptom. Each support draw trains L2-regularized logistic regression (`C=1`, liblinear solver), whose objective is binary cross-entropy/log loss.

F1 is the plotted primary metric because the intended use is constructing a positive symptom set, where both precision and recall matter. It is calculated at the fixed logistic threshold of logit 0 (sigmoid 0.5); no test-fold labels are used to tune the threshold. Held-out confusion counts are pooled across the five folds before each repeat's F1, precision, and recall are calculated. Precision and recall are retained separately. Average precision and ROC-AUC provide threshold-free checks, while balanced accuracy, log loss, and ordinary accuracy are also retained in `classifier_trials.csv` and summarized in `classifier_metric_summary.csv`.

## Scores for every image

`image_symptom_scores.csv` contains four symptom outputs for every preferred 50 cm image, representation, and embedding model. These are out-of-fold by pot: the classifier scoring an image was trained without any image from that pot. For each symptom it includes:

- `*_logit`: the unbounded linear score `w x + b`;
- `*_probability`: the sigmoid of that logit;
- `*_control_percentile`: its percentile among out-of-fold untreated-reference logits in the same representation/model arm; and
- `*_above_control_95pct`: an exploratory high-score flag.

Because training uses balanced classes and the untreated reference is not a verified symptom-negative set, the sigmoid values are not calibrated biological probabilities. The percentile and 95th-percentile flag are reference-relative, not symptom diagnoses.

`final_classifiers.npz` stores coefficients and intercepts for all-data models for later inference. Those final models use every known row with balanced class weights; they are distinct from the out-of-fold models used to create the score CSV.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/analyze_symptom_logistic.py
```

`evaluation_plan.json` records folds and nested support selections. `analysis_manifest.json` records label rules, counts, parameters, checksums, and software versions.
