# Treated versus untreated: symptom-matched evaluation strategy

This report repeats the treated-versus-untreated linear-probe analysis using the same evaluation strategy as the per-symptom learning curves. It exists for a direct methodological comparison and leaves the earlier six-fold balanced-accuracy report unchanged.

## Target and cohort

Every preferred 50 cm image is included. Images in the legacy `healthy_untreated` group are labeled untreated at that image time; images with a positive symptom source label are labeled treated. Untreated does not mean asymptomatic or healthy.

The full, cropped, and segmented representations are tested independently with BioCLIP, BioCLIP 2, BioCLIP 2.5, and DINOv3 ViT-7B/16 embeddings.

## Strategy matched to the symptom probes

- Five stratified folds hold out complete pots.
- Each support set contains equal treated and untreated image counts.
- Support sampling visits distinct pots before reusing another date from a pot.
- Nested supports use 1, 2, 5, 10, 20, 40, and 60 images per class.
- Twenty repeated support draws are evaluated with L2 logistic regression (`C=1`, liblinear solver).
- F1 uses the fixed logit-zero decision boundary.
- Confusion counts are pooled across all five held-out folds within each repeat before F1 is calculated.
- Curves show the mean and 10th-90th percentiles across support draws.

## Interpretation

This remains a treatment/reference classifier, not a symptom classifier. Treatment status is strongly associated with capture date, growth, drought, and other post-treatment changes. Holding out pots prevents repeated-measure leakage but does not remove those shared calendar-time effects.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/analyze_treatment_logistic_matched.py
```

`classifier_trials.csv` retains fold-level metrics and confusion counts. `classifier_metric_summary.csv` contains pooled-fold summaries used in the figure. `evaluation_plan.json` records folds and support selections, while `analysis_manifest.json` records parameters, provenance, and checksums.
