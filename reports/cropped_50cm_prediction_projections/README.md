# Cropped 50 cm symptom-prediction projections

This visualization applies the four independent symptom heads to every preferred cropped 50 cm image and displays the resulting multi-label predictions on the corresponding embedding projections. Each model column uses its matching cropped embedding and classifier arm.

## Two prediction sources

Both scientifically useful views are retained:

- **Pot-held-out OOF:** the existing predictions in `../symptom_vs_untreated_logistic/image_symptom_scores.csv`. Every image was scored by a model whose training set excluded that image's entire pot. This is the appropriate view for assessing predictions on the existing cohort.
- **Final all-data, in-sample:** the saved cropped coefficients in `../symptom_vs_untreated_logistic/final_classifiers.npz` applied to the same cohort. These models saw these pots during training, so the view is expected to be optimistic. It is included to expose, rather than hide, that difference and to demonstrate the saved inference path.

No embeddings are recomputed. The existing cropped 50 cm feature matrices are used directly.

## Two decision thresholds

The selector provides both thresholds for both prediction sources:

- `logit >= 0`, equivalent to sigmoid >= 0.5 and matching the F1 decision rule;
- above the arm- and symptom-specific 95th percentile of pot-held-out untreated-reference logits.

The reference-relative threshold is always defined from the OOF untreated-reference distribution, including when it is applied to the final-model logits. It is not necessarily stricter than logit 0: its numerical value depends on the arm and symptom. The exact cutoff is retained in each `*_control95_logit_threshold` column. Sigmoid values are not calibrated biological probabilities.

## Glyphs and hover

Colored fixed quadrants mean predicted positive under the selected threshold: chlorosis upper-left, necrosis upper-right, epinasty lower-left, and foliar distortion lower-right. White means not selected by that decision rule; it does not establish biological absence.

Hovering retains the cropped image, prediction source, threshold, all four logits, sigmoid scores, untreated-control percentiles, predicted symptom set, and original positive assertions.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/plot_cropped_prediction_projections.py
```

`prediction_scores.csv` contains both prediction sources before projection. `projection_coordinates_and_predictions.csv` is the complete plotted table. `analysis_manifest.json` records input and output hashes, prediction counts, classifier provenance, and threshold definitions.
