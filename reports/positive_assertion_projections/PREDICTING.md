# Running the binary heads on an image

Mechanics for turning a JPEG into four symptom scores using the stored classifiers. Read the caveats at the end before reporting any number produced this way.

## What is stored

`../symptom_vs_untreated_logistic/final_classifiers.npz` holds 96 arrays — one `coef` and one `intercept` for each of 4 symptoms × 3 representations × 4 encoders. Keys are `{representation}__{model_key}__{symptom}__{coef|intercept}`, for example `segmented__bioclip__chlorosis__coef`. Each `coef` is shape `(1, D)` where `D` is that encoder's dimension.

These are **all-data** models: fit on every row where the symptom is known, with `class_weight="balanced"`. They are not the out-of-fold models used to build `image_symptom_scores.csv`, so do not score a training image with them and treat the result as held out.

## The four steps

**1 · Match the representation.** A classifier trained on `segmented` features only makes sense on a segmented image. Produce the same representation the arm was trained on:

- `full` — the image as captured, nothing to do
- `cropped` — `scripts/crop_images.py` (`agdamage.crop`)
- `segmented` — `scripts/segment_images.py` (`agdamage.segment`)

**2 · Encode with the matching frozen encoder.** `MODEL_SPECS` in `agdamage.embed` pins the repo and revision, so a rerun reproduces the same vector. `embed_images` returns **L2-normalized** float32 rows — every stored feature has ‖x‖ = 1, and the coefficients were fit on that scale. Do not skip the normalization or feed in raw logits from another pipeline.

```python
import numpy as np, torch
from pathlib import Path
from agdamage.embed import load_encoder, embed_images

REP, MODEL = "segmented", "bioclip"  # must be a matched pair
paths = [Path("path/to/segmented_image.jpg")]

encoder = load_encoder(MODEL, torch.device("cuda"))
features, truncated = embed_images(encoder, paths)  # (n, D), L2-normalized

z = np.load("reports/symptom_vs_untreated_logistic/final_classifiers.npz")
for symptom in ["chlorosis", "necrosis", "epinasty", "foliar_distortion"]:
    w = z[f"{REP}__{MODEL}__{symptom}__coef"]  # (1, D)
    b = z[f"{REP}__{MODEL}__{symptom}__intercept"]  # (1,)
    logit = features @ w.T + b  # (n, 1)
    prob = 1.0 / (1.0 + np.exp(-logit))
    print(symptom, float(logit[0]), float(prob[0]))
```

**3 · Threshold.** `logit >= 0` (equivalently `prob >= 0.5`) is the decision boundary the reported F1 uses. No test labels were used to tune it.

**4 · Prefer the reference-relative score.** More defensible than the raw probability: rank the logit against the out-of-fold logits of the untreated reference images in the *same* representation/model arm.

```python
import pandas as pd

sc = pd.read_csv("reports/symptom_vs_untreated_logistic/image_symptom_scores.csv")
arm = sc[(sc.representation == REP) & (sc.model == "BioCLIP")]
ref = np.sort(
    arm.loc[
        arm[f"{symptom}_label_state"] == "control_reference", f"{symptom}_logit"
    ].to_numpy()
)
percentile = np.searchsorted(ref, float(logit[0]), side="right") / len(ref)
above_95 = float(logit[0]) >= np.quantile(ref, 0.95, method="higher")
```

This is what `*_control_percentile` and `*_above_control_95pct` already contain for every image in the 50 cm cohort — no need to recompute them for those.

## Which model to use for a "classify everything" run

**The five fold classifiers are not one of your choices.** Each symptom was fit five times during cross-validation, once per held-out group, but those models were transient — they existed only to produce out-of-fold scores and were never written to disk. `final_classifiers.npz` contains exactly one model per (representation, encoder, symptom): the all-data fit. So there is nothing to pick between, and picking one fold arbitrarily would be the wrong instinct anyway, since each saw only ~62 of the 78 pots.

What matters is *which images you are scoring*:

| images | use | why |
|---|---|---|
| new images, or any pot outside the 78 | `final_classifiers.npz` (all-data) | nothing was held out from it, and these pots were never seen |
| images already in the 50 cm cohort | `image_symptom_scores.csv` | already out-of-fold — each image was scored by the model whose training set excluded its pot |

Do not re-score cohort images with the all-data model and present the result as a prediction. That model saw those pots in training; the number will be optimistic and is not comparable to anything in `classifier_metric_summary.csv`.

**The choice you do have to make is the arm** — one of 3 representations × 4 encoders. Two rules:

1. **Pick one arm and use it for all four symptoms.** Choosing each symptom's best-scoring arm looks appealing and is selection bias: those bests were measured on the same data, and with 12 arms the winner is partly noise — foliar distortion's "winner" was decided on 11 positive images.
2. **`segmented__bioclip` is the reasonable default.** Masking to plant-only pixels removes the acquisition-batch structure that survives in the full and cropped views, and it also shows the smallest drift on the never-treated control pots. BioCLIP wins or ties on three of the four symptoms, and the 4096-d DINOv3 buys nothing here — every encoder saturates training accuracy, so the extra capacity is not the binding constraint.

Whatever you choose, write the representation and `model_key` into the output table next to every score. A bare probability column with no arm recorded cannot be reproduced or compared later.

If you want to hedge across arms, average the **control percentiles**, not the logits — each arm's logits live on its own scale, so a raw mean is dominated by whichever arm happens to have the widest spread.

## Batch scoring

The same four steps vectorize. Encode once per (representation, encoder) pair, then apply all four symptom heads to the same feature matrix — the encoder pass dominates the cost, the classifiers are a matrix multiply. Limit BLAS threads on a login node (`OMP_NUM_THREADS=8`).

## Caveats that travel with every number

- **The sigmoid is not a calibrated probability.** Training used balanced classes, so the implied prior is 50/50, not any real base rate. Ordering is meaningful; the absolute value is not, and crossing 0.5 is a threshold artifact rather than a belief tipping past even odds.
- **The negative class is *untreated*, not symptom-free.** A high score means "looks less like the untreated reference group," not "has this symptom."
- **The heads track calendar time.** Against the untreated reference the chlorosis direction sits at cosine 0.71 with an early→late axis; on never-treated control pots the mean score climbs 0.26 → 0.69 across three weeks. Score an image from a late timepoint and it will read high largely because it is late. Compare only within a fixed date, or report the date-only baseline alongside.
- **Epinasty and foliar distortion have no treated-plant negatives.** They can be scored, but there is no honest validation for them in this dataset.
- **A high score on an untreated plant is not necessarily wrong.** Every pot was droughted, so real chlorosis and marginal necrosis are plausible in the controls. That makes the reference class unreliable in both directions.
