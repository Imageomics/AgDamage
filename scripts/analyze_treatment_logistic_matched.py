"""Run treated-vs-untreated probes using the symptom-analysis protocol."""

import argparse
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from agdamage.data import DATA_DIR
from agdamage.rekey import file_digest, manifest_path
from agdamage.treatment import (
    MODEL_KEYS,
    MODEL_NAMES,
    REPRESENTATIONS,
    build_50cm_cohort,
    build_stratified_group_evaluation_plan,
    evaluate_logistic_trials,
    load_metadata,
)

REPORT_DIR = Path("reports/treated_vs_untreated_logistic_matched")
MODEL_COLORS = {
    "bioclip": "#0072B2",
    "bioclip-2": "#009E73",
    "bioclip-2.5": "#E69F00",
    "dinov3-vit7b16": "#CC79A7",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "metadata.csv")
    parser.add_argument(
        "--embeddings", type=Path, default=DATA_DIR / "embeddings" / "rekeyed"
    )
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40, 60]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args()


def metric_summary(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "accuracy",
    )
    keys = [
        "representation",
        "model",
        "model_key",
        "support_images_per_class",
    ]
    repeat_keys = [*keys, "repeat"]
    ranking_scores = trials.groupby(repeat_keys, as_index=False)[
        ["roc_auc", "average_precision"]
    ].mean()
    counts = trials.groupby(repeat_keys, as_index=False)[
        ["true_positive", "false_positive", "true_negative", "false_negative"]
    ].sum()
    repeat_scores = ranking_scores.merge(counts, on=repeat_keys, validate="one_to_one")
    tp = repeat_scores["true_positive"]
    fp = repeat_scores["false_positive"]
    tn = repeat_scores["true_negative"]
    fn = repeat_scores["false_negative"]

    def safe_ratio(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros(len(repeat_scores), dtype=float),
            where=np.asarray(denominator) != 0,
        )

    repeat_scores["precision"] = safe_ratio(tp, tp + fp)
    repeat_scores["recall"] = safe_ratio(tp, tp + fn)
    repeat_scores["f1"] = safe_ratio(2 * tp, 2 * tp + fp + fn)
    specificity = safe_ratio(tn, tn + fp)
    repeat_scores["balanced_accuracy"] = (repeat_scores["recall"] + specificity) / 2
    repeat_scores["accuracy"] = safe_ratio(tp + tn, tp + fp + tn + fn)

    records = []
    for group_values, group in repeat_scores.groupby(keys, sort=False):
        base = dict(zip(keys, group_values, strict=True))
        for metric in metrics:
            values = group[metric]
            records.append(
                {
                    **base,
                    "metric": metric,
                    "mean": values.mean(),
                    "q10": values.quantile(0.10),
                    "q90": values.quantile(0.90),
                }
            )
    return pd.DataFrame(records)


def performance_figure(summary: pd.DataFrame, output_path: Path) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "agdamage-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for column_index, representation in enumerate(REPRESENTATIONS):
        axis = axes[column_index]
        subset = summary[
            summary["representation"].eq(representation) & summary["metric"].eq("f1")
        ]
        for model_key in MODEL_KEYS:
            model_rows = subset[subset["model_key"].eq(model_key)].sort_values(
                "support_images_per_class"
            )
            x = model_rows["support_images_per_class"].to_numpy()
            axis.plot(
                x,
                model_rows["mean"],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[model_key],
                label=MODEL_NAMES[model_key],
            )
            axis.fill_between(
                x,
                model_rows["q10"],
                model_rows["q90"],
                color=MODEL_COLORS[model_key],
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(f"50 cm · {representation}")
        axis.set_xscale("symlog", linthresh=2)
        axis.set_xticks(sorted(subset["support_images_per_class"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Support images per class")
        if column_index == 0:
            axis.set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Treated vs untreated using the symptom-probe evaluation strategy",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def package_versions() -> dict:
    cache_dir = Path(tempfile.gettempdir()) / "agdamage-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    cohort = build_50cm_cohort(metadata)
    shots = tuple(sorted(set(args.shots)))
    plan = build_stratified_group_evaluation_plan(
        cohort,
        shots,
        args.repeats,
        args.seed,
        n_folds=args.folds,
    )
    (args.out / "evaluation_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )

    trials = evaluate_logistic_trials(
        cohort,
        plan,
        args.embeddings,
        c_value=args.C,
        heights=(50,),
    )
    trials.to_csv(args.out / "classifier_trials.csv", index=False)
    summary = metric_summary(trials)
    summary.to_csv(args.out / "classifier_metric_summary.csv", index=False)
    performance_figure(summary, args.out / "classifier_performance_vs_support.png")

    bundle_paths = sorted(args.embeddings.glob("*.npz"))
    outputs = sorted(
        path
        for path in args.out.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema": "agdamage.treated_vs_untreated_matched_strategy",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "target": {
            "false": "source_label == healthy_untreated",
            "true": "source_label is a positive symptom label",
            "interpretation": "image-time treatment status, not symptom absence",
        },
        "cohort": {
            "rule": "every preferred 50 cm image",
            "observations": len(cohort),
            "pots": int(cohort["pot_id"].nunique()),
            "treated": int(cohort["herbicide_treated"].sum()),
            "untreated": int((~cohort["herbicide_treated"]).sum()),
        },
        "arms": {
            "representations": list(REPRESENTATIONS),
            "models": list(MODEL_KEYS),
        },
        "classifier": {
            "type": "sklearn.linear_model.LogisticRegression",
            "penalty": "L2",
            "C": args.C,
            "solver": "liblinear",
            "support_images_per_class": list(shots),
            "repeats": args.repeats,
            "folds": args.folds,
            "primary_metric": "pooled-fold F1 at logit >= 0",
        },
        "strategy_match": {
            "reference": "reports/symptom_vs_untreated_logistic",
            "shared_features": [
                "five stratified pot-held-out folds",
                "balanced nested support draws",
                "pot-diverse sampling",
                "twenty support repeats",
                "F1 at logit >= 0 pooled over folds within repeat",
                "10th-90th percentile across repeats",
            ],
        },
        "inputs": {
            "metadata": {
                "path": manifest_path(args.metadata),
                "sha256": file_digest(args.metadata),
            },
            "embedding_bundles": {
                path.name: file_digest(path) for path in bundle_paths
            },
        },
        "outputs": {path.name: file_digest(path) for path in outputs},
        "software": package_versions(),
    }
    (args.out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote matched treated/untreated analysis to {args.out}", flush=True)


if __name__ == "__main__":
    main()
