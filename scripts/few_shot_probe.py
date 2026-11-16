"""Few-shot linear-SVM probes over every embedding bundle for a dataset."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC

from agdamage.data import DATA_DIR
from agdamage.embed import MODEL_SPECS, load_bundle
from agdamage.probe import (
    build_plan,
    check_plan,
    load_plan,
    row_index,
    save_plan,
    support_rows,
)

SHOTS = [1, 5, 10, 20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DATA_DIR / "embeddings")
    parser.add_argument(
        "--datasets", nargs="+", default=["Waterhemp_GreenHouse_Images"]
    )
    parser.add_argument("--plan", type=Path, default=DATA_DIR / "probe_plan.json")
    parser.add_argument("--out", type=Path, default=Path("reports/few_shot_probe.csv"))
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    bundles = {}
    for dataset in args.datasets:
        for key in MODEL_SPECS:
            path = args.embeddings / f"{dataset}__{key}.npz"
            if path.exists():
                bundles[(dataset, key)] = load_bundle(path)
            else:
                print(f"skipping missing {path.name}")
    if not bundles:
        raise SystemExit("no embedding bundles found")

    # One plan for every model and every image variant, so differences in the
    # results come from the representation and never from the draw.
    if args.plan.exists():
        plan = load_plan(args.plan)
        print(f"using existing plan {args.plan}")
    else:
        first = next(iter(bundles.values()))
        plan = build_plan(
            first["ids"],
            first["labels"],
            args.shots,
            args.repeats,
            args.test_fraction,
            args.seed,
        )
        save_plan(plan, args.plan)
        print(f"wrote plan {args.plan}")

    print(f"classes    : {plan['classes']}")
    print(f"train pool : {plan['train_pool_sizes']}")
    print(
        f"test images: {len(plan['test'])}   repeats: {plan['repeats']}   shots: {plan['shots']}\n"
    )

    rows = []
    for (dataset, key), bundle in bundles.items():
        check_plan(plan, bundle["ids"])
        index = row_index(bundle["ids"])
        features = bundle["features"]
        truth = np.asarray(bundle["labels"])

        test_rows = np.array([index[i] for i in plan["test"]])
        test_x, test_y = features[test_rows], truth[test_rows]

        for shots in args.shots:
            scores = []
            for repeat in range(plan["repeats"]):
                train_rows, train_y = support_rows(plan, repeat, shots, index)
                model = LinearSVC(C=args.C, random_state=args.seed)
                model.fit(features[train_rows], train_y)
                pred = model.predict(test_x)
                row = {
                    "dataset": dataset,
                    "model": MODEL_SPECS[key]["name"],
                    "model_key": key,
                    "shots": shots,
                    "repeat": repeat,
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                    "balanced_accuracy": balanced_accuracy_score(test_y, pred),
                    "macro_f1": f1_score(
                        test_y, pred, average="macro", zero_division=0
                    ),
                    "accuracy": accuracy_score(test_y, pred),
                }
                rows.append(row)
                scores.append(row["balanced_accuracy"])
            print(
                f"{dataset:38s} {MODEL_SPECS[key]['name']:18s} {shots:3d}-shot  "
                f"bal-acc {np.mean(scores):.3f} +/- {np.std(scores):.3f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
