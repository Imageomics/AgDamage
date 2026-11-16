# Reports

Each maintained analysis has a dedicated directory and method README:

- [`treated_vs_untreated_logistic/`](treated_vs_untreated_logistic/) — the longitudinal treated/reference probe plus embedding projections;
- [`treated_vs_untreated_logistic_matched/`](treated_vs_untreated_logistic_matched/) — treated/reference learning curves using the symptom-probe split strategy;
- [`symptom_vs_untreated_logistic/`](symptom_vs_untreated_logistic/) — four independent positive-symptom versus untreated-reference heads;
- [`positive_assertion_projections/`](positive_assertion_projections/) — interactive partial-label glyph projections;
- [`cropped_50cm_prediction_projections/`](cropped_50cm_prediction_projections/) — projections of OOF and all-data symptom-head outputs on cropped images.

Git retains the method READMEs, compact manifests, summary tables, and static figures needed for review. Large HTML visualizations, fold/support plans, per-trial tables, per-image score tables, and fitted classifier arrays are reproducible outputs and remain local. Run the command in an analysis README to regenerate its complete directory after preparing `data/metadata.csv` and the UUID-keyed embedding bundles.

`misc/` holds superseded exploratory outputs and is not part of the maintained analysis surface.
