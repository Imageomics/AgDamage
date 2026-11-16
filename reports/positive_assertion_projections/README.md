# Positive-assertion embedding projections

This visualization represents the greenhouse labels as partial, multi-label assertions rather than mutually exclusive image classes. It uses every preferred 50 cm image and provides a selector for the full, cropped, and segmented image representations. Each view contains BioCLIP, BioCLIP 2, BioCLIP 2.5, and DINOv3 ViT-7B/16 projections under t-SNE and UMAP.

## Glyph semantics

Each image is drawn as a circle with four fixed quadrants:

- upper-left: chlorosis;
- upper-right: necrosis;
- lower-left: epinasty; and
- lower-right: foliar distortion.

A quadrant is colored only when the corresponding metadata field contains an explicit positive assertion. A white quadrant means unknown or not asserted; it does not mean that the symptom is absent. A fully hollow glyph therefore means only that no positive symptom assertion was recorded. It is not labeled healthy.

The fixed positions are preferable to a conventional pie chart whose colored slices expand to fill the circle: quadrant location continues to identify the symptom even when only one assertion is present.

## Coordinates and interaction

The label-independent 50 cm coordinates from the treated-versus-untreated projection analysis are reused exactly. Each panel is centered and scaled by a single factor for display, preserving its aspect ratio and geometry. Labels affect only the marker glyphs, not t-SNE or UMAP.

Use the dropdown to switch representations. Hovering a glyph shows its image, positive assertions, original source label, pot, date, capture role, path, and UUID.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/plot_positive_assertion_projections.py
```

`projection_coordinates.csv` retains the enriched coordinates and assertions. `analysis_manifest.json` records the inputs, glyph mapping, normalization, software versions, and output checksums.
