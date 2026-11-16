import numpy as np
import pandas as pd

from agdamage.assertion_projection import (
    circle_polygons,
    normalize_panel_coordinates,
    prepare_projection_coordinates,
    symptom_wedge_polygons,
)
from agdamage.treatment import MODEL_KEYS, REPRESENTATIONS


def cohort():
    return pd.DataFrame(
        [
            {
                "image_id_50": "i1",
                "observation_id": "o1",
                "pot_id": "p1",
                "pot_label": "001",
                "capture_date": "2026-05-07",
                "file_name": "images/i1.jpg",
                "original_relative_path": "healthy_untreated/i1.jpg",
                "source_label": "healthy_untreated",
                "capture_role": "standard",
                "chlorosis": "",
                "necrosis": "",
                "epinasty": "",
                "foliar_distortion": "",
            },
            {
                "image_id_50": "i2",
                "observation_id": "o2",
                "pot_id": "p2",
                "pot_label": "002",
                "capture_date": "2026-05-11",
                "file_name": "images/i2.jpg",
                "original_relative_path": "chlorosis_necrosis/i2.jpg",
                "source_label": "chlorosis_necrosis",
                "capture_role": "standard",
                "chlorosis": "true",
                "necrosis": "true",
                "epinasty": "",
                "foliar_distortion": "",
            },
        ]
    )


def coordinates():
    rows = []
    for representation in REPRESENTATIONS:
        for model_key in MODEL_KEYS:
            for method in ("t-SNE", "UMAP"):
                for index, row in cohort().iterrows():
                    rows.append(
                        {
                            "height_cm": 50,
                            "representation": representation,
                            "model": model_key,
                            "model_key": model_key,
                            "method": method,
                            "observation_id": row.observation_id,
                            "image_id": row.image_id_50,
                            "pot_id": row.pot_id,
                            "capture_date": row.capture_date,
                            "x": index,
                            "y": index + 1,
                        }
                    )
    return pd.DataFrame(rows)


def test_prepare_projection_coordinates_preserves_partial_assertions():
    result = prepare_projection_coordinates(cohort(), coordinates())
    healthy = result[result["image_id"].eq("i1")]
    both = result[result["image_id"].eq("i2")]
    assert healthy["positive_assertions"].eq("none recorded").all()
    assert both["positive_assertions"].eq("chlorosis, necrosis").all()
    assert both["chlorosis"].all() and both["necrosis"].all()
    assert not both["epinasty"].any()


def test_normalization_preserves_aspect_ratio_and_center():
    normalized = normalize_panel_coordinates(np.asarray([[0, 0], [4, 2]]))
    assert np.allclose(normalized, [[-0.5, -0.25], [0.5, 0.25]])


def test_fixed_quadrants_and_circle_geometry():
    xy = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    circle_x, circle_y = circle_polygons(xy, radius=0.1, arc_points=8)
    assert circle_x.count(None) == 2 and circle_y.count(None) == 2

    wedge_x, wedge_y = symptom_wedge_polygons(
        xy,
        np.asarray([True, False]),
        "chlorosis",
        radius=0.1,
        arc_points=5,
    )
    asserted_x = np.asarray([value for value in wedge_x if value is not None])
    asserted_y = np.asarray([value for value in wedge_y if value is not None])
    assert wedge_x.count(None) == 1
    assert (asserted_x <= 1e-12).all()
    assert (asserted_y >= -1e-12).all()
