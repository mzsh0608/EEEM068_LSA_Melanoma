from pathlib import Path

import pandas as pd

from src.publication_assets import (
    FIGURES,
    MODEL_ORDER,
    TABLE_NAMES,
    figure_manifest_rows,
    load_sources,
    publication_model_results,
)


ROOT = Path(__file__).resolve().parents[1]


def test_publication_model_results_preserves_order_and_uses_display_precision():
    source = pd.read_csv(ROOT / "outputs/final/tables/main_model_results.csv")
    source_copy = source.copy(deep=True)

    result = publication_model_results(source)

    assert tuple(result["Model"]) == MODEL_ORDER
    assert result.columns.tolist() == [
        "Model", "Representation", "Metadata", "ROC-AUC", "AP",
        "Sensitivity", "Specificity", "F1", "Accuracy",
    ]
    assert result.loc[result.Model.eq("M1"), "AP"].item() == "0.169"
    assert result.loc[result.Model.eq("M2"), "Sensitivity"].item() == "0.897"
    pd.testing.assert_frame_equal(source, source_copy)


def test_frozen_publication_sources_pass_pre_generation_validation():
    sources = load_sources(ROOT)

    assert tuple(sources["main"]["model_id"]) == MODEL_ORDER
    assert set(sources["threshold"]["model"]) == {"M1", "M2"}
    assert len(sources["bootstrap_samples"]) == 1000
    assert len(sources["failure_cases"]) == 24
    assert sources["gradcam"]["gradient_target"] == "raw_melanoma_logit"


def test_figure_manifest_has_complete_unique_output_map_and_boundaries():
    rows = figure_manifest_rows(ROOT)

    assert len(rows) == len(FIGURES) == 10
    assert len({row["figure_id"] for row in rows}) == 10
    assert all(row["figure_path_png"].endswith(".png") for row in rows)
    assert all(row["figure_path_pdf"].endswith(".pdf") for row in rows)
    assert all(row["generation_script_or_notebook"] == "src/publication_assets.py" for row in rows)
    assert all(row["source_files"] and row["interpretation_boundary"] for row in rows)
    assert len(TABLE_NAMES) == 6
