from pathlib import Path

import pandas as pd
import pytest

from src.review_package import (
    PACKAGE_NAME,
    build_review_package,
    render_figure_review,
    render_review_checklist,
)


ROOT = Path(__file__).resolve().parents[1]


def test_review_documents_cover_all_candidates_and_special_questions():
    review = render_figure_review(ROOT)
    checklist = render_review_checklist()

    assert review.count("\nReview:\n") == 16
    assert "## Main-01 - Pipeline" in review
    assert "## Main-Table-01 - Model results" in review
    assert "## App-08 - Subgroup performance" in review
    assert "## App-Table-05 - Subgroup summary" in review
    assert "Does this match the actual implemented system?" in review
    assert "Is AP used, not PR-AUC?" in review
    assert "Does nothing imply metadata explanation?" in review
    assert "- [ ] Main-01 approved" in checklist
    assert "- [ ] App-08 approved" in checklist
    assert 'no "test performance"' in checklist
    assert 'no "optimal threshold"' in checklist


@pytest.fixture(scope="module")
def built_package(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("j2c-package")
    package_root = output_root / PACKAGE_NAME
    zip_path = output_root / "Phase_J_Figure_Review.zip"
    return build_review_package(
        ROOT,
        package_root=package_root,
        zip_path=zip_path,
        include_html=False,
    )


def test_review_package_manifest_and_inventory_are_complete(built_package):
    manifest = pd.read_csv(built_package["manifest_path"])

    assert built_package["zip_opens"]
    assert built_package["all_expected_figures_present"]
    assert built_package["notebook_present"]
    assert built_package["manifest_rows"] == len(manifest)
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert manifest["bytes"].gt(0).all()
    assert set(manifest["category"]) >= {
        "review document", "notebook", "main figure", "main table",
        "appendix figure", "appendix table", "provenance",
    }


def test_review_zip_excludes_sensitive_or_redundant_scientific_inputs(built_package):
    assert not built_package["raw_dataset_included"]
    assert not built_package["checkpoints_included"]
    assert not built_package["repository_internals_included"]
    assert not built_package["credential_like_content_detected"]
