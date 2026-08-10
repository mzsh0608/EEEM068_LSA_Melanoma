"""Build and verify the Phase J manual figure-review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
from nbconvert import HTMLExporter

from src.publication_assets import FIGURES, TABLE_NAMES, find_repo_root, sha256_file


ZIP_NAME = "Phase_J_Figure_Review.zip"
PACKAGE_NAME = "review_package"


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    title: str
    files: tuple[str, ...]
    placement: str
    purpose: str
    source: str
    boundary: str
    questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackageRecord:
    item_id: str
    path: Path
    category: str
    intended_report_location: str
    source_artifact: str


def _review_items(root: Path) -> list[ReviewItem]:
    figure_manifest = pd.read_csv(root / "outputs/final/manifests/J2B_figure_sources.csv").set_index("figure_id")
    figure_placements = {
        "Fig_Main_01": "Main report - Methodology",
        "Fig_Main_02": "Main report - Results",
        "Fig_App_01": "Appendix - Dataset overview",
        "Fig_App_02": "Appendix - Training behaviour",
        "Fig_App_03": "Appendix - Threshold-0.5 errors",
        "Fig_App_04": "Appendix - Ranking curves",
        "Fig_App_05": "Appendix - Internal uncertainty",
        "Fig_App_06": "Appendix - Qualitative failure review",
        "Fig_App_07": "Appendix - Explainability",
        "Fig_App_08": "Appendix - Exploratory subgroups",
    }
    figure_titles = {
        "Fig_Main_01": ("Main-01", "Pipeline"),
        "Fig_Main_02": ("Main-02", "Threshold trade-off"),
        "Fig_App_01": ("App-01", "Class distribution"),
        "Fig_App_02": ("App-02", "Training curves"),
        "Fig_App_03": ("App-03", "Confusion matrices"),
        "Fig_App_04": ("App-04", "ROC and precision-recall curves"),
        "Fig_App_05": ("App-05", "Bootstrap differences"),
        "Fig_App_06": ("App-06", "Failure cases"),
        "Fig_App_07": ("App-07", "Grad-CAM"),
        "Fig_App_08": ("App-08", "Subgroup performance"),
    }
    special_questions = {
        "Fig_Main_01": (
            "Does this match the actual implemented system?",
            "Is patient-aware validation obvious?",
            "Is M2 metadata preprocessing understandable?",
            "Are there any obsolete architectures?",
            "Is text readable at paper scale?",
        ),
        "Fig_Main_02": (
            "Is 0.5 clearly marked?",
            'Is nothing labelled "optimal"?',
            "Are M1 and M2 distinguishable?",
            "Are all axes readable?",
        ),
        "Fig_App_04": (
            "Is AP used, not PR-AUC?",
            "Are all five models correctly labelled?",
            "Are curves and legends readable?",
        ),
        "Fig_App_05": (
            "Is paired patient-level wording correct?",
            "Is the zero reference visible?",
            "Is there no significance or external-validation claim?",
        ),
        "Fig_App_06": (
            "Do all images display correctly?",
            "Are FN/FP/TP/TN rows correctly labelled?",
            "Are there no byte-identical duplicate visual examples?",
            "Are there no stretched or distorted images?",
        ),
        "Fig_App_07": (
            "Do original and overlay pairs match?",
            "Are heatmaps aligned?",
            "Are TP/TN/FP/FN labels correct?",
            "Is there no duplicate content?",
            "Does nothing imply metadata explanation?",
        ),
        "Fig_App_08": (
            "Are positive counts visible or recoverable?",
            "Are tiny groups not hidden?",
            "Is there no bias or fairness label?",
            "Are labels readable?",
        ),
    }

    items: list[ReviewItem] = []
    for figure_id, location, stem in FIGURES:
        row = figure_manifest.loc[figure_id]
        display_id, title = figure_titles[figure_id]
        package_dir = "main" if location == "main" else "appendix"
        items.append(ReviewItem(
            item_id=display_id,
            title=title,
            files=(f"{package_dir}/{stem}.png", f"{package_dir}/{stem}.pdf"),
            placement=figure_placements[figure_id],
            purpose=str(row.primary_message),
            source=str(row.source_files),
            boundary=str(row.interpretation_boundary),
            questions=special_questions.get(figure_id, ()),
        ))

    items.insert(2, ReviewItem(
        item_id="Main-Table-01",
        title="Model results",
        files=tuple(f"main/Table_Main_01_ModelResults.{suffix}" for suffix in ("csv", "md", "tex")),
        placement="Main report - Results",
        purpose="Present the frozen five-model hierarchy at publication display precision.",
        source="outputs/final/tables/main_model_results.csv",
        boundary=(
            "H0/H1 use the historical 5,000-image training subset; B0/M1/M2 use the full training partition; "
            "threshold-dependent metrics use 0.5 on fixed Fold-0 validation."
        ),
        questions=(
            "Are model names correct?",
            "Is AP terminology correct?",
            "Is the H0/H1 training-scope note visible?",
            "Are threshold-dependent metrics clearly at 0.5?",
            "Are the numbers plausible?",
        ),
    ))

    table_items = (
        ReviewItem(
            "App-Table-01", "Deep protocol",
            tuple(f"tables/Table_App_01_DeepProtocol.{suffix}" for suffix in ("csv", "md", "tex")),
            "Appendix - Experimental protocol", "Summarise matched deep-model settings and actual duration.",
            "outputs/final/tables/deep_model_protocol.csv",
            "Checkpoint and duration selection use validation ROC-AUC; settings are not claimed as tuned optima.",
        ),
        ReviewItem(
            "App-Table-02", "Hyperparameter selection",
            tuple(f"tables/Table_App_02_HyperparameterSelection.{suffix}" for suffix in ("csv", "md", "tex")),
            "Appendix - Experimental protocol", "Document parameter values and their selection basis.",
            "outputs/final/tables/hyperparameter_selection.csv; outputs/final/manifests/hyperparameter_strategy.json",
            "No systematic grid, random, or Bayesian search was performed.",
        ),
        ReviewItem(
            "App-Table-03", "Metadata ablation",
            tuple(f"tables/Table_App_03_MetadataAblation.{suffix}" for suffix in ("csv", "md", "tex")),
            "Appendix - Metadata ablation", "Show M1/M2 metric differences at the common comparison point.",
            "outputs/final/tables/M1_M2_metadata_ablation.csv",
            "This is a system ablation; per-sample differences are not causal metadata effects.",
        ),
        ReviewItem(
            "App-Table-04", "Bootstrap summary",
            tuple(f"tables/Table_App_04_BootstrapSummary.{suffix}" for suffix in ("csv", "md", "tex")),
            "Appendix - Internal uncertainty", "Report paired patient-level bootstrap difference intervals.",
            "outputs/final/tables/bootstrap_summary.csv",
            "Internal Fold-0 uncertainty only; not external validation or a significance test.",
        ),
        ReviewItem(
            "App-Table-05", "Subgroup summary",
            tuple(f"tables/Table_App_05_SubgroupSummary.{suffix}" for suffix in ("csv", "md", "tex")),
            "Appendix - Exploratory subgroups", "Provide subgroup support and descriptive performance metrics.",
            "outputs/final/tables/subgroup_summary.csv",
            "Sparse-positive groups are unstable and do not support strong bias claims.",
        ),
    )
    items.extend(table_items)
    return items


def render_figure_review(root: Path) -> str:
    lines = [
        "# Figure and Table Manual Review",
        "",
        "This document records user approval decisions before report writing. Quantitative values and scientific artifacts are frozen.",
        "",
    ]
    for item in _review_items(root):
        lines.extend([
            f"## {item.item_id} - {item.title}",
            "",
            "Files:",
            *[f"- `{filename}`" for filename in item.files],
            "",
            "Intended placement:",
            item.placement,
            "",
            "Purpose:",
            item.purpose,
            "",
            "Source evidence:",
            item.source,
            "",
            "Important interpretation boundary:",
            item.boundary,
            "",
        ])
        if item.questions:
            lines.extend(["Special review questions:", *[f"- [ ] {question}" for question in item.questions], ""])
        lines.extend([
            "Review:",
            "- [ ] APPROVED",
            "- [ ] APPROVED WITH NOTE",
            "- [ ] REVISE",
            "- [ ] REMOVE",
            "",
            "Notes:",
            "________________________________________",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_review_checklist() -> str:
    return """# Manual Review Checklist

## General

- [ ] all fonts readable
- [ ] no clipped axes
- [ ] no overlapping titles
- [ ] legends readable
- [ ] grayscale distinction acceptable
- [ ] model labels consistent
- [ ] AP terminology consistent
- [ ] Fold 0 described as validation
- [ ] no "test performance"
- [ ] no "optimal threshold"
- [ ] no unsupported causal wording

## Main Paper

- [ ] Main-01 approved
- [ ] Main-02 approved
- [ ] Main Table 01 approved

## Appendix

- [ ] App-01 approved
- [ ] App-02 approved
- [ ] App-03 approved
- [ ] App-04 approved
- [ ] App-05 approved
- [ ] App-06 approved
- [ ] App-07 approved
- [ ] App-08 approved

## Final

- [ ] all revision notes recorded
- [ ] ready to freeze figures
"""


def _copy_file(
    root: Path,
    source: Path,
    destination: Path,
    *,
    category: str,
    intended_location: str,
    records: list[PackageRecord],
) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError(f"Copied file hash mismatch: {source}")
    records.append(PackageRecord(
        item_id=f"{destination.stem}-{destination.suffix.lstrip('.').upper()}",
        path=destination,
        category=category,
        intended_report_location=intended_location,
        source_artifact=source.relative_to(root.resolve()).as_posix(),
    ))


def _add_generated_record(
    path: Path,
    *,
    item_id: str,
    category: str,
    intended_location: str,
    records: list[PackageRecord],
) -> None:
    records.append(PackageRecord(
        item_id=item_id,
        path=path,
        category=category,
        intended_report_location=intended_location,
        source_artifact="src/review_package.py",
    ))


def _export_notebook_html(notebook_path: Path, html_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if errors:
        raise ValueError("Executed notebook contains stored error outputs")
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(notebook)
    html_path.write_text(html, encoding="utf-8")
    if "Final Results Analysis" not in html or html_path.stat().st_size == 0:
        raise ValueError("Notebook HTML export is incomplete")


def _write_manifest(package_root: Path, records: list[PackageRecord]) -> Path:
    rows = []
    for record in sorted(records, key=lambda item: item.path.relative_to(package_root).as_posix()):
        rows.append({
            "item_id": record.item_id,
            "filename": record.path.relative_to(package_root).as_posix(),
            "category": record.category,
            "intended_report_location": record.intended_report_location,
            "source_artifact": record.source_artifact,
            "sha256": sha256_file(record.path),
            "bytes": record.path.stat().st_size,
        })
    manifest_path = package_root / "provenance" / "review_package_manifest.csv"
    pd.DataFrame(rows, columns=[
        "item_id", "filename", "category", "intended_report_location",
        "source_artifact", "sha256", "bytes",
    ]).to_csv(manifest_path, index=False, lineterminator="\n")
    return manifest_path


def _write_deterministic_zip(package_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            archive_name = (
                Path(package_root.name) / path.relative_to(package_root)
            ).as_posix()
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def verify_review_zip(package_root: Path, zip_path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Review ZIP is not a readable ZIP archive")
    expected_names = {
        f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
        for path in package_root.rglob("*")
        if path.is_file()
    }
    prohibited_suffixes = {".jpg", ".jpeg", ".dcm", ".pt", ".pth", ".ckpt"}
    prohibited_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "train_images", "test_images"}
    secret_patterns = (
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(rb"AKIA[0-9A-Z]{16}"),
        re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
        re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    )

    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("Review ZIP contains a corrupt member")
        names = set(archive.namelist())
        if names != expected_names:
            missing = sorted(expected_names - names)
            unexpected = sorted(names - expected_names)
            raise ValueError(f"Review ZIP inventory mismatch; missing={missing}, unexpected={unexpected}")
        for info in archive.infolist():
            member_path = Path(info.filename)
            lower_parts = {part.lower() for part in member_path.parts}
            if member_path.suffix.lower() in prohibited_suffixes or lower_parts.intersection(prohibited_parts):
                raise ValueError(f"Prohibited package member: {info.filename}")
            data = archive.read(info.filename)
            if any(pattern.search(data) for pattern in secret_patterns):
                raise ValueError(f"Credential-like content detected in {info.filename}")
            local_path = package_root / Path(*member_path.parts[1:])
            if sha256_file(local_path) != hashlib.sha256(data).hexdigest():
                raise ValueError(f"ZIP member differs from package source: {info.filename}")

    required_names = {
        f"{PACKAGE_NAME}/notebook/02_results_analysis.ipynb",
        f"{PACKAGE_NAME}/FIGURE_REVIEW.md",
        f"{PACKAGE_NAME}/REVIEW_CHECKLIST.md",
        f"{PACKAGE_NAME}/provenance/review_package_manifest.csv",
    }
    for _, location, stem in FIGURES:
        destination = "main" if location == "main" else "appendix"
        required_names.update({
            f"{PACKAGE_NAME}/{destination}/{stem}.png",
            f"{PACKAGE_NAME}/{destination}/{stem}.pdf",
        })
    if not required_names.issubset(expected_names):
        raise ValueError("Required review artifacts are missing")

    return {
        "zip_path": zip_path,
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "member_count": len(expected_names),
        "zip_opens": True,
        "all_expected_figures_present": True,
        "notebook_present": True,
        "provenance_present": True,
        "raw_dataset_included": False,
        "checkpoints_included": False,
        "repository_internals_included": False,
        "credential_like_content_detected": False,
    }


def build_review_package(
    root: Path,
    *,
    package_root: Path | None = None,
    zip_path: Path | None = None,
    include_html: bool = True,
) -> dict[str, Any]:
    root = find_repo_root(root)
    package_root = (package_root or root / PACKAGE_NAME).resolve()
    zip_path = (zip_path or root / ZIP_NAME).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    allowed_roots = (root, temporary_root)
    if (
        package_root.name != PACKAGE_NAME
        or not any(parent in package_root.parents for parent in allowed_roots)
    ):
        raise ValueError(f"Review package directory must be named {PACKAGE_NAME!r}")
    if (
        zip_path.name != ZIP_NAME
        or not any(parent in zip_path.parents for parent in allowed_roots)
    ):
        raise ValueError(f"Review ZIP must be named {ZIP_NAME!r}")
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)
    for directory in ("notebook", "main", "appendix", "tables", "provenance"):
        (package_root / directory).mkdir()

    records: list[PackageRecord] = []
    figure_review_path = package_root / "FIGURE_REVIEW.md"
    figure_review_path.write_text(render_figure_review(root), encoding="utf-8")
    _add_generated_record(
        figure_review_path, item_id="Figure-Review", category="review document",
        intended_location="Manual review", records=records,
    )
    checklist_path = package_root / "REVIEW_CHECKLIST.md"
    checklist_path.write_text(render_review_checklist(), encoding="utf-8")
    _add_generated_record(
        checklist_path, item_id="Review-Checklist", category="review document",
        intended_location="Manual review", records=records,
    )

    notebook_source = root / "notebooks" / "02_results_analysis.ipynb"
    notebook_copy = package_root / "notebook" / notebook_source.name
    _copy_file(
        root, notebook_source, notebook_copy, category="notebook",
        intended_location="Manual review", records=records,
    )
    html_included = False
    if include_html:
        html_path = package_root / "notebook" / "02_results_analysis.html"
        _export_notebook_html(notebook_source, html_path)
        _add_generated_record(
            html_path, item_id="Results-Notebook-HTML", category="notebook",
            intended_location="Manual review", records=records,
        )
        html_included = True

    publication_table_dir = root / "outputs" / "final" / "tables" / "publication"
    for _, location, stem in FIGURES:
        source_dir = root / "outputs" / "final" / "figures" / location
        destination_dir = "main" if location == "main" else "appendix"
        category = "main figure" if location == "main" else "appendix figure"
        intended_location = "Main report candidate" if location == "main" else "Appendix candidate"
        for suffix in ("png", "pdf"):
            source = source_dir / f"{stem}.{suffix}"
            _copy_file(
                root, source, package_root / destination_dir / source.name,
                category=category, intended_location=intended_location, records=records,
            )

    for table_name in TABLE_NAMES:
        destination_dir = "main" if table_name.startswith("Table_Main_") else "tables"
        category = "main table" if destination_dir == "main" else "appendix table"
        intended_location = "Main report - Results" if destination_dir == "main" else "Appendix candidate"
        for suffix in ("csv", "md", "tex"):
            source = publication_table_dir / f"{table_name}.{suffix}"
            _copy_file(
                root, source, package_root / destination_dir / source.name,
                category=category, intended_location=intended_location, records=records,
            )

    provenance_sources = (
        root / "outputs/final/manifests/J2B_figure_sources.csv",
        root / "outputs/final/manifests/J2B_failure_case_sources.csv",
        root / "outputs/final/manifests/J2B_visual_qc.json",
        root / "outputs/final/manifests/authoritative_sources.json",
        root / "outputs/final/manifests/comparison_boundaries.json",
        root / "outputs/final/manifests/hyperparameter_strategy.json",
    )
    for source in provenance_sources:
        _copy_file(
            root, source, package_root / "provenance" / source.name, category="provenance",
            intended_location="Provenance only", records=records,
        )

    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    direct_copies = [record for record in records if record.source_artifact != "src/review_package.py"]
    package_audit = {
        "phase": "J2C",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_git_head": source_head,
        "scientific_artifacts_modified": False,
        "direct_copy_hashes_verified": len(direct_copies),
        "notebook_html_exported": html_included,
        "expected_main_figure_files": 4,
        "expected_appendix_figure_files": 16,
        "expected_publication_table_files": 18,
        "excluded": ["raw dataset", "raw images", "checkpoints", ".venv", ".git", "cache/temp files"],
    }
    audit_path = package_root / "provenance" / "review_package_audit.json"
    audit_path.write_text(json.dumps(package_audit, indent=2) + "\n", encoding="utf-8")
    _add_generated_record(
        audit_path, item_id="Review-Package-Audit", category="provenance",
        intended_location="Provenance only", records=records,
    )

    manifest_path = _write_manifest(package_root, records)
    _write_deterministic_zip(package_root, zip_path)
    verification = verify_review_zip(package_root, zip_path)
    verification.update({
        "package_root": package_root,
        "manifest_path": manifest_path,
        "manifest_rows": len(records),
        "html_included": html_included,
    })
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()
    result = build_review_package(args.root, include_html=not args.no_html)
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
