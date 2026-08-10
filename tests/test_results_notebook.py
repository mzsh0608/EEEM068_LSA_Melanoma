from pathlib import Path

import nbformat

from src.results_notebook import SECTION_HEADINGS, TABLE_NAMES, build_notebook


def _notebook_sources(path: Path) -> tuple[str, str]:
    notebook = nbformat.read(path, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    return markdown, code


def test_results_notebook_contains_all_required_sections_and_sources(tmp_path):
    path = build_notebook(tmp_path / "02_results_analysis.ipynb")
    markdown, code = _notebook_sources(path)

    positions = [markdown.index(heading) for heading in SECTION_HEADINGS]
    assert positions == sorted(positions)
    assert all(f'"{name}"' in code for name in TABLE_NAMES)


def test_results_notebook_is_analysis_only_and_uses_required_terminology(tmp_path):
    path = build_notebook(tmp_path / "02_results_analysis.ipynb")
    markdown, code = _notebook_sources(path)
    notebook_text = f"{markdown}\n{code}".lower()

    assert "average precision" in notebook_text
    assert "fixed patient-aware validation fold" in notebook_text
    assert "test performance" not in notebook_text
    assert "pr-auc" not in notebook_text
    assert "optimal" not in notebook_text
    assert "--fit" not in code
    assert "savefig" not in code
    assert ".to_csv" not in code
    assert "requests." not in code
    assert "urlopen" not in code
    assert "c:\\" not in code.lower()


def test_results_notebook_has_no_stored_outputs_before_execution(tmp_path):
    path = build_notebook(tmp_path / "02_results_analysis.ipynb")
    notebook = nbformat.read(path, as_version=4)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)
