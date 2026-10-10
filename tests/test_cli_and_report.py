"""Tests of the command line interface and the report generators."""

import json
from pathlib import Path

import pytest

from qcog.cli import main
from qcog.explain import explain_backend, explain_interference, explain_order_effect, explain_qq
from qcog.registry import get_model
from qcog.report import write_catalogue_report, write_markdown_report, write_model_report
from qcog.selftest import run_evidence


def test_cli_lists_models(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "question-order" in output
    assert "Module D" in output


def test_cli_info_and_backends(capsys):
    assert main(["info", "conjunction-fallacy"]) == 0
    info = capsys.readouterr().out
    assert "caveats" in info.lower()
    assert "mechanism" in info.lower()
    assert main(["backends"]) == 0
    assert "exact" in capsys.readouterr().out


def test_cli_run_text_and_json(capsys):
    assert main(["run", "question-order", "--param", "angle_a=1.0"]) == 0
    assert "Findings" in capsys.readouterr().out
    assert main(["run", "question-order", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["slug"] == "question-order"


def test_cli_run_with_shots(capsys):
    assert main(["run", "question-order", "--backend", "sampled", "--shots", "2000", "--seed", "2"]) == 0
    assert "shots" in capsys.readouterr().out


def test_cli_sweep_and_evidence(capsys):
    assert main(["sweep", "question-order"]) == 0
    assert "sharpness" in capsys.readouterr().out
    assert main(["evidence"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_rejects_unknown_model(capsys):
    with pytest.raises(KeyError):
        main(["run", "nope"])


def test_evidence_is_all_passing():
    rows = run_evidence()
    assert len(rows) >= 15
    failures = [row for row in rows if not row["passed"]]
    assert not failures, failures


def test_model_report_is_accessible(tmp_path):
    model = get_model("question-order")
    result = model.run()
    path = write_model_report(model, result, tmp_path)
    html = Path(path).read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert 'class="skip"' in html
    assert 'alt="' in html and "<figcaption>" in html
    assert "<caption>" in html
    assert "Long description" in html
    assert "supported by Qollab" in html
    assert "What this does not claim" in html
    assert path.with_suffix(".png") or list(tmp_path.glob("*.png"))


def test_catalogue_report_embeds_figures_with_alternatives(tmp_path):
    """A regression guard: the catalogue report must contain figures, not placeholders."""
    path = write_catalogue_report(tmp_path / "report.html", max_models=2, include_figures=True)
    html = Path(path).read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") >= 2
    assert html.count("<figcaption>") >= 2
    assert html.count("Long description of this chart") >= 2
    assert "Figures unavailable" not in html


def test_catalogue_and_markdown_reports(tmp_path):
    html_path = write_catalogue_report(tmp_path / "report.html", max_models=2, include_figures=False)
    html = Path(html_path).read_text(encoding="utf-8")
    assert "Q-Cog model catalogue" in html
    assert "Evidence" in html
    assert "Self-checks" in html
    markdown_path = write_markdown_report(tmp_path / "report.md", max_models=2)
    markdown = Path(markdown_path).read_text(encoding="utf-8")
    assert "# Q-Cog model catalogue" in markdown
    assert "Does not claim" in markdown


def test_explanations_are_generated_from_numbers():
    model = get_model("question-order")
    result = model.run()
    text = explain_order_effect(result.analysis["order_effect"])
    assert "asked first" in text
    assert explain_interference(result.analysis["interference"])
    assert "parameter-free" in explain_qq(result.analysis["qq_equality"])
    assert "shots" in explain_backend(next(iter(model.run(backend="sampled", shots=500, seed=1).sequential_results().values())))