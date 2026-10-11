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
    assert main(["run", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_a_user_error_is_a_message_rather_than_a_traceback(capsys):
    """A misspelled backend, parameter or file is a message, not a stack trace."""
    assert main(["run", "question-order", "--backend", "nonsense"]) == 2
    error = capsys.readouterr().err
    assert "unknown backend" in error and "Traceback" not in error
    assert main(["run", "question-order", "--param", "not_a_parameter=1"]) == 2
    assert "not_a_parameter" in capsys.readouterr().err


def test_cli_fit_reads_the_dataset_the_library_ships(capsys):
    """``qcog fit`` has to cope with the bundle in ``qcog/data``, not only with a flat mapping."""
    data = Path(__file__).resolve().parents[1] / "src" / "qcog" / "data" / "benchmarks.json"
    assert data.exists()
    assert main(["fit", "question-order", str(data)]) == 0
    output = capsys.readouterr().out
    assert "fitted" in output.lower() and "response" in output


def test_cli_prints_the_classical_comparison(capsys):
    assert main(["run", "question-order", "--fast"]) == 0
    output = capsys.readouterr().out
    assert "Against the classical baseline" in output
    assert "order effect on the first question" in output
    assert "Quantum circuit(s)" in output


def test_cli_circuit_draws_only_the_design_that_was_asked_for(capsys, tmp_path):
    """``--which`` has to filter the drawings, not only the listing."""
    assert main(["circuit", "question-order", "--which", "B then A", "--out", str(tmp_path),
                 "--format", "svg"]) == 0
    written = sorted(path.name for path in tmp_path.iterdir())
    assert len(written) == 1 and "B_then_A" in written[0], written
    assert "B then A" in capsys.readouterr().out


def test_cli_circuit_command(capsys, tmp_path):
    assert main(["circuit", "quantum-prisoners-dilemma"]) == 0
    output = capsys.readouterr().out
    assert "entangling game circuit" in output
    assert "qubits" in output
    assert main(["circuit", "collective-judgment", "--out", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*.png"))


def test_cli_circuit_reports_notes_alongside_the_drawing(capsys):
    """When part of a design cannot be drawn, the CLI says why."""
    assert main(["circuit", "collective-judgment"]) == 0
    output = capsys.readouterr().out
    assert "GHZ preparation" in output
    assert "density matrix" in output
    assert main(["circuit", "decision-tree"]) == 0
    assert "sharp version" in capsys.readouterr().out


def test_reports_include_comparison_circuit_and_caveats(tmp_path):
    model = get_model("quantum-prisoners-dilemma")
    result = model.run(sweeps=False)
    path = write_model_report(model, result, tmp_path)
    html = Path(path).read_text(encoding="utf-8")
    assert "Model versus classical baseline" in html
    assert "Circuit view" in html
    assert "classical equilibrium" in html
    assert html.count("<figure>") >= 2


def test_fast_mode_omits_sweep_contents():
    quick = get_model("question-order").run(sweeps=False)
    full = get_model("question-order").run()
    assert len(quick.analysis["sweeps"]["sharpness"].table.columns) == 1
    assert len(full.analysis["sweeps"]["sharpness"].table.columns) > 1


def test_showcase_script_runs_from_a_checkout():
    """The top-level demo must keep working: it is the first thing a reader runs."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "demos" / "showcase" / "run_showcase.py"),
            "--models",
            "question-order",
            "--fast",
            "--no-figures",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    for section in ("PARAMETERS", "SUMMARY", "FINDINGS", "AGAINST THE CLASSICAL BASELINE",
                    "QUANTUM CIRCUIT", "WHAT THIS DOES NOT CLAIM", "ADJUST THEM"):
        assert section in output, section


def test_showcase_accepts_parameter_overrides(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "demos" / "showcase" / "run_showcase.py"),
            "--models",
            "question-order",
            "--param",
            "angle_a=2.0",
            "--fast",
            "--no-figures",
        ],
        cwd=str(tmp_path),   # also proves the script works from any working directory
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines()]
    assert any(line.startswith("angle_a") and "= 2.0" in line for line in lines), lines[:12]
    assert "overrides: {'angle_a': 2.0}" in completed.stdout


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
    assert "Generated by Q-Cog" in html
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