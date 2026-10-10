"""Generate the model documentation from the code.

``python scripts/generate_docs.py``

The catalogue, the parameter tables, the caveats and the references in this document
are read from the model classes themselves, so the documentation cannot drift from
the implementation: if a parameter's default changes, the docs change with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcog.registry import MODELS, MODULES, models_by_module  # noqa: E402

HEADER = """# Model catalogue (generated)

> **Generated file - do not edit by hand.** Run `python scripts/generate_docs.py`.
>
> This document is produced from the model classes, so the parameters, defaults, bounds, caveats and references
> below always match the code.

"""


def module_block(letter: str, models: list) -> str:
    lines = [f"## Module {letter} - {models[0].module_title()}", ""]
    for model in models:
        lines += [
            f"### `{model.slug}` - {model.title}",
            "",
            f"*{model.one_liner}*",
            "",
            f"- **Cognitive question.** {model.cognitive_question}",
            f"- **Quantum mechanism.** {model.quantum_mechanism}",
            f"- **Classical baseline.** {model.classical_baseline}",
            f"- **Difficulty.** {model.difficulty}",
            f"- **Keywords.** {', '.join(model.keywords) or 'none'}",
            "",
        ]
        if model.params:
            lines += [
                "| parameter | kind | default | bounds | fittable | meaning |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for spec in model.params:
                bounds = (
                    f"{spec.low} to {spec.high}"
                    if spec.low is not None or spec.high is not None
                    else "-"
                )
                choices = f" ({', '.join(str(choice) for choice in spec.choices)})" if spec.choices else ""
                lines.append(
                    f"| `{spec.name}` | {spec.kind}{choices} | {spec.default!r} | {bounds} | "
                    f"{'yes' if spec.fittable else 'no'} | {spec.description} |"
                )
            lines.append("")
        options = getattr(model, "classical_option_names", ())
        lines += [
            "**Adjustable classical comparison.** `classical=False` switches it off; `rows` and `max_rows` select "
            "comparison rows for every model; "
            + (
                "this model's baseline also honours "
                + ", ".join(f"`{name}`" for name in options)
                + "."
                if options
                else "this model's baseline is a structural constraint with no parameters of its own."
            ),
            "",
        ]
        designs = "built from the model's own design"
        lines += [
            f"**Circuit view.** A drawing is produced wherever one exists ({designs}); "
            "`circuit_style`, `circuit_scale` and `circuit_fold` adjust the image, and "
            "`circuit_image(result, name=..., path=...)` writes it in any format.",
            "",
        ]
        if model.caveats:
            lines += ["**What it does not claim.**", ""]
            lines += [f"- {caveat}" for caveat in model.caveats]
            lines.append("")
        if model.references:
            lines += ["**References.**", ""]
            lines += [f"- {reference}" for reference in model.references]
            lines.append("")
        lines += [
            "```bash",
            f"qcog run {model.slug}",
            f"qcog info {model.slug}",
            "```",
            "",
        ]
    return "\n".join(lines)


#: Files that quote the size of the test suite.
COUNT_FILES = ("README.md", "CONTRIBUTING.md", "docs/project_page.md", "docs/methodology.md",
               "docs/qollab_runbook.md", "docs/submission_checklist.md", "docs/accessibility.md",
               "ATTRIBUTION.md", "showcase/README.md")


def count_tests() -> int:
    """Number of tests the suite collects, without importing it."""
    import re
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    collected = 0
    for line in completed.stdout.splitlines():
        text = line.strip()
        summary = re.match(r"^(tests[/\\][^:]+):\s*(\d+)$", text)
        if summary:
            collected += int(summary.group(2))
        elif "::" in text:
            collected += 1
    return collected


def sync_test_counts(write: bool = True) -> list[str]:
    """Make every quoted ``NN tests`` in the documentation match the suite."""
    import re

    current = count_tests()
    changed = []
    for relative in COUNT_FILES:
        path = ROOT / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = re.sub(r"\b\d{2,3}(?=-? tests)", str(current), text)
        updated = re.sub(r"\b\d{2,3}(?=-test)", str(current), updated)
        if updated != text:
            if write:
                path.write_text(updated, encoding="utf-8", newline="\n")
            changed.append(relative)
    return changed


def main() -> int:
    """Write ``docs/models.generated.md``, and align the quoted test counts."""
    parts = [HEADER]
    parts.append(f"**{len(MODELS)} models in {len(MODULES)} modules.**\n")
    for letter, models in models_by_module().items():
        titles = ", ".join(f"`{model.slug}`" for model in models)
        parts.append(f"- Module {letter} ({models[0].module_title()}): {titles}")
    parts.append("")
    for letter, models in models_by_module().items():
        parts.append(module_block(letter, models))
    target = ROOT / "docs" / "models.generated.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    print(f"written {target} ({target.stat().st_size / 1024:.1f} KB, {len(MODELS)} models)")
    if "--no-sync" not in sys.argv:
        changed = sync_test_counts()
        print(
            f"test count synced to {count_tests()} in: {', '.join(changed) if changed else 'nothing to change'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())