"""Accessible HTML and Markdown reports.

The report generator is the "publish it" step: it turns any model run into a
self-contained HTML page that works with no JavaScript, no server and no network -
which is what makes it usable on Qollab, in a pull request, or as a submission
attachment.  Accessibility is built in rather than bolted on: semantic landmarks, a
skip link, a heading hierarchy, a caption and a text description for every figure, a
data table for every chart, visible focus outlines and high-contrast colours.

The same generator serves three audiences: ``write_model_report`` for one run,
``write_catalogue_report`` for the whole library, and ``write_markdown_report`` for
somewhere a plain text file fits better.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from .registry import MODELS, catalogue
from .viz.figures import savefig, text_alternatives

__all__ = [
    "write_model_report",
    "write_catalogue_report",
    "write_markdown_report",
    "model_html",
    "CATALOGUE_INTRO",
    "CSS",
]

CATALOGUE_INTRO = (
    "Q-Cog is a Qiskit-native library for modelling, simulating and explaining quantum-probability models of "
    "human judgement, decision making and collective behaviour. Every model below reports what it simulates, "
    "what a classical account of the same situation predicts, and what it does not claim."
)

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; line-height: 1.55;
       margin: 0 auto; max-width: 60rem; padding: 1.5rem 1.25rem 4rem; color: #111; background: #fff; }
a { color: #00539b; }
a:focus-visible, button:focus-visible, summary:focus-visible, input:focus-visible,
textarea:focus-visible, select:focus-visible { outline: 3px solid #d55e00; outline-offset: 2px; }
.skip { position: absolute; left: -9999px; top: 0; background: #fff; padding: .6rem 1rem; }
.skip:focus { left: 0; z-index: 10; }
h1 { font-size: 1.9rem; margin: 0 0 .3rem; }
h2 { font-size: 1.35rem; margin-top: 2.4rem; border-bottom: 2px solid #eee; padding-bottom: .3rem; }
h3 { font-size: 1.1rem; margin-top: 1.6rem; }
p.lead { font-size: 1.05rem; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .95rem; }
caption { text-align: left; font-weight: 600; padding-bottom: .4rem; }
th, td { border: 1px solid #ccc; padding: .35rem .5rem; text-align: left; }
th { background: #f2f6fa; }
figure { margin: 1.4rem 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #ddd; }
figcaption { font-size: .92rem; color: #333; margin-top: .4rem; }
details { margin: .5rem 0 1rem; }
summary { cursor: pointer; font-weight: 600; }
.meta { color: #444; font-size: .95rem; }
.badge { display: inline-block; border: 1px solid #999; border-radius: .4rem; padding: 0 .4rem;
         font-size: .8rem; margin-right: .3rem; }
.caveat { background: #fff8e6; border-left: 4px solid #e69f00; padding: .6rem .8rem; margin: .8rem 0; }
.finding { margin: .4rem 0; }
.pass { color: #006b3c; font-weight: 600; }
.fail { color: #b3261e; font-weight: 600; }
nav ol { padding-left: 1.2rem; }
@media (prefers-color-scheme: dark) {
  body { background: #111; color: #eee; }
  a { color: #8ab8e8; }
  th { background: #1d2530; }
  th, td { border-color: #444; }
  .caveat { background: #2a2415; }
  figure img { border-color: #444; }
}
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _table(frame, caption: str) -> str:
    """A data table with a caption, headers and scope attributes."""
    header = "".join(f'<th scope="col">{_esc(column)}</th>' for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td>{value:.4f}</td>' if isinstance(value, float) else f"<td>{_esc(value)}</td>"
            for value in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><caption>{_esc(caption)}</caption><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _figure_html(name: str, figure, description: str) -> str:
    """A figure with alt text and an expandable long description."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    long_description = " ".join(getattr(figure, "_qcog_extra", []) or [])
    body = f'<p>{_esc(long_description)}</p>' if long_description else ""
    return (
        f"<figure><img src=\"data:image/png;base64,{encoded}\" alt=\"{_esc(description)}\">"
        f"<figcaption>{_esc(name)}. {_esc(description)}</figcaption>"
        f"<details><summary>Long description of this chart</summary>{body or '<p>' + _esc(description) + '</p>'}"
        f"</details></figure>"
    )


def _as_instance(model):
    """Accept either a model class or a model instance."""
    return model() if isinstance(model, type) else model


def model_html(model, result=None, *, include_figures: bool = True, heading_level: int = 2) -> str:
    """The HTML section for one model (optionally running it first).

    ``model`` may be a class or an instance; both are accepted because the CLI, the
    report generator and the tests all reach this function from different places.
    """
    instance = _as_instance(model)
    if result is None:
        result = instance.run()
    level = max(2, min(6, heading_level))
    heading = "h" + str(level)
    sub = "h" + str(min(6, level + 1))
    subsub = "h" + str(min(6, level + 2))

    parts = [f"<section id=\"{_esc(result.slug)}\" aria-labelledby=\"{_esc(result.slug)}-title\">"]
    parts.append(
        f'<{heading} id="{_esc(result.slug)}-title">{_esc(result.title)}</{heading}>'
        f'<p class="meta"><span class="badge">module {_esc(result.module)}</span>'
        f'<span class="badge">{_esc(result.backend)}</span>'
        f'<span class="badge">{len(result.findings)} findings</span></p>'
        f'<p class="lead">{_esc(result.summary)}</p>'
    )
    parts.append(
        f"<{sub}>What this model asks</{sub}><p>{_esc(model.cognitive_question)}</p>"
        f"<{sub}>How it works</{sub}><p>{_esc(model.quantum_mechanism)}</p>"
        f"<p><strong>Classical baseline.</strong> {_esc(model.classical_baseline)}</p>"
    )
    if model.references:
        items = "".join(f"<li>{_esc(reference)}</li>" for reference in model.references)
        parts.append(f"<{sub}>References</{sub}><ul>{items}</ul>")

    parts.append(f"<{sub}>What this run shows</{sub}>")
    parts.append(
        "<ul class=\"findings\">"
        + "".join(f'<li class="finding">{_esc(finding)}</li>' for finding in result.findings)
        + "</ul>"
    )
    parameters = ", ".join(f"{key} = {value}" for key, value in result.parameters.items())
    parts.append(
        f'<p class="meta">Parameters: {_esc(parameters)}. Backend: {_esc(result.backend)}'
        + (f", {result.shots} shots, seed {result.seed}" if result.shots else " (exact)")
        + f". Runtime {result.metadata.get('runtime_seconds', 0.0):.4f} s.</p>"
    )

    for name, frame in result.tables().items():
        if len(frame.index) > 0:
            parts.append(_table(frame, f"Data table: {name.replace('_', ' ')}"))

    if include_figures:
        try:
            figures = instance.figures(result)
        except Exception as error:  # pragma: no cover - plotting must never break a report
            figures = {}
            parts.append(f"<p class=\"meta\">Figures unavailable: {_esc(error)}</p>")
        descriptions = text_alternatives(figures)
        for name, figure in figures.items():
            parts.append(_figure_html(name, figure, descriptions.get(name, "")))

    if result.caveats:
        items = "".join(f"<li>{_esc(caveat)}</li>" for caveat in result.caveats)
        parts.append(
            f"<{sub}>What this does not claim</{sub}>"
            f'<div class="caveat"><ul>{items}</ul></div>'
        )
    parts.append("</section>")
    return "\n".join(parts)


def _page(title: str, body: str, description: str = "") -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f'<meta name="description" content="{_esc(description or title)}">\n'
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        '<a class="skip" href="#main">Skip to main content</a>\n'
        "<header><h1>" + _esc(title) + "</h1></header>\n"
        '<main id="main">' + body + "</main>\n"
        "<footer><p>Generated by Q-Cog. This effort is supported by Qollab &amp; IonQ.</p></footer>\n"
        "</body>\n</html>\n"
    )


def write_model_report(model, result, out_dir: str | Path, include_figures: bool = True) -> Path:
    """Write an HTML report for a single run (plus a PNG of every figure)."""
    instance = _as_instance(model)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    slug = instance.slug
    body = model_html(instance, result, include_figures=include_figures, heading_level=2)
    path = directory / f"{slug}.html"
    path.write_text(_page(result.title, body, result.summary), encoding="utf-8")
    if include_figures:
        try:
            figures = instance.figures(result)
        except Exception:  # pragma: no cover
            figures = {}
        for name, figure in figures.items():
            safe = "".join(character if character.isalnum() else "_" for character in name)[:60]
            savefig(figure, directory / f"{slug}_{safe}.png")
    return path


def write_catalogue_report(
    out_path: str | Path = "out/qcog_report.html",
    max_models: int | None = None,
    include_figures: bool = True,
) -> Path:
    """Write a report covering the whole catalogue, plus the library's evidence table."""
    from .selftest import evidence_frame

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    models = list(MODELS.values())
    if max_models is not None:
        models = models[:max_models]

    body: list[str] = [
        f"<p class=\"lead\">{_esc(CATALOGUE_INTRO)}</p>",
        "<nav aria-label=\"Contents\"><h2>Contents</h2><ol>",
    ]
    for letter in sorted({model.module for model in models}):
        grouped = [model for model in models if model.module == letter]
        body.append(f"<li>Module {letter}: {_esc(grouped[0].module_title())}<ul>")
        for model in grouped:
            body.append(f'<li><a href="#{_esc(model.slug)}">{_esc(model.title)}</a></li>')
        body.append("</ul></li>")
    body.append('<li><a href="#evidence">Evidence: what the library checks about itself</a></li>')
    body.append("</ol></nav>")

    for model in models:
        try:
            result = model().run()
            body.append(model_html(model, result, include_figures=include_figures, heading_level=2))
        except Exception as error:  # pragma: no cover - a failing model must not break the report
            body.append(
                f"<section><h2>{_esc(model.title)}</h2>"
                f"<p class=\"meta\">This model could not be run in the current environment: "
                f"{_esc(type(error).__name__)}: {_esc(error)}</p></section>"
            )

    frame = evidence_frame()
    rows = "".join(
        f"<tr><td>{_esc(row['check'])}</td>"
        f"<td class=\"{'pass' if row['passed'] else 'fail'}\">{_esc(row['result'])}</td>"
        f"<td>{_esc(row['detail'])}</td></tr>"
        for _, row in frame.iterrows()
    )
    body.append(
        '<section id="evidence" aria-labelledby="evidence-title"><h2 id="evidence-title">'
        "Evidence: what the library checks about itself</h2>"
        "<p>Each row below is executed when the report is generated. A failure would be visible here rather "
        "than silently ignored.</p>"
        "<table><caption>Self-checks</caption><thead><tr><th scope=\"col\">check</th>"
        "<th scope=\"col\">result</th><th scope=\"col\">detail</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )
    path.write_text(_page("Q-Cog model catalogue", "\n".join(body), CATALOGUE_INTRO), encoding="utf-8")
    return path


def write_markdown_report(out_path: str | Path = "out/qcog_report.md", max_models: int | None = None) -> Path:
    """Write a Markdown version of the catalogue report (no figures, plain tables)."""
    models = list(MODELS.values())
    if max_models is not None:
        models = models[:max_models]
    lines = ["# Q-Cog model catalogue", "", CATALOGUE_INTRO, ""]
    for model in models:
        result = model().run()
        lines += [
            f"## {result.title}",
            "",
            f"*Module {result.module} - {model.module_title()}.* {result.summary}",
            "",
            f"**Question.** {model.cognitive_question}",
            "",
            f"**Mechanism.** {model.quantum_mechanism}",
            "",
            f"**Classical baseline.** {model.classical_baseline}",
            "",
            "**Findings.**",
            "",
        ]
        lines += [f"- {finding}" for finding in result.findings]
        lines += ["", "**Does not claim.**", ""]
        lines += [f"- {caveat}" for caveat in result.caveats]
        lines += ["", "| response | probability |", "| --- | --- |"]
        for name, item in result.sequential_results().items():
            for _, row in item.marginal_frame().iterrows():
                lines.append(
                    f"| {name}: {row['question']} = {row['outcome']} | {row['probability']:.4f} |"
                )
        lines.append("")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path