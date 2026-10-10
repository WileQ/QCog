"""Build the static website: models, figures, circuits, examples, notebook, study, docs.

The site is generated from the library itself, so it cannot drift from it: the model pages come
from the same metadata, analyses and figures the CLI and the reports use.  It is plain HTML with
one stylesheet and no JavaScript, which is what lets GitHub Pages serve it unchanged and lets a
reader open it from a local folder.

    python -m qcog.website            # or: python scripts/build_website.py
    python -c "from qcog.website import build_site; build_site('website')"
"""

from __future__ import annotations

import datetime
import shutil
import subprocess
import sys
import re
from pathlib import Path

from ..registry import MODELS, MODULES, models_by_module
from ..report import _comparison_table, _esc, _table
from . import markdown

__all__ = ["build_site", "SITE_SECTIONS", "DEFAULT_REPO_URL"]

#: Change this to your repository before publishing; it is used for links to files that the
#: site does not render (source files, datasets) and for the "edit on GitHub" links.
DEFAULT_REPO_URL = "https://github.com/qcog/qcog"

#: The top-level navigation, in order.  ``href`` is relative to the site root.
SITE_SECTIONS = (
    ("index.html", "Home"),
    ("models/index.html", "Models"),
    ("notebook.html", "Notebook"),
    ("examples/index.html", "Examples"),
    ("study.html", "Study"),
    ("docs/index.html", "Docs"),
)


def _nav(active: str, depth: int, repo_url: str) -> str:
    """The site navigation, with the current page marked for assistive technology."""
    prefix = "../" * depth
    items: list[str] = []
    for href, label in SITE_SECTIONS:
        current = ' aria-current="page"' if href == active else ""
        items.append(f'<li><a href="{prefix}{href}"{current}>{label}</a></li>')
    items.append(f'<li><a href="{repo_url}">GitHub</a></li>')
    return (
        '<header class="site"><div class="brand">Q-Cog</div>'
        '<nav class="site" aria-label="Site"><ul>'
        + "".join(items)
        + "</ul></nav></header>"
    )


def _page(
    title: str,
    body: str,
    *,
    active: str,
    depth: int,
    description: str = "",
    repo_url: str = DEFAULT_REPO_URL,
) -> str:
    prefix = "../" * depth
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f'<meta name="description" content="{_esc(description or title)}">\n'
        f'<link rel="stylesheet" href="{prefix}assets/style.css">\n'
        "</head>\n<body>\n"
        '<a class="skip" href="#main">Skip to main content</a>\n'
        + _nav(active, depth, repo_url)
        + f'\n<main id="main">\n{body}\n</main>\n'
        '<footer class="site"><p>Q-Cog, generated from the library. '
        "This effort is supported by Qollab &amp; IonQ.</p></footer>\n"
        "</body>\n</html>\n"
    )


def _figure_block(path: str, name: str, description: str) -> str:
    """A figure with alt text, a caption and the long description in a disclosure."""
    return (
        f'<figure><img src="{_esc(path)}" alt="{_esc(description)}">'
        f"<figcaption>{_esc(name)}. {_esc(description)}</figcaption>"
        f'<details><summary>Long description of this chart</summary><p>{_esc(description)}</p></details>'
        "</figure>"
    )


def _safe(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)[:56]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _model_page(model_class, result, figures: dict, relative: str) -> str:
    """One model page: metadata, parameters, findings, comparison, figures, circuits, caveats."""
    model = model_class()
    parts = [
        f"<h1>{_esc(model.title)}</h1>",
        f'<p class="lede-meta"><span class="badge">module {_esc(model.module)}: '
        f"{_esc(model.module_title())}</span><span class=\"badge\">{_esc(model.difficulty)}</span>"
        f'<span class="badge">slug: {_esc(model.slug)}</span></p>',
        f'<p class="lead">{_esc(result.summary)}</p>',
        f"<h2>What it asks</h2><p>{_esc(model.cognitive_question)}</p>"
        f"<h2>How it works</h2><p>{_esc(model.quantum_mechanism)}</p>"
        f"<p><strong>Classical baseline.</strong> {_esc(model.classical_baseline)}</p>",
    ]
    run_hint = (
        f"qcog run {model.slug}"
        + (f" --classical-option {model.classical_option_names[0]}=..." if model.classical_option_names else "")
    )
    parts.append(
        "<h2>Run it</h2><pre class=\"code\" data-language=\"bash\"><code>"
        + _esc(run_hint)
        + "\nqcog circuit " + _esc(model.slug) + " --style iqp --format svg"
        + "</code></pre>"
    )
    if model.params:
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "parameter": spec.name,
                    "kind": spec.kind,
                    "default": spec.default,
                    "bounds": f"{spec.low} … {spec.high}" if spec.low is not None or spec.high is not None else "",
                    "fittable": "yes" if spec.fittable else "no",
                    "meaning": spec.description,
                }
                for spec in model.params
            ]
        )
        parts.append(_table(frame, "Parameters (every one adjustable from Python and the CLI)"))
    parts.append("<h2>What this run shows</h2><ul>")
    parts += [f"<li>{_esc(finding)}</li>" for finding in result.findings]
    parts.append("</ul>")
    parts.append("<h2>Model versus classical baseline</h2>")
    parts.append(_comparison_table(result))
    for name, frame in result.tables().items():
        if len(frame.index):
            parts.append(_table(frame, f"Data table: {name.replace('_', ' ')}"))
    if figures:
        parts.append("<h2>Figures</h2>")
        for name, information in figures.items():
            parts.append(_figure_block(information["path"], name, information["description"]))
    notes = model.circuit_notes(result)
    if notes or figures:
        parts.append("<h2>Circuit view</h2>")
        if notes:
            parts.append("<ul>" + "".join(f"<li>{_esc(note)}</li>" for note in notes) + "</ul>")
        parts.append(
            '<p class="lede-meta">Each drawing above is a graph produced by Qiskit\'s matplotlib drawer; '
            "the description under it says what the circuit contains, so the information is available without "
            "the image.</p>"
        )
    if result.caveats:
        parts.append(
            '<h2>What this does not claim</h2><div class="caveat"><ul>'
            + "".join(f"<li>{_esc(caveat)}</li>" for caveat in result.caveats)
            + "</ul></div>"
        )
    if model.references:
        parts.append(
            "<h2>References</h2><ul>"
            + "".join(f"<li>{_esc(reference)}</li>" for reference in model.references)
            + "</ul>"
        )
    parts.append(f'<p class="lede-meta">Page for <a href="{relative}">{_esc(model.slug)}</a>.</p>')
    return "\n".join(parts)


def _render_markdown_page(relative: str, root: Path, *, repo_url: str, depth: int) -> tuple[str, str]:
    """Render one Markdown file, rewriting its links for the site."""
    text = (root / relative).read_text(encoding="utf-8")

    def resolve(target: str) -> str | None:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return target
        if target.endswith(".md"):
            candidate = (root / relative).parent / target
            try:
                candidate = candidate.resolve().relative_to(root)
            except ValueError:
                return None
            if (root / candidate).exists():
                stem = candidate.with_suffix("")
                name = "index.html" if stem.name == "README" else f"{stem.name}.html"
                return ("../" * depth) + f"docs/{name}"
        for prefix in ("../", "./"):
            if target.startswith(prefix):
                target = target[len(prefix) :]
        if target.startswith(("docs/", "website/", "demos/", "studies/")):
            return f"{repo_url}/blob/main/{target}"
        return None

    # the file's own top heading becomes the page's h1; anything else at that level is demoted,
    # so every rendered documentation page has exactly one h1
    lines = text.splitlines()
    seen_top = False
    for index, line in enumerate(lines):
        if line.startswith("# "):
            if seen_top:
                lines[index] = "#" + line
            seen_top = True
    text = "\n".join(lines)
    body, headings = markdown.to_html(text, resolve, heading_shift=0)
    title = headings[0][1].replace("-", " ") if headings else relative
    return title, body


def _examples_page(root: Path, name: str, script: Path, output: str | None, repo_url: str) -> tuple[str, str]:
    """One example: its docstring as prose, its source, and the output it produced."""
    text = script.read_text(encoding="utf-8")
    docstring = ""
    if text.startswith('"""'):
        docstring = text[3 : text.index('"""', 3)]
    title = docstring.splitlines()[0] if docstring else script.name
    body = [
        f"<h1>{_esc(title)}</h1>",
        f'<p class="lede-meta">Source: <a href="{repo_url}/blob/main/{name}">{_esc(name)}</a></p>',
    ]
    prose = "\n".join(docstring.splitlines()[1:]).strip()
    if prose:
        body.append(markdown.to_html(prose)[0])
    if output:
        body.append("<h2>Output</h2><pre class=\"code\" data-language=\"text\"><code>" + _esc(output) + "</code></pre>")
    else:
        body.append(
            '<p class="lede-meta">This script is not run while the site is built; run it yourself to see its '
            "output.</p>"
        )
    body.append(
        "<h2>The code</h2><details><summary>Show the whole script</summary><pre class=\"code\" "
        'data-language="python"><code>' + _esc(text) + "</code></pre></details>"
    )
    return title, "\n".join(body)


def _notebook_page(notebook_path: Path) -> tuple[str, int, int]:
    """Render the quickstart notebook: prose, code, and text output (charts are linked)."""
    import json

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    charts = 0
    title_done = False
    parts.append(
        '<p class="lede-meta">The notebook below is rendered from '
        f'<code>{_esc(notebook_path.as_posix())}</code> in the repository; its charts appear on the model pages '
        "with their descriptions.</p>"
    )
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            # the notebook's own title becomes the page's h1; every other heading is shifted down
            # one level so the page keeps exactly one top-level heading
            shift = 0 if (not title_done and source.lstrip().startswith("# ")) else 1
            if shift == 0:
                title_done = True
            shifted = re.sub(
                r"^(#{1,5})\s",
                lambda match: "#" * min(6, len(match.group(1)) + shift) + " ",
                source,
                flags=re.M,
            )
            parts.append(markdown.to_html(shifted)[0])
            continue
        parts.append(
            '<pre class="code" data-language="python"><code>' + _esc(source) + "</code></pre>"
        )
        for output in cell.get("outputs", []):
            if "text" in output:
                parts.append(
                    '<pre class="code" data-language="output"><code>'
                    + _esc("".join(output["text"]))
                    + "</code></pre>"
                )
            elif "data" in output and any(key.startswith("image") for key in output["data"]):
                charts += 1
    if charts:
        parts.append(
            f'<p class="lede-meta">{charts} chart(s) were produced by the cells above and are not repeated here: '
            "every one of them appears, with its description and the data table behind it, on the "
            '<a href="models/index.html">model pages</a>.</p>'
        )
    return "\n".join(parts), charts, len(notebook.get("cells", []))


def build_site(
    out_dir: str | Path = "website",
    *,
    root: str | Path | None = None,
    repo_url: str = DEFAULT_REPO_URL,
    models: list[str] | None = None,
    include_figures: bool = True,
    include_examples: bool = True,
    run_examples: bool = True,
    example_timeout: int = 300,
    include_notebook: bool = True,
    include_docs: bool = True,
    include_study: bool = True,
) -> dict:
    """Generate the whole site and return a summary of what was written."""
    repository = Path(root) if root else Path(__file__).resolve().parents[3]
    site = Path(out_dir)
    assets = site / "assets"
    figures_dir = assets / "figures"
    if site.exists():
        shutil.rmtree(site)
    (assets / "style").mkdir(parents=True, exist_ok=True)

    style_source = Path(__file__).resolve().parent / "style" / "site.css"
    shutil.copyfile(style_source, assets / "style.css")

    summary = {"pages": 0, "figures": 0, "assets": ["assets/style.css"], "sections": {}}

    def write(relative: str, html: str) -> None:
        _write(site / relative, html)
        summary["pages"] += 1

    # ------------------------------------------------------------------ model pages
    slugs = models or list(MODELS)
    catalogue_rows = []
    model_pages = {}
    for slug in slugs:
        model_class = MODELS[slug]
        model = model_class()
        result = model.run()
        figures: dict = {}
        if include_figures:
            figures_dir.mkdir(parents=True, exist_ok=True)
            for name, figure in model.figures(result).items():
                filename = f"{slug}__{_safe(name)}.png"
                figure.savefig(figures_dir / filename, bbox_inches="tight")
                figures[name] = {
                    "path": f"../assets/figures/{filename}",
                    "description": getattr(figure, "_qcog_description", ""),
                }
                summary["figures"] += 1
                import matplotlib.pyplot as plt

                plt.close(figure)
        body = _model_page(model_class, result, figures, "../models/index.html")
        write(
            f"models/{slug}.html",
            _page(
                f"{model.title} - Q-Cog",
                body,
                active="models/index.html",
                depth=1,
                description=model.one_liner,
                repo_url=repo_url,
            ),
        )
        model_pages[slug] = f"{slug}.html"
        catalogue_rows.append(
            {
                "model": f'<a href="{slug}.html">{_esc(model.title)}</a>',
                "module": model.module,
                "what it explores": model.one_liner,
            }
        )

    import pandas as pd

    overview = ["<h1>Models</h1>", f"<p class=\"lead\">{len(slugs)} models in {len(MODULES)} modules. "
                "Every page below is generated from the model itself: its parameters, its findings, its "
                "comparison with a classical baseline, its figures and its circuits.</p>"]
    for letter, grouped in models_by_module().items():
        if not all(model.slug in slugs for model in grouped):
            continue
        overview.append(f"<h2>Module {letter}: {_esc(grouped[0].module_title())}</h2>")
        overview.append(f'<p class="lede-meta">{len(grouped)} models.</p><ul class="cards">')
        for model in grouped:
            overview.append(
                f'<li><h3><a href="{model.slug}.html">{_esc(model.title)}</a></h3>'
                f'<p class="lede-meta">{_esc(model.difficulty)} · {_esc(model.slug)}</p>'
                f"<p>{_esc(model.one_liner)}</p></li>"
            )
        overview.append("</ul>")
    write(
        "models/index.html",
        _page(
            "Models - Q-Cog",
            "\n".join(overview),
            active="models/index.html",
            depth=1,
            description="Every Q-Cog model, with its figures, its classical comparison and its circuits.",
            repo_url=repo_url,
        ),
    )
    summary["sections"]["models"] = len(slugs)

    # ------------------------------------------------------------------ the study
    if include_study:
        study_dir = repository / "studies" / "quantum_advantage"
        if (study_dir / "REPORT.md").exists():
            if include_figures:
                for figure in sorted((study_dir / "figures").glob("*.png")):
                    shutil.copyfile(figure, figures_dir / f"study__{figure.name}")
                    summary["assets"].append(f"assets/figures/study__{figure.name}")
            text = (study_dir / "REPORT.md").read_text(encoding="utf-8")
            text = text.replace("](figures/", "](assets/figures/study__")
            body, _headings = markdown.to_html(text, lambda target: target)
            write(
                "study.html",
                _page(
                    "Where the quantum account does better - Q-Cog",
                    body,
                    active="study.html",
                    depth=0,
                    description="Five measured cases where the quantum account beats the classical one.",
                    repo_url=repo_url,
                ),
            )
            summary["sections"]["study"] = 1

    # ------------------------------------------------------------------ the notebook
    if include_notebook:
        notebook = repository / "demos" / "quickstart.ipynb"
        if notebook.exists():
            body, charts, cells = _notebook_page(notebook)
            write(
                "notebook.html",
                _page(
                    "Quickstart notebook - Q-Cog",
                    body,
                    active="notebook.html",
                    depth=0,
                    description="The quickstart notebook: a two-question survey, end to end.",
                    repo_url=repo_url,
                ),
            )
            summary["sections"]["notebook"] = {"cells": cells, "charts": charts}

    # ------------------------------------------------------------------ the examples
    if include_examples:
        examples_dir = repository / "demos" / "examples"
        scripts = sorted(examples_dir.glob("*.py")) if examples_dir.exists() else []
        landing = ["<h1>Examples</h1>",
                   '<p class="lead">Runnable scripts, one task each. The output shown is what the script prints '
                   'when it runs from the repository root.</p><ul class="cards">']
        for script in scripts:
            relative = f"demos/examples/{script.name}"
            output = None
            if run_examples and not script.name.startswith("05"):
                try:
                    completed = subprocess.run(
                        [sys.executable, str(script)], cwd=str(repository), capture_output=True,
                        text=True, timeout=example_timeout,
                    )
                    output = completed.stdout if completed.returncode == 0 else (
                        completed.stdout + "\n[exit code " + str(completed.returncode) + "]\n" + completed.stderr
                    )
                except Exception as error:  # pragma: no cover - reported in the page
                    output = f"[the site build could not run this script: {type(error).__name__}: {error}]"
            title, body = _examples_page(repository, relative, script, output, repo_url)
            page = script.stem + ".html"
            write(
                f"examples/{page}",
                _page(
                    f"{title} - Q-Cog",
                    body,
                    active="examples/index.html",
                    depth=1,
                    description=title,
                    repo_url=repo_url,
                ),
            )
            landing.append(
                f'<li><h3><a href="{page}">{_esc(title)}</a></h3>'
                f'<p class="lede-meta">{_esc(script.name)}</p></li>'
            )
        landing.append("</ul>")
        write(
            "examples/index.html",
            _page(
                "Examples - Q-Cog",
                "\n".join(landing),
                active="examples/index.html",
                depth=1,
                description="Six runnable Q-Cog examples, with their output.",
                repo_url=repo_url,
            ),
        )
        summary["sections"]["examples"] = len(scripts)

    # ------------------------------------------------------------------ the docs
    if include_docs:
        documentation = ["README.md", "ATTRIBUTION.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"]
        documentation += [f"docs/{path.name}" for path in sorted((repository / "docs").glob("*.md"))]
        index = ["<h1>Documentation</h1>",
                 '<p class="lead">The written documentation, rendered from the repository. The generated '
                 'reference for every model and parameter is in <a href="../models/index.html">Models</a>.</p><ul>']
        for relative in documentation:
            if not (repository / relative).exists():
                continue
            title, body = _render_markdown_page(relative, repository, repo_url=repo_url, depth=1)
            name = "index" if relative == "README.md" else Path(relative).stem
            write(
                f"docs/{name}.html",
                _page(
                    f"{title} - Q-Cog",
                    body,
                    active="docs/index.html",
                    depth=1,
                    description=title,
                    repo_url=repo_url,
                ),
            )
            index.append(f'<li><a href="{name}.html">{_esc(title)}</a> <span class="lede-meta">{_esc(relative)}</span></li>')
        index.append("</ul>")
        write(
            "docs/index.html",
            _page(
                "Documentation - Q-Cog",
                "\n".join(index),
                active="docs/index.html",
                depth=1,
                description="The Q-Cog documentation.",
                repo_url=repo_url,
            ),
        )
        summary["sections"]["docs"] = len(documentation)
        summary["sections"]["model_pages"] = len(model_pages)

    # ------------------------------------------------------------------ the home page
    study_cases = ""
    study_path = repository / "studies" / "quantum_advantage" / "results.json"
    if study_path.exists():
        import json

        payload = json.loads(study_path.read_text(encoding="utf-8"))
        rows = []
        for case in payload["cases"]:
            first = case["metrics"][0]
            rows.append(
                {
                    "case": case["title"],
                    "sense": case["sense"].split(" (")[0],
                    "headline": f"{first['metric']}: {first['quantum']:.3f} against {first['classical']:.3f}",
                }
            )
        study_cases = _table(pd.DataFrame(rows), "Where the quantum account does better")
    home = [
        "<h1>Q-Cog</h1>",
        '<p class="lead">A Qiskit-native library for modelling, simulating and explaining quantum-probability '
        "models of human judgement, decision making and collective behaviour.</p>",
        f'<p><span class="badge">{len(MODELS)} models</span><span class="badge">{len(MODULES)} modules</span>'
        f'<span class="badge">circuits on every page</span><span class="badge">a classical baseline everywhere</span></p>',
        "<h2>Start here</h2>",
        '<ul class="cards">'
        '<li><h3><a href="models/index.html">The models</a></h3><p>Twenty-three documented models in four '
        "modules, each with its parameters, findings, classical comparison and circuit drawings.</p></li>"
        '<li><h3><a href="notebook.html">The notebook</a></h3><p>One survey asked in both orders, end to end: '
        "the order effect, the classical identity that fails, the circuits, a simulator run.</p></li>"
        '<li><h3><a href="examples/index.html">The examples</a></h3><p>Six runnable scripts with their output, '
        "from a five-minute first look to the full catalogue.</p></li>"
        '<li><h3><a href="study.html">The comparative study</a></h3><p>Five measured cases where the quantum '
        "account beats the classical one - and two where a classical model survives.</p></li>"
        "</ul>",
        "<h2>What it does</h2>",
        "<p>Every model is simulated with <strong>Qiskit</strong>, reports the same quantity computed by a "
        "<strong>classical</strong> account of the same situation, states the <strong>caveats</strong> that limit "
        "its claims, and draws its <strong>circuit</strong> wherever one exists. The library checks itself: "
        "the self-test table, the figure checks and the documentation checks run in the test suite.</p>",
        "<h2>Install</h2>",
        '<pre class="code" data-language="bash"><code>pip install qcog            # once released\n'
        "git clone &lt;this repository&gt; &amp;&amp; cd qcog &amp;&amp; pip install -e .   # from a checkout</code></pre>",
        "<h2>The five comparative cases</h2>" if study_cases else "",
        study_cases,
        f'<p class="lede-meta">Generated from the library; the model pages come from the same metadata, '
        f'analyses and figures the CLI and the HTML reports use.</p>',
    ]
    write(
        "index.html",
        _page(
            "Q-Cog - quantum cognition, with the circuits and the classical comparison",
            "\n".join(part for part in home if part),
            active="index.html",
            depth=0,
            description="Q-Cog: a Qiskit-native library for quantum-probability models of cognition, with figures, "
            "circuits and a classical comparison for every model.",
            repo_url=repo_url,
        ),
    )
    _write(site / ".nojekyll", "")
    summary["assets"].append(".nojekyll")
    return summary


if __name__ == "__main__":  # pragma: no cover - the scripts/ entry point is the documented one
    outcome = build_site(sys.argv[1] if len(sys.argv) > 1 else "website")
    print(outcome)