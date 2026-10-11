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

import matplotlib.pyplot as plt

from ..circuits.builder import circuit_statistics
from ..registry import MODELS, MODULES, models_by_module
from ..report import _comparison_table, _esc, _table
from ..viz.style import monochrome
from . import markdown

__all__ = ["build_site", "SITE_SECTIONS", "DEFAULT_REPO_URL"]

#: Change this to your repository before publishing; it is used for links to files that the
#: site does not render (source files, datasets) and for the "edit on GitHub" links.
DEFAULT_REPO_URL = "https://github.com/WileQ/QCog"

#: The top-level navigation, in order.  ``href`` is relative to the site root.
SITE_SECTIONS = (
    ("index.html", "Home"),
    ("models/index.html", "Models"),
    ("circuits.html", "Circuits"),
    ("notebook.html", "Quickstart"),
    ("examples/index.html", "Examples"),
    ("study.html", "Study"),
    ("docs/index.html", "Docs"),
)

#: What each example is for: the scenario it sets up, what to look for in its output, and what
#: the numbers mean.  Written by hand, because it is the part a reader cannot get from the code.
EXAMPLE_GUIDES = {
    "01_first_steps": {
        "one_liner": "One belief, two questions, both orders: the shortest complete example.",
        "scenario": "A respondent has an attitude you cannot see. You ask them one question, then a "
        "second, related one - and then you ask the same two in the opposite order to somebody else. A "
        "classical account says the order cannot matter: both answers come out of one joint "
        "distribution, so the marginal of the first question is fixed. Here it is not.",
        "look_for": [
            "the belief state and the two questions, in the first block;",
            "the two joint distributions, one per order;",
            "the order effect - how far the first question's marginal moved - and the QQ equality, a "
            "prediction with no free parameters;",
            "the classical comparison, which is the point of the exercise.",
        ],
        "reading": "A non-zero order effect already excludes any single joint distribution: no "
        "arrangement of one probability space moves a marginal because of the order the questions were "
        "asked in. The QQ line is the part the model cannot tune, which is what makes it a test of the "
        "theory rather than a description of one survey.",
    },
    "02_question_order": {
        "one_liner": "The flagship model run three ways: exactly, on the compiled circuit, and with noise.",
        "scenario": "The same two-question design as the first example, run the way a study would run "
        "it: once analytically, once by compiling the design into a Qiskit circuit and reading the "
        "result, and once with depolarising noise applied between the two questions. The last block "
        "writes an HTML report of the run.",
        "look_for": [
            "three summaries of one model, which should tell the same story;",
            "the compiled circuit agreeing with the analytic numbers - that agreement is what makes the "
            "circuit drawing on the model page worth trusting;",
            "the noisy run, where the interference term shrinks;",
            "the path of the report it writes.",
        ],
        "reading": "Nothing in the second run is a re-implementation: the same objects are compiled and "
        "evaluated, so a disagreement between them would be a bug rather than a modelling choice. Noise "
        "is where the claim gets testable in practice - an order effect that survives noise is a "
        "stronger statement than one that needs perfect coherence.",
    },
    "03_hardware_and_ionq": {
        "one_liner": "Compiling for a device, and running the same circuits on IonQ if a token is set.",
        "scenario": "Hardware imposes a gate set and does not allow mid-circuit measurement. The script "
        "builds the two constructions the library ships - one circuit for the whole interview, and one "
        "per branch, which avoids mid-circuit measurement and reset altogether - prints what each costs "
        "in depth and gates, and, only if an IonQ token is in the environment, submits the same circuits "
        "to IonQ.",
        "look_for": [
            "the two compilations and their depth;",
            "the note saying whether IonQ is available here;",
            "the reminder that no code changes are needed to switch backend.",
        ],
        "reading": "The point is portability: the free local simulator is enough for everything in "
        "this repository, and the same circuits run on IonQ without a rewrite. The branch "
        "construction exists because restricted devices cannot measure mid-circuit, which is exactly the "
        "operation the interview construction relies on.",
    },
    "04_fit_to_data": {
        "one_liner": "Fitting a model to observed frequencies, and comparing it with a classical baseline.",
        "scenario": "You have response frequencies from a study, not a theory. The script fits the "
        "model's parameters to the observed joint distribution, fits a classical alternative to the same "
        "numbers, and compares the two - the workflow you would use with real data in place of the "
        "illustrative synthetic file shipped here.",
        "look_for": [
            "the fitted parameters, and how far they moved from the defaults;",
            "the two fits' scores side by side;",
            "the note that the data file is synthetic.",
        ],
        "reading": "A fit is a parameter search inside the model's own declared bounds, so the "
        "parameters it reports are always legal; what matters is whether the fitted quantum model beats "
        "the classical one on the same data. Read this as a method demonstration, not an empirical "
        "claim.",
    },
    "05_catalogue_report": {
        "one_liner": "Every model in the catalogue, the self-check table, and the full report.",
        "scenario": "The whole library in one run: every model in all four modules, the evidence table "
        "the library keeps about itself, and a self-contained HTML report - no JavaScript, no network, "
        "every chart with its description and the data table behind it - that can be published as it is.",
        "look_for": [
            "one block per module, with each model's headline;",
            "the evidence table, which is the library checking its own identities;",
            "the paths of the two reports it writes.",
        ],
        "reading": "The evidence table is the useful part: it checks identities that have to hold - the "
        "interference decomposition adding up, the QQ equality, Condorcet's theorem at zero correlation - "
        "rather than printing numbers to admire. The report is what a project page can link to.",
    },
    "06_economy_and_vote": {
        "one_liner": "A real survey, end to end, with the circuits drawn.",
        "scenario": "Question A: &quot;Are you satisfied with the economy?&quot; Question B: &quot;Will "
        "you vote for the current administration?&quot; Half the respondents are asked A first, half B "
        "first. In a classical account that cannot matter; in this model the two questions are "
        "incompatible, so answering one changes the state the other is put to. The script prints the "
        "numbers and then draws the circuits that produce them, with the survey questions on the gates.",
        "look_for": [
            "the marginal of question A in each order, and the shift between them;",
            "the law-of-total-probability check, which a classical account satisfies exactly;",
            "the QQ equality, which the model satisfies without being fitted;",
            "the circuit files it writes into out/economy_vote/.",
        ],
        "reading": "The shift is the whole argument: a classical joint distribution has no way to "
        "produce two different marginals for the same question. The circuits are the mechanism made "
        "visible - prepare the belief, rotate into the first question's basis, measure, rotate back, ask "
        "the second.",
    },
}


#: Short, real questions the library answers, used on the examples page and the home page.
REAL_WORLD = (
    (
        "question-order",
        "Does the answer change if the other question is asked first?",
        "A survey asks &quot;Are you satisfied with the economy?&quot; and &quot;Will you vote for the "
        "current administration?&quot;, in both orders. A classical joint distribution forces the two "
        "orders to give the first question the same marginal.",
        "order effect on the first question",
    ),
    (
        "collective-judgment",
        "Does a group decide better than one person?",
        "Seven judges, each right 70% of the time. Condorcet's theorem says the majority is right 87% of "
        "the time - if the judges are independent. Correlated judges do worse, and the model measures "
        "how much.",
        "majority accuracy",
    ),
    (
        "quantum-prisoners-dilemma",
        "Can two players cooperate without trusting each other?",
        "The Eisert-Wilkens-Lewenstein scheme: an entangling gate, then each player's own unitary, then a "
        "measurement. At full entanglement the classical temptation to defect is gone.",
        "expected payoff for the first player",
    ),
    (
        "auctions",
        "Does thinking about rivals change what you bid?",
        "The value estimate and the bid are different questions, so asking them in a different order - or "
        "thinking about the competition - moves the bid. A risk-neutral classical bidder cannot be moved.",
        "shift caused by information about rivals",
    ),
    (
        "semantic-ambiguity",
        "Which sense of an ambiguous word does a sentence select?",
        "A context rotates the meaning of a word before the sense is fixed. Once the sense has been "
        "resolved, a later context has no effect - which a prior-updating account cannot reproduce.",
        "change in p(dominant sense) caused by context",
    ),
    (
        "information-cascades",
        "When does a crowd stop listening to its own information?",
        "Agents decide in sequence and see what the others did. A cascade starts when the history "
        "outweighs any private signal, and the model reports the position where that happens.",
        "adoption share",
    ),
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
        '<div class="tagline">quantum cognition &middot; Qiskit &middot; black and white</div>'
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
        '<footer class="site"><p>Q-Cog, generated from the library itself &mdash; plain HTML, one '
        'stylesheet, no JavaScript, and every chart a graph with a written description.</p>'
        f'<p><a href="{prefix}index.html">Home</a> &middot; <a href="{prefix}circuits.html">Circuits</a> '
        f'&middot; <a href="{prefix}study.html">The five cases</a> &middot; <a href="{repo_url}">Source</a></p>'
        '</footer>\n'
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


def _html_table(rows: list[dict], caption: str) -> str:
    """A table whose cells are already HTML.

    ``report._table`` escapes every cell, which is right for data and wrong for a cell that is a
    link; this one leaves the caller's markup alone and escapes only the caption.
    """
    if not rows:
        return ""
    header = "".join(f'<th scope="col">{_esc(column)}</th>' for column in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>" for row in rows
    )
    return (
        '<div class="table-wrap"><table>'
        f"<caption>{_esc(caption)}</caption>"
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _anchor(slug: str, design: str) -> str:
    """The id of one circuit design on the circuits page, and the link to it."""
    return "c-" + re.sub(r"[^a-z0-9]+", "-", f"{slug}-{design}".lower()).strip("-")


def _safe(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)[:56]


def _greyscale(path: Path) -> None:
    """Rewrite a saved figure as 8-bit greyscale.

    The site is black and white; this is the guarantee, whatever a figure happens to be drawn
    with.
    """
    try:
        from PIL import Image

        with Image.open(path) as handle:
            handle.convert("L").save(path, optimize=True)
    except Exception:  # pragma: no cover - a missing Pillow must not stop the build
        return


def _circuit_text(circuit) -> str:
    """Qiskit's ASCII drawing of a circuit, for the terminal-styled blocks."""
    try:
        return circuit.draw("text")
    except Exception:  # pragma: no cover - a drawer failure must not stop the build
        return ""


def _circuit_material(model, result) -> list[dict]:
    """Everything the site shows about a model's circuits: statistics and the ASCII drawing."""
    material: list[dict] = []
    for name, circuit in sorted(model.circuit_designs(result).items()):
        statistics = circuit_statistics(circuit)
        material.append(
            {
                "name": name,
                "text": _circuit_text(circuit),
                "qubits": statistics["qubits"],
                "clbits": statistics["clbits"],
                "depth": statistics["depth"],
                "size": statistics["size"],
                "operations": statistics["operations"],
            }
        )
    return material


def _circuit_block(design: dict, figure_path: str | None, description: str = "") -> str:
    """One circuit: the drawing, the terminal-style diagram and what it is made of."""
    operations = ", ".join(f"{op} &times;{count}" for op, count in sorted(design["operations"].items()))
    parts = [f'<h3>{_esc(design["name"])}</h3>']
    if figure_path:
        parts.append(_figure_block(figure_path, design["name"], description or design["name"]))
    if design["text"]:
        parts.append('<pre class="terminal"><code>' + _esc(design["text"]) + "</code></pre>")
    parts.append(
        '<p class="lede-meta">'
        f'{design["qubits"]} qubit(s) &middot; {design["clbits"]} classical bit(s) &middot; '
        f'depth {design["depth"]} &middot; {design["size"]} gate(s) &middot; {operations}</p>'
    )
    return "\n".join(parts)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _model_page(model_class, result, figures: dict, relative: str, circuits: list | None = None) -> str:
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
            if circuits and name.startswith("circuit: "):
                continue          # drawn again, with the text diagram, under Circuits
            parts.append(_figure_block(information["path"], name, information["description"]))
    if circuits:
        parts.append("<h2>Circuits</h2>")
        parts.append(
            '<p class="lede-meta">Every design this model compiles, drawn by Qiskit and written out as '
            "text. The drawing carries a description; the text diagram is the same circuit without the "
            "image.</p>"
        )
        for design in circuits:
            figure_path = None
            information = figures.get(f"circuit: {design['name']}")
            if information:
                figure_path = information["path"]
            parts.append(
                _circuit_block(design, figure_path, (information or {}).get("description", ""))
            )
    notes = model.circuit_notes(result)
    if notes:
        parts.append(
            '<div class="note"><p><strong>Why some parts cannot be drawn.</strong></p><ul>'
            + "".join(f"<li>{_esc(note)}</li>" for note in notes)
            + "</ul></div>"
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


def _docs_page_name(relative: str) -> str:
    """The file the site writes for one rendered Markdown document, unique per source file."""
    if relative == "README.md":
        # not "index": that is the documentation index page, and the README would overwrite it
        return "readme"
    if relative.startswith("docs/"):
        return Path(relative).stem
    return re.sub(r"[^a-z0-9]+", "-", relative.lower()).strip("-").removesuffix("-md")


def _render_markdown_page(
    relative: str, root: Path, *, repo_url: str, depth: int, documentation: frozenset[str] = frozenset()
) -> tuple[str, str]:
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
                # a document the site renders becomes a page; anything else is a source link
                if str(candidate) in documentation:
                    return ("../" * depth) + f"docs/{_docs_page_name(str(candidate))}.html"
                return f"{repo_url}/blob/main/{candidate}"
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
    guide = EXAMPLE_GUIDES.get(script.stem)
    if guide:
        body.append(f"<h2>The scenario</h2><p>{guide['scenario']}</p>")
        body.append(
            "<h2>What to look for in the output</h2><ul>"
            + "".join(f"<li>{item}</li>" for item in guide["look_for"])
            + "</ul>"
        )
        body.append(f"<h2>What the numbers mean</h2><p>{guide['reading']}</p>")
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


def _notebook_page(
    notebook_path: Path, figures_dir: Path, root: Path | None = None
) -> tuple[str, int, int]:
    """Render the quickstart notebook: prose, code, text output, and the charts it produced."""
    import base64
    import json

    try:  # the notebook is named relative to the repository, never by its absolute path
        shown = notebook_path.relative_to(root).as_posix() if root else notebook_path.as_posix()
    except ValueError:  # pragma: no cover - only if the notebook sits outside the checkout
        shown = notebook_path.as_posix()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    charts = 0
    title_done = False
    heading = "the notebook"
    parts.append(
        '<p class="lede-meta">The notebook below is rendered from '
        f'<code>{_esc(shown)}</code> in the repository, with the charts its cells '
        "produced. Run it with <code>jupyter lab demos/quickstart.ipynb</code> to do the same yourself."
        "</p>"
    )
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            for line in source.splitlines():
                if line.startswith("#"):
                    heading = line.lstrip("# ").strip()
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
            data = output.get("data", {})
            image = next((key for key in data if key.startswith("image/")), None)
            if image is None:
                continue
            charts += 1
            suffix = image.split("/", 1)[1].replace("jpeg", "jpg")
            filename = f"notebook__{charts:02d}.{suffix}"
            target = figures_dir / filename
            raw = data[image]
            if isinstance(raw, str):
                target.write_bytes(base64.b64decode(raw))
            else:  # a list of text lines, which some kernels emit
                target.write_text("".join(raw), encoding="utf-8")
            _greyscale(target)
            caption = (
                f"Figure {charts} of the notebook - a chart produced by the cell above, from the "
                f"section &lsquo;{heading}&rsquo;. Every chart in the notebook is also on the model "
                "pages, with its description and the data table behind it."
            )
            parts.append(
                f'<figure><img src="assets/figures/{_esc(filename)}" alt="{_esc(caption)}">'
                f"<figcaption>{caption}</figcaption></figure>"
            )
    return "\n".join(parts), charts, len(notebook.get("cells", []))


def _circuits_page(all_models: dict, all_circuits: dict, all_figures: dict) -> tuple[str, int]:
    """Every circuit in the catalogue: the drawing, the terminal diagram and the statistics."""
    total = sum(len(all_circuits[slug]) for slug in all_circuits)
    parts = [
        '<h1 id="top">Circuits</h1>',
        f'<p class="lead">{total} circuit designs, drawn in black and white by Qiskit and written out '
        "again as text. Each one is the design the model compiles for that situation, so the picture is "
        "the simulation rather than an illustration of it.</p>",
        '<p class="lede-meta">Use <code>qcog circuit &lt;model&gt; --style iqp --format svg</code> for a '
        "version with the Qiskit styles, or <code>qcog run &lt;model&gt; --circuit-style bw</code> for this "
        "one.</p>",
        "<h2>Every design at a glance</h2>",
    ]
    import pandas as pd

    rows = []
    for slug, designs in all_circuits.items():
        for design in designs:
            rows.append(
                {
                    "design": f'<a href="#{_anchor(slug, design["name"])}">{_esc(design["name"])}</a>',
                    "model": f'<a href="models/{slug}.html">{_esc(all_models[slug].title)}</a>',
                    "qubits": design["qubits"],
                    "classical bits": design["clbits"],
                    "depth": design["depth"],
                    "gates": design["size"],
                }
            )
    parts.append(
        _html_table(
            rows,
            f"Every circuit design in the catalogue - click a design to jump to it",
        )
    )
    for letter, grouped in models_by_module().items():
        slugs_here = [model.slug for model in grouped if model.slug in all_circuits]
        if not slugs_here:
            continue
        parts.append(f"<h2>Module {letter}: {_esc(grouped[0].module_title())}</h2>")
        for slug in slugs_here:
            links = " &middot; ".join(
                f'<a href="#{_anchor(slug, design["name"])}">{_esc(design["name"])}</a>'
                for design in all_circuits[slug]
            )
            parts.append(
                f'<h3><a href="models/{slug}.html">{_esc(all_models[slug].title)}</a></h3>'
                f'<p class="lede-meta">{links}</p>'
            )
            for design in all_circuits[slug]:
                information = all_figures[slug].get(f"circuit: {design['name']}", {})
                path = (information.get("path") or "").replace("../", "", 1)
                parts.append(
                    f'<div id="{_anchor(slug, design["name"])}">'
                    + _circuit_block(design, path or None, information.get("description", ""))
                    + '<p class="lede-meta"><a href="#top">&#8593; back to the list of designs</a></p>'
                    + "</div>"
                )
    return "\n".join(parts), total


def _study_page(repository: Path, figures_dir: Path, *, include_figures: bool) -> tuple[str, int]:
    """The five comparative cases: what is compared, by how much the quantum account wins, and its limits."""
    import json

    payload = json.loads((repository / "studies" / "quantum_advantage" / "results.json").read_text(encoding="utf-8"))
    cases = payload["cases"]
    parts = [
        "<h1>Where the quantum account does better</h1>",
        f'<p class="lead">Five measured cases. Each one states the sense in which the quantum account is '
        "better - <strong>S1</strong> a classical model cannot represent the observation at all, "
        "<strong>S2</strong> the quantum model predicts something with no free parameters, "
        "<strong>S3</strong> it accounts for the same data with fewer assumptions - and what the case does "
        "<em>not</em> show.</p>",
    ]
    import pandas as pd

    rows = []
    for case in cases:
        first = case["metrics"][0]
        rows.append(
            {
                "case": case["title"],
                "sense": case["sense"].split(" (")[0],
                "compared on": first["metric"],
                "quantum": f'{first["quantum"]:.3f}',
                "classical": f'{first["classical"]:.3f}',
                "favours": first.get("favours", ""),
            }
        )
    parts.append(_table(pd.DataFrame(rows), "The five cases, at a glance"))
    for index, case in enumerate(cases, start=1):
        parts.append(f'<h2>{index}. {_esc(case["title"])}</h2>')
        parts.append(f'<p class="lede-meta">{_esc(case["sense"])}</p>')
        parts.append(f'<p>{_esc(case["phenomenon"])}</p>')
        parts.append(
            f'<p class="lede-meta"><strong>Data.</strong> {_esc(case["data"])}</p>'
            f'<p class="lede-meta"><strong>Models.</strong> {_esc(case["models"])}</p>'
        )
        frame = pd.DataFrame(
            [
                {
                    "quantity": metric["metric"],
                    "quantum": f'{metric["quantum"]:.4f}',
                    "classical": f'{metric["classical"]:.4f}',
                    "favours": metric.get("favours", ""),
                }
                for metric in case["metrics"]
            ]
        )
        parts.append(_table(frame, "What is measured, and what each account predicts"))
        figure = case.get("figure", "")
        if figure:
            filename = f"study__{Path(figure).name}"
            if include_figures:
                source = repository / "studies" / "quantum_advantage" / figure
                if source.exists():
                    target = figures_dir / filename
                    shutil.copyfile(source, target)
                    _greyscale(target)
            parts.append(
                f'<figure><img src="assets/figures/{_esc(filename)}" '
                f'alt="{_esc(case.get("figure_title", case["title"]))}">'
                f'<figcaption>{_esc(case.get("figure_title", case["title"]))}</figcaption></figure>'
            )
        if case.get("evidence"):
            parts.append("<h3>The evidence in the data</h3><ul>")
            parts += [f"<li>{_esc(item)}</li>" for item in case["evidence"]]
            parts.append("</ul>")
        if case.get("not_shown"):
            parts.append(
                '<div class="caveat"><p><strong>What this case does not show.</strong> '
                + _esc(case["not_shown"])
                + "</p></div>"
            )
    parts.append(
        '<p class="lede-meta">The full study, with its method and its two negative controls, is in '
        f'<a href="{DEFAULT_REPO_URL}/blob/main/studies/quantum_advantage/REPORT.md">studies/quantum_advantage/REPORT.md</a>, '
        "and every number above comes from "
        f'<a href="{DEFAULT_REPO_URL}/blob/main/studies/quantum_advantage/results.json">results.json</a>.</p>'
    )
    return "\n".join(parts), len(cases)


def _real_world_page(all_models: dict, all_results: dict, all_figures: dict, all_circuits: dict, prefix: str = "") -> str:
    """Short, concrete questions, with the command, the numbers and the circuit for each."""
    parts = [
        "<h1>Real questions, run with Q-Cog</h1>",
        '<p class="lead">Six questions a person might actually ask. Each one shows the command, what the '
        "model reports, what a classical account of the same situation predicts, and the circuit that "
        "computes it. The numbers come from running the model while this page was built.</p>",
    ]
    for slug, question, prose, metric in REAL_WORLD:
        if slug not in all_results:
            continue
        model = all_models[slug]
        result = all_results[slug]
        parts.append(f"<h2>{_esc(question)}</h2>")
        parts.append(f"<p>{prose}</p>")
        parts.append(
            '<pre class="code" data-language="bash"><code>'
            + f"qcog run {slug}\nqcog circuit {slug} --style bw --format svg"
            + "</code></pre>"
        )
        parts.append(f'<p class="lead">{_esc(result.summary)}</p>')
        rows = [row for row in result.classical_comparison if row.get("metric")]
        if rows:
            import pandas as pd

            frame = pd.DataFrame(
                [
                    {
                        "quantity": row["metric"],
                        "quantum model": f'{float(row["model"]):.4f}',
                        "classical account": f'{float(row["baseline"]):.4f}',
                        "why the classical value is what it is": row.get("note", ""),
                    }
                    for row in rows
                ]
            )
            parts.append(_table(frame, f"The comparison, including {metric!r}"))
        circuit_name = f"circuit: {all_circuits[slug][0]['name']}" if all_circuits.get(slug) else None
        information = all_figures[slug].get(circuit_name) if circuit_name else None
        if information:
            parts.append(_figure_block(information["path"], all_circuits[slug][0]["name"], information["description"]))
        if all_circuits.get(slug):
            design = all_circuits[slug][0]
            if design["text"]:
                parts.append('<pre class="terminal"><code>' + _esc(design["text"]) + "</code></pre>")
        headline = next(
            (
                name
                for name in all_figures[slug]
                if not name.startswith("circuit:") and name != "model versus classical baseline"
            ),
            None,
        )
        if headline:
            information = all_figures[slug][headline]
            parts.append(_figure_block(information["path"], headline, information["description"]))
        parts.append(
            f'<p class="lede-meta">Full page: <a href="{prefix}models/{slug}.html">{_esc(model.title)}</a> '
            "&middot; parameters, findings, caveats, every figure.</p>"
        )
    return "\n".join(parts)


def build_site(*args, **kwargs) -> dict:
    """Generate the whole site in black and white and return a summary of what was written."""
    with monochrome():
        return _build_site(*args, **kwargs)


def _build_site(
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
    all_models: dict = {}
    all_results: dict = {}
    all_figures: dict = {}
    all_circuits: dict = {}
    for slug in slugs:
        model_class = MODELS[slug]
        model = model_class()
        # the circuit drawings use Qiskit's black-and-white style, so the whole site is monochrome
        result = model.run(circuit_style="bw")
        figures: dict = {}
        if include_figures:
            figures_dir.mkdir(parents=True, exist_ok=True)
            for name, figure in model.figures(result).items():
                filename = f"{slug}__{_safe(name)}.png"
                figure.savefig(figures_dir / filename, bbox_inches="tight")
                _greyscale(figures_dir / filename)
                figures[name] = {
                    "path": f"../assets/figures/{filename}",
                    "description": getattr(figure, "_qcog_description", ""),
                }
                summary["figures"] += 1
                plt.close(figure)
        all_models[slug] = model
        all_results[slug] = result
        all_figures[slug] = figures
        all_circuits[slug] = _circuit_material(model, result)
        body = _model_page(model_class, result, figures, "../models/index.html", all_circuits[slug])
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

    # ------------------------------------------------------------------ the circuit gallery
    circuits_body, circuit_count = _circuits_page(all_models, all_circuits, all_figures)
    write(
        "circuits.html",
        _page(
            "Circuits - Q-Cog",
            circuits_body,
            active="circuits.html",
            depth=0,
            description=f"{circuit_count} circuit designs from the catalogue, drawn in black and white and "
            "written out as text.",
            repo_url=repo_url,
        ),
    )
    summary["sections"]["circuits"] = circuit_count

    # ------------------------------------------------------------------ the five comparative cases
    if include_study:
        study_results = repository / "studies" / "quantum_advantage" / "results.json"
        if study_results.exists():
            figures_dir.mkdir(parents=True, exist_ok=True)
            body, cases = _study_page(repository, figures_dir, include_figures=include_figures)
            write(
                "study.html",
                _page(
                    "Where the quantum account does better - Q-Cog",
                    body,
                    active="study.html",
                    depth=0,
                    description="Five measured cases where the quantum account beats the classical one, each "
                    "with its graph, its numbers, its evidence and its limits.",
                    repo_url=repo_url,
                ),
            )
            summary["sections"]["study"] = cases

    # ------------------------------------------------------------------ the notebook
    if include_notebook:
        notebook = repository / "demos" / "quickstart.ipynb"
        if notebook.exists():
            body, charts, cells = _notebook_page(notebook, figures_dir, repository)
            write(
                "notebook.html",
                _page(
                    "Q-Cog quickstart: a two-question survey - Q-Cog",
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
                   '<p class="lead">Six runnable scripts, one task each, from a five-minute first look to the '
                   "full catalogue. Every page gives the scenario, what to look for in the output, what the "
                   "numbers mean, the captured output itself and the whole script. The output shown is what "
                   'the script prints when it runs from the repository root.</p><ul class="cards">']
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
            one_liner = (EXAMPLE_GUIDES.get(script.stem) or {}).get("one_liner", "")
            landing.append(
                f'<li><h3><a href="{page}">{_esc(title)}</a></h3>'
                f'<p class="lede-meta">{_esc(script.name)}</p>'
                + (f"<p>{_esc(one_liner)}</p>" if one_liner else "")
                + "</li>"
            )
        landing.append("</ul>")
        landing.append(
            '<h2>Real questions</h2><p class="lede-meta">Six questions a person might actually ask, with '
            'the command, the numbers, the classical comparison and the circuit for each: '
            '<a href="real-world.html">real questions, run with Q-Cog</a>.</p>'
        )
        landing.append(
            '<h2>The showcase</h2><p class="lede-meta">A scripted tour of four models, one per module, '
            'with its captured output: the cognitive question, the mechanism, the classical baseline, '
            'every parameter with its bounds, the findings, the model-versus-classical table and the '
            'caveats for each. <a href="showcase.html">Read the captured run</a>, or run '
            '<code>python demos/showcase/run_showcase.py</code> yourself.</p>'
        )
        write(
            "examples/real-world.html",
            _page(
                "Real questions, run with Q-Cog",
                _real_world_page(all_models, all_results, all_figures, all_circuits, prefix="../"),
                active="examples/index.html",
                depth=1,
                description="Six real questions, with the command, the numbers, the classical comparison and "
                "the circuit for each.",
                repo_url=repo_url,
            ),
        )
        showcase_output = repository / "demos" / "showcase" / "output.txt"
        showcase_readme = repository / "demos" / "showcase" / "README.md"
        showcase_body = [
            "<h1>The showcase: four models, one run, captured</h1>",
            '<p class="lead">A scripted tour of four models - one per module - chosen because between them '
            "they exercise every feature of the library. For each model it prints the cognitive question, the "
            "quantum mechanism, the classical baseline, every parameter with its bounds, the findings, the "
            "model-versus-classical table, whether a circuit was drawn and what it contains, and the caveats. "
            "Then it saves every figure and an accessible HTML report per model.</p>",
            '<p class="lede-meta">Run it with <code>python demos/showcase/run_showcase.py</code> '
            "(or <code>--all</code> for the whole catalogue, <code>--reports</code> for the HTML reports). "
            "The text below is exactly what a plain run prints.</p>",
        ]
        if showcase_readme.exists():
            # the README's own title becomes a section: the page already has its h1
            body, _headings = markdown.to_html(showcase_readme.read_text(encoding="utf-8"), heading_shift=1)
            showcase_body.append(body)
        if showcase_output.exists():
            showcase_body.append(
                '<h2>The captured output</h2><pre class="terminal"><code>'
                + _esc(showcase_output.read_text(encoding="utf-8"))
                + "</code></pre>"
            )
        write(
            "examples/showcase.html",
            _page(
                "The showcase: four models, one run, captured - Q-Cog",
                "\n".join(showcase_body),
                active="examples/index.html",
                depth=1,
                description="The showcase's captured run: four models, their findings, their classical "
                "comparisons, their circuits and their caveats.",
                repo_url=repo_url,
            ),
        )
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
        documentation = ["README.md", "CONTRIBUTING.md"]
        documentation += [f"docs/{path.name}" for path in sorted((repository / "docs").glob("*.md"))]
        documentation += [
            "model_tests/README.md",
            "demos/README.md",
            "studies/quantum_advantage/README.md",
        ]
        index = ["<h1>Documentation</h1>",
                 '<p class="lead">The written documentation, rendered from the repository. The generated '
                 'reference for every model and parameter is in <a href="../models/index.html">Models</a>.</p><ul>']
        rendered = frozenset(relative for relative in documentation if (repository / relative).exists())
        for relative in documentation:
            if not (repository / relative).exists():
                continue
            title, body = _render_markdown_page(
                relative, repository, repo_url=repo_url, depth=1, documentation=rendered
            )
            name = _docs_page_name(relative)
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
        '<p class="lead">Q-Cog models human judgement with quantum probability, on Qiskit, and always '
        'beside the classical account of the same situation. Twenty-three models of judgement, decision '
        "making and collective behaviour: each one simulated, drawn as a circuit, compared with what a "
        "classical probability space predicts, and honest about what it does not claim.</p>"
        '<p>It exists to answer one question - <em>does the quantum account earn its place, and '
        "where?</em> - and to make the answer checkable rather than rhetorical.</p>",
        f'<p><span class="badge">{len(MODELS)} models</span><span class="badge">{len(MODULES)} modules</span>'
        f'<span class="badge">{summary["figures"]} figures</span>'
        f'<span class="badge">{summary["sections"].get("circuits", 0)} circuits</span>'
        '<span class="badge">no JavaScript</span></p>',
        "<h2>Start here</h2>",
        '<ul class="cards">'
        '<li><h3><a href="models/index.html">The models</a></h3><p>Twenty-three models in four modules: '
        "parameters, findings, the classical comparison, every figure and every circuit.</p></li>"
        '<li><h3><a href="circuits.html">The circuits</a></h3><p>Every circuit design in the catalogue, drawn '
        "in black and white and written out again as text.</p></li>"
        '<li><h3><a href="examples/real-world.html">Real questions</a></h3><p>Six questions a person might '
        "actually ask, each with the command, the numbers and the circuit.</p></li>"
        '<li><h3><a href="study.html">The five cases</a></h3><p>Where the quantum account measurably beats the '
        "classical one - and the two controls where a classical model survives.</p></li>"
        '<li><h3><a href="examples/index.html">The examples</a></h3><p>Six runnable scripts, each with its '
        "scenario, its output and what the numbers mean.</p></li>"
        '<li><h3><a href="notebook.html">Quickstart</a></h3><p>One survey asked in both orders, end to end: '
        "the order effect, the classical identity that fails, the circuits, a simulator run - with the "
        "charts it produced.</p></li>"
        "</ul>",
        "<h2>Why: five measured cases</h2>" if study_cases else "",
        "<p>Quantum cognition is worth the trouble only if it does something the classical account "
        "cannot. So the library measures that first, on five phenomena, and publishes the result either "
        "way. Each case shows one of three things: the classical model <strong>cannot represent the "
        "observation at all</strong> (S1), the quantum model <strong>predicts a number with no free "
        "parameters</strong> (S2), or it explains the same data with <strong>fewer assumptions</strong> "
        "(S3). Two of the five include a classical model that is not rejected, and every case states "
        "what it does <em>not</em> show.</p>",
        study_cases,
        '<p class="lede-meta"><a href="study.html">The five cases in full</a> &middot; what is compared, '
        "by how much, and the limits of each claim.</p>"
        if study_cases
        else "",
        "<h2>What every model does</h2>",
        "<p>Each one is simulated with <strong>Qiskit</strong>, reports the same quantity computed by a "
        "<strong>classical</strong> account of the same situation, states the <strong>caveats</strong> that "
        "limit its claims, and draws its <strong>circuit</strong>. The library checks itself: the evidence "
        "table, the figure checks, the documentation checks and a model-by-model test suite all run in the "
        "repository.</p>",
        "<h2>Install and run</h2>",
        '<pre class="code" data-language="bash"><code>pip install qcog                 # from PyPI, once released\n'
        "git clone &lt;this repository&gt;     # or from a checkout\n"
        "python model_tests/run_all.py    # check every model in about two minutes\n"
        "qcog list                        # the catalogue\n"
        "qcog run question-order --backend sampled --shots 8000</code></pre>",
        '<p class="smallprint">Generated from the library itself: the same metadata, analyses, figures and '
        "circuits the command line and the HTML reports use. Black and white by design - the only colour "
        "anywhere is red, and it is reserved for errors.</p>",
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