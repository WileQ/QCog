"""The website is generated, so it can be checked mechanically.

Three things are asserted: the Markdown subset the site is built on behaves (and never crashes on the
project's own documentation), a small site builds correctly into a temporary directory with all its
accessibility properties, and a site that has been built here passes `scripts/check_website.py` (the
website workflow runs that same check on the site it deploys).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_subset_renders_the_constructs_the_docs_use():
    from qcog.website import to_html

    text = """# Title

A paragraph with **bold**, *italic*, `code` and a [link](docs/x.md).

![A chart](figures/chart.png)

- first
- second

1. one
2. two

> a quotation

| a | b |
| --- | --- |
| 1 | 2 |

```python
print("hello")
```
"""
    html, headings = to_html(text, resolve=lambda target: f"mapped/{target}")
    assert headings[:1] == [(1, "title")]
    assert "<strong>bold</strong>" in html and "<em>italic</em>" in html
    assert "<code>code</code>" in html
    assert 'href="mapped/docs/x.md"' in html
    assert '<img src="mapped/figures/chart.png" alt="A chart">' in html
    assert html.count("<ul>") == 1 and html.count("<ol>") == 1
    assert html.count("<li>") == 4  # two unordered items and two ordered ones
    assert "<blockquote>a quotation</blockquote>" in html
    assert html.count('<th scope="col">') == 2          # two columns
    assert html.count("<tr>") == 2                      # one header row and one data row
    assert 'data-language="python"' in html and "print(&quot;hello&quot;)" in html


def test_markdown_never_crashes_on_the_project_documentation():
    from qcog.website import to_html

    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += [ROOT / "studies" / "quantum_advantage" / "REPORT.md"]
    for path in files:
        if not path.exists():
            continue
        html, headings = to_html(path.read_text(encoding="utf-8"))
        assert html.strip(), path.name
        assert headings, path.name


def test_a_small_site_builds_with_every_property(tmp_path):
    """Build two model pages and check the whole structure, without touching the committed site."""
    from qcog.registry import MODELS
    from qcog.website import build_site
    from scripts.check_website import check_site  # noqa: F401  (imported for its side-effect-free API)

    summary = build_site(tmp_path, root=ROOT, models=["question-order", "auctions"],
                         include_examples=False, include_notebook=False, include_study=False,
                         include_docs=False)
    assert summary["pages"] >= 4  # home, the model index and the two model pages
    assert (tmp_path / "models" / "question-order.html").exists()
    html = (tmp_path / "models" / "question-order.html").read_text(encoding="utf-8")
    assert 'lang="en"' in html and 'class="skip"' in html
    assert html.count("<h1") == 1
    assert html.count("<img") == html.count("alt=")
    assert "Model versus classical baseline" in html
    assert "circuit" in html.lower()


def test_the_notebook_page_names_the_notebook_relatively(tmp_path):
    """The published site must never print the build machine's absolute paths."""
    from qcog.website import build_site

    build_site(
        tmp_path,
        root=ROOT,
        models=[],
        include_examples=False,
        include_notebook=True,
        include_study=False,
        include_docs=False,
    )
    page = tmp_path / "notebook.html"
    assert page.exists()
    html = page.read_text(encoding="utf-8")
    assert "demos/quickstart.ipynb" in html
    assert str(ROOT) not in html
    assert not re.search(r"(?<![A-Za-z0-9._-])[A-Za-z]:[\\/]", html)
    assert not re.search(r"/Users/|/home/[a-z]", html)


def test_the_generated_site_passes_its_checks():
    if not (ROOT / "website" / "index.html").exists():
        pytest.skip("website/ has not been built here; run python scripts/build_website.py")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_website.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "website checks passed" in completed.stdout
