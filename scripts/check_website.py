"""Check the generated website.

    python scripts/check_website.py                 # structure and accessibility of website/
    python scripts/check_website.py --dir out/site

The site is generated, so it can be checked mechanically: every page must carry the language, a
skip link, exactly one top-level heading, a navigation landmark, and alternative text on every
image; every figure it references must exist; every internal link must resolve to a file that was
actually written; and no `<figure>` may be a block of text instead of a chart.  Exits non-zero if
anything fails, so it can run in a pipeline as well as by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qcog.registry import MODELS  # noqa: E402


def check_site(directory: Path, models: list[str] | None = None) -> list[dict]:
    """Return one row per check."""
    rows: list[dict] = []
    pages = sorted(directory.rglob("*.html"))

    def record(name: str, ok: bool, detail: str = "") -> None:
        rows.append({"name": name, "ok": bool(ok), "detail": detail})

    record("the site was generated", bool(pages), f"{len(pages)} pages in {directory}")

    for slug in models or list(MODELS):
        record(f"model page for {slug}", (directory / "models" / f"{slug}.html").exists())

    required_files = ["index.html", "models/index.html", "examples/index.html", "docs/index.html",
                      "assets/style.css", ".nojekyll"]
    for relative in required_files:
        record(f"{relative} exists", (directory / relative).exists())

    missing_lang: list[str] = []
    missing_skip: list[str] = []
    multiple_h1: list[str] = []
    no_nav: list[str] = []
    images_without_alt: list[str] = []
    missing_images: list[str] = []
    empty_figures: list[str] = []
    broken_links: list[str] = []
    for page in pages:
        html = page.read_text(encoding="utf-8")
        relative = page.relative_to(directory).as_posix()
        if 'lang="en"' not in html:
            missing_lang.append(relative)
        if 'class="skip"' not in html:
            missing_skip.append(relative)
        if html.count("<h1") != 1:
            multiple_h1.append(relative)
        if 'aria-label="Site"' not in html:
            no_nav.append(relative)
        for match in re.finditer(r"<img\b[^>]*>", html):
            tag = match.group(0)
            if "alt=" not in tag:
                images_without_alt.append(f"{relative}: {tag[:60]}")
                continue
            source = re.search(r'src="([^"]+)"', tag)
            if source and not source.group(1).startswith(("http", "data:")):
                target = (page.parent / source.group(1)).resolve()
                if not target.exists():
                    missing_images.append(f"{relative} -> {source.group(1)}")
        for match in re.finditer(r"<figure>(.*?)</figure>", html, flags=re.S):
            if "<img" not in match.group(1):
                empty_figures.append(relative)
        for match in re.finditer(r'href="([^"]+)"', html):
            target = match.group(1)
            if target.startswith(("http", "#", "mailto:")):
                continue
            path = target.split("#")[0]
            if not path:
                continue
            if not (page.parent / path).resolve().exists():
                broken_links.append(f"{relative} -> {target}")

    record("every page declares its language", not missing_lang, ", ".join(missing_lang[:5]))
    record("every page has a skip link", not missing_skip, ", ".join(missing_skip[:5]))
    record("every page has exactly one h1", not multiple_h1, ", ".join(multiple_h1[:5]))
    record("every page has the navigation landmark", not no_nav, ", ".join(no_nav[:5]))
    record("every image has alternative text", not images_without_alt, "; ".join(images_without_alt[:4]))
    record("every referenced figure exists", not missing_images, "; ".join(missing_images[:4]))
    record("no figure is a block of text without a chart", not empty_figures, ", ".join(empty_figures[:4]))
    record("every internal link resolves", not broken_links, "; ".join(broken_links[:4]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(ROOT / "website"), help="the generated site directory")
    args = parser.parse_args(argv)
    rows = check_site(Path(args.dir))
    width = max(len(row["name"]) for row in rows) + 2
    failures = 0
    for row in rows:
        print(f"{'PASS' if row['ok'] else 'FAIL'}  {row['name']:<{width}} {row['detail']}")
        failures += 0 if row["ok"] else 1
    print()
    print(f"{len(rows) - failures}/{len(rows)} website checks passed" + (f"; {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())