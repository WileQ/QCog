"""A small Markdown subset for the generated website.

The project's documentation is written in Markdown and the site shows it as HTML, which needs
a converter. Rather than take a dependency for it, this module implements the subset the
documentation actually uses - headings, paragraphs, fenced code, lists, blockquotes, pipe
tables, and inline code, bold, italic, links and images - and nothing else. Anything it does
not recognise is passed through as plain text rather than being silently mangled.

The converter is defensive on purpose: a documentation file that grows a construct it does not
know must still render, so a doc build can never fail on prose.
"""

from __future__ import annotations

import html
import re
from typing import Callable

__all__ = ["to_html", "split_front_matter"]

#: A resolver maps a link or image target to the URL used on the site, or returns ``None``
#: to keep it as written.
Resolver = Callable[[str], "str | None"]


def split_front_matter(text: str) -> tuple[dict, str]:
    """Split an optional ``---`` front-matter block from the body."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    block = text[4:end]
    body = text[end + 4 :].lstrip("\n")
    fields = {}
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields, body


def _inline(text: str, resolve: Resolver | None) -> str:
    """Inline markup: escapes first, then code, images, links, bold and italic."""
    out = html.escape(text, quote=False)
    # inline code first, so its contents are not further processed
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
    if resolve is not None:
        def image(match: re.Match) -> str:
            alt, target = match.group(1), match.group(2)
            url = resolve(target) or target
            return f'<img src="{html.escape(url, quote=True)}" alt="{html.escape(alt, quote=True)}">'

        out = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", image, out)

        def link(match: re.Match) -> str:
            label, target = match.group(1), match.group(2)
            url = resolve(target)
            if url is None:
                return label
            return f'<a href="{html.escape(url, quote=True)}">{label}</a>'

        out = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", out)
    return out


def _table(rows: list[str], resolve: Resolver | None) -> str:
    """A pipe table: the first row is the header, the second the alignment rule."""
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], cells[2:]
    out = ["<div class=\"table-wrap\"><table>"]
    out.append(
        "<thead><tr>"
        + "".join(f'<th scope="col">{_inline(cell, resolve)}</th>' for cell in header)
        + "</tr></thead>"
    )
    out.append("<tbody>")
    for row in body:
        out.append(
            "<tr>" + "".join(f"<td>{_inline(cell, resolve)}</td>" for cell in row) + "</tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


def to_html(text: str, resolve: Resolver | None = None, heading_shift: int = 0) -> tuple[str, list[tuple[int, str]]]:
    """Convert a Markdown subset to HTML.

    Returns the HTML and the list of ``(level, slug)`` headings it produced, so a caller can
    build a table of contents and add anchors.
    """
    _, body = split_front_matter(text)
    lines = body.splitlines()
    out: list[str] = []
    headings: list[tuple[int, str]] = []
    paragraph: list[str] = []
    index = 0

    def flush() -> None:
        if paragraph:
            out.append(f"<p>{_inline(' '.join(paragraph), resolve)}</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            language = stripped[3:].strip() or "text"
            body_text = html.escape("\n".join(block))
            out.append(
                f'<pre class="code" data-language="{html.escape(language)}"><code>{body_text}</code></pre>'
            )
            index += 1
            continue

        if re.match(r"^#{1,6}\s", stripped):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            shown = min(6, level + heading_shift)
            headings.append((shown, slug))
            out.append(f'<h{shown} id="{slug}">{_inline(title, resolve)}</h{shown}>')
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and set(lines[index + 1].strip()) <= set("|-: "):
            flush()
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            out.append(_table(block, resolve))
            continue

        if stripped.startswith(("- ", "* ")) or re.match(r"^\d+\.\s", stripped):
            flush()
            ordered = bool(re.match(r"^\d+\.\s", stripped))
            items: list[str] = []
            while index < len(lines) and (
                lines[index].strip().startswith(("- ", "* ")) or re.match(r"^\d+\.\s", lines[index].strip())
                or (items and lines[index].startswith("  ") and lines[index].strip())
            ):
                text_line = lines[index].strip()
                text_line = re.sub(r"^(- |\* |\d+\.\s)", "", text_line)
                if lines[index].startswith("  ") and items:
                    items[-1] = items[-1] + " " + text_line
                else:
                    items.append(text_line)
                index += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(item, resolve)}</li>" for item in items) + f"</{tag}>")
            continue

        if stripped.startswith(">"):
            flush()
            quoted = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quoted.append(lines[index].strip().lstrip(">").strip())
                index += 1
            out.append(f"<blockquote>{_inline(' '.join(quoted), resolve)}</blockquote>")
            continue

        if stripped in {"", "---", "***"}:
            flush()
            if stripped:
                out.append("<hr>")
            index += 1
            continue

        paragraph.append(stripped)
        index += 1

    flush()
    return "\n".join(out), headings