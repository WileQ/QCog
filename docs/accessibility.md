# Accessibility

The submission requirement is to "follow industry best practices on UI accessibility". This document states
what Q-Cog does, and - more usefully - how anyone can verify it.

## Principles applied

1. **Perceivable.** Nothing essential is conveyed by colour, position or animation alone. Every figure is
   accompanied by (a) a caption, (b) a long description in plain language, and (c) the data table it was drawn
   from, so every number in every chart is reachable as text.
2. **Operable.** The command line needs no pointer. The explorer uses standard, labelled native controls
   (sliders, select boxes, checkboxes, buttons) that are keyboard operable and announced by screen readers. The
   HTML report starts with a skip link, has a logical tab order and shows a visible focus outline on every
   interactive element.
3. **Understandable.** Explanations are short, rule-based sentences generated from the computed numbers, so
   they cannot drift from the data. Each model states its cognitive question, its mechanism, its classical
   baseline and its caveats in the same order, everywhere.
4. **Robust.** The report is self-contained HTML: no JavaScript, no external assets, no network. It renders in
   any browser and remains readable with images disabled.

## Colour

* The Okabe-Ito colour-blind-safe palette (`qcog.viz.PALETTE`) is used for every figure and the report.
* No series is identified by colour alone: bars are labelled, differences are printed as signed numbers, and
  legends are always paired with axis labels or direct text.
* A high-contrast mode is one checkbox away in the explorer (black background, white text, underlined links).

## Text alternatives

The mechanism is simple and enforced by construction: figures are created through `qcog.viz`, which attaches a
plain-language description to the figure object itself (`figure._qcog_description`) plus optional extra
sentences. `qcog.viz.text_alternatives(figures)` returns them; the report embeds the description as the image's
`alt` attribute and inside a `<figcaption>`, and the long form inside a `<details>` element. A chart therefore
*cannot* be exported without an accessible alternative, because the description travels with the object.

An example of what a description looks like:

> Bar chart decomposing p(B) into the classical (law of total probability) contribution and the interference
> contribution. yes: classical 0.764 plus interference -0.236 equals 0.528; no: classical 0.236 plus
> interference 0.236 equals 0.472.

## Controls, keyboard and screen readers

* The HTML report contains `<nav aria-label="Contents">`, one `<h1>`, ordered headings, `<section>`s with
  `aria-labelledby`, a skip link that becomes visible on focus, table `<caption>`s and `scope="col"` headers,
  and `:focus-visible` outlines (3 px, high contrast).
* The explorer relies on Streamlit's native widgets, which expose labels and help text; each parameter's
  `description` from the model's `ParamSpec` is passed to the widget as help text, so the same explanation
  appears to a screen reader as to a sighted user.
* Reading options in the explorer: dark background, larger type (118% root font size), high contrast, and
  **text-only mode**, which hides all images and shows the data tables instead.

## Language

* Every model's finding sentences are generated from computed values by `qcog.explain` and the model classes.
* Parameters carry human-readable labels and descriptions (`ParamSpec.label` / `.description`), used as the
  widget labels, CLI help and documentation text - one declaration, three surfaces.
* Caveats are written as complete sentences about what the model does *not* claim.

## Verifying it

```bash
qcog report --out out/qcog_report.html
```

Then inspect the generated file (it is plain text) and check:

```bash
python - <<'PY'
from pathlib import Path
html = Path("out/qcog_report.html").read_text(encoding="utf-8")
for needle in ('lang="en"', 'class="skip"', 'aria-labelledby', '<figcaption>', 'alt="',
               '<caption>', 'scope="col"', 'focus-visible', 'prefers-color-scheme'):
    print(f"{needle:22s}", needle in html)
PY
```

Automated checks are also part of the test suite (`tests/test_cli_and_report.py::test_model_report_is_accessible`),
so a regression that removes the skip link, the alt text or the captions fails the build.

## Known limits

* The figures are static images. The **data tables** are the accessible equivalent, and the text descriptions
  give the values that matter; a colour-vision or screen-reader user should use those, not the PNGs.
* Streamlit itself is not fully screen-reader-annotated in every widget; the explorer is the convenience layer,
  and the CLI plus the HTML report are the fully accessible surfaces.
* Mathematical notation is written in ASCII in the interface and in LaTeX-free prose in the reports, so it is
  read correctly by speech synthesisers.