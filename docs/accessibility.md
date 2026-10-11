# Accessibility

The library follows industry best practices on UI accessibility. This document states
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

* The colour-blind-safe Okabe-Ito palette (`qcog.viz.PALETTE`) is used for every figure and the report. Eight of
  its colours form the line/series cycle (`qcog.viz.CATEGORICAL`); the pale yellow is kept for areas and
  annotations because a pale line on a white page is not readable.
* **Series differ by more than hue.** `qcog.viz.series_style(index)` returns a colour *and* a linestyle *and* a
  marker, cycling colour first, then linestyle, then marker - so up to 8 x 4 x 8 series stay distinguishable, and
  two series never differ only by colour. Bar charts use hatches for the same reason, and the classical-comparison
  chart draws the baseline as a hatched outline and the model as a solid bar, so the pair reads in greyscale.
* No series is identified by colour alone: bars are labelled with their values, differences are printed as signed
  numbers, and legends are always paired with axis labels or direct text.
* A high-contrast mode is one checkbox away in the explorer (black background, white text, underlined links).
* Matplotlib is put in the headless `Agg` backend only when the library is *not* running inside a notebook or
  another interactive session, so charts render inline for a Jupyter user (see `demos/quickstart.ipynb`) and
  still work in scripts, servers and report generation.

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
* Every page of the website (`python scripts/build_website.py`) is static semantic HTML: a skip link, headings
  in order, a navigation landmark with `aria-current` on the active page, alternative text on every figure, and
  visible focus outlines. Each parameter's `description` from its `ParamSpec` is the text used in the documentation, the
  tables and the generated pages, so the same explanation reaches every reader.
* Reading options: the site follows the system colour-scheme preference and stays readable with images
  disabled, because every chart appears with its description and with the data table it was drawn from.

## Language

* Every model's finding sentences are generated from computed values by `qcog.explain` and the model classes.
* Parameters carry human-readable labels and descriptions (`ParamSpec.label` / `.description`), used as the
  widget labels, CLI help and documentation text - one declaration, three surfaces.
* Caveats are written as complete sentences about what the model does *not* claim.

## Charts in a notebook

`demos/quickstart.ipynb` is the accessible-by-example tour: it prints the classical comparison as a table
*and* shows the chart with its description underneath, draws the circuit with an adjustable style, and saves a
report. The notebook's cells are executed by the test suite from top to bottom, so it cannot rot.

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

Two further checks keep the charts honest. `qcog.viz.structure_checks()` reads each drawing back - bar heights,
stacked tops, line data, heatmap cells, the Bloch vector, the circuit's own statistics - and compares it with the
numbers that produced it, so a chart that renders but shows the wrong data fails; `python
scripts/check_figures.py` runs that audit together with the render checks. And a figure that would mislabel
itself is drawn differently rather than misleadingly: when two questions have different outcome labels, the
marginal chart uses one panel per question instead of grouping bars under shared labels.

Automated checks are also part of the test suite (`tests/test_cli_and_report.py::test_model_report_is_accessible`),
so a regression that removes the skip link, the alt text or the captions fails the build. Two further checks cover
the charts themselves: `python scripts/check_figures.py` renders every figure function and every model's figures
and fails if one is broken or lacks its description, and `tests/test_circuits_and_comparison.py` asserts that
series styles stay distinguishable and that the comparison chart separates its two series by a hatch.

## Known limits

* The figures are static images. The **data tables** are the accessible equivalent, and the text descriptions
  give the values that matter; a colour-vision or screen-reader user should use those, not the PNGs.
* The website is generated, so its content is identical to what the CLI and the reports produce; the CLI
  remains the surface with no HTML at all.
* Mathematical notation is written in ASCII in the interface and in LaTeX-free prose in the reports, so it is
  read correctly by speech synthesisers.