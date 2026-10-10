# Demos: the notebook, the examples and the showcase

Everything a reader needs to *use* Q-Cog lives here, in one place, outside the package:

| Path | What it is | Start with it if… |
| --- | --- | --- |
| [`quickstart.ipynb`](quickstart.ipynb) | a short notebook built around one real survey: *"Are you satisfied with the economy?"* and *"Will you vote for the current administration?"*, asked in both orders | you want the whole workflow with the figures inline |
| [`examples/`](examples/) | six runnable scripts, from a five-minute first look to the full catalogue | you prefer scripts, or want one task per file |
| [`showcase/`](showcase/) | a scripted tour of four models with its **captured output**, every figure and an HTML report per model | you want the results without running anything |

## Run them

```bash
python demos/examples/01_first_steps.py             # the shortest complete example
python demos/examples/06_economy_and_vote.py        # the survey, with the circuits written out
python demos/showcase/run_showcase.py               # four models, figures and reports
python demos/showcase/run_showcase.py --all         # every model in the catalogue
```

Open the notebook with Jupyter (`jupyter lab demos/quickstart.ipynb`). None of these needs an
installation: each puts `src/` on the path itself, so a plain checkout works. If you installed the
package with `pip`, they run unchanged as well.

## What each example shows

| Script | Shows |
| --- | --- |
| `01_first_steps.py` | a belief state, two questions, both orders, the order effect and the QQ equality |
| `02_question_order.py` | the flagship model: exact, on a compiled circuit, with decoherence, plus a report |
| `03_hardware_and_ionq.py` | the two circuit compilations, and the optional IonQ submission path |
| `04_fit_to_data.py` | fitting a model to observed frequencies and comparing it with a classical baseline |
| `05_catalogue_report.py` | every model run, the self-check table, the full accessible HTML report |
| `06_economy_and_vote.py` | the real survey: both orders, the classical identity that fails, the circuits |

## Where the same material appears elsewhere

* **the static website** ([`scripts/build_website.py`](../scripts/build_website.py)) - generated from the library: every model with
  its figures, its classical comparison and its circuit drawings, this notebook rendered, the
  examples with their captured output, the comparative study and the documentation. It is published
  on GitHub Pages by `.github/workflows/pages.yml`, and it replaces the Streamlit explorer the
  project used to ship - a server was one more thing to run, and the site reaches the same content
  with no dependencies at all.
* **[`../studies/quantum_advantage/`](../studies/quantum_advantage/)** - five measured cases where
  the quantum account beats the classical one.
* **[`../docs/`](../docs/)** - the written documentation, including
  [`packaging.md`](../docs/packaging.md) for publishing the package to PyPI.