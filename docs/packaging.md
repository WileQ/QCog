# Building and publishing the package

What is already in place, what only you can do, and how to check each step.  The short version:

```bash
python -m pip install --upgrade pip wheel build twine
# edit the version in pyproject.toml, set your repository URLs, commit, and tag it
python -m build                      # dist/qcog-<version>.tar.gz and a wheel
python -m twine check dist/*         # renders the README exactly as PyPI will
python -m twine upload dist/*        # or: push the tag and let the workflow publish
```

After that, anyone can `pip install qcog`.

## What the repository already provides

| Piece | Where |
| --- | --- |
| Build system and metadata | `pyproject.toml` (setuptools, `src/` layout, Python >= 3.10) |
| Runtime dependencies | `numpy`, `scipy`, `pandas`, `matplotlib`, `qiskit`, `qiskit-aer` |
| Optional extras | `[ionq]` for the IonQ path, `[dev]` for the tests, `[release]` for `build` and `twine` |
| Console command | `qcog = "qcog.cli:main"`, plus `python -m qcog` and `python -m qcog.website` |
| Package data | the illustrative datasets and the website stylesheet, declared under `[tool.setuptools.package-data]` |
| Licence | `LICENSE` (MIT), also declared in `pyproject.toml` |
| Long description | `README.md` (rendered on the PyPI page) |
| CI publishing | `.github/workflows/release.yml` (build, check, smoke-install, publish) |

## What you have to do

1. **Check the name.** `qcog` has to be free on PyPI:
   `python -m pip index versions qcog` (or search on pypi.org). If it is taken, change
   `name = "qcog"` in `pyproject.toml` to something you own - for example `qcog-cognition` - and keep the
   import name (`qcog`) as it is. The import name and the distribution name do not have to match.
2. **Point the URLs at your repository.** In `pyproject.toml`, `[project.urls]` has placeholder links; replace
   them with yours. The same URL appears in `src/qcog/website/build.py` (`DEFAULT_REPO_URL`, used for source
   links on the site) and in the README's clone line. One edit each, or pass `--repo-url` when building the site.
3. **Decide the version.** `version = "0.1.0"` in `pyproject.toml`. Bump it for every upload: PyPI refuses a
   version that already exists, and there is no way to replace one (you can only *yank* it, which hides it from
   new installs). Tag the release commit with the same number, `v0.1.0`.
4. **Install the tools.** `python -m pip install --upgrade pip wheel build twine`. `wheel` is needed to build a
   wheel; without it the build fails with `invalid command 'bdist_wheel'`.
5. **Build.** `python -m build` writes `dist/qcog-<version>.tar.gz` and `dist/qcog-<version>-py3-none-any.whl`.
6. **Check the metadata.** `python -m twine check dist/*` reports any problem PyPI would reject, and renders the
   long description as PyPI will.
7. **Install your own artefact and run it**, which is the only check that really counts:

   ```bash
   python -m venv /tmp/smoke && /tmp/smoke/bin/python -m pip install dist/*.whl
   /tmp/smoke/bin/python -c "import qcog; print(qcog.__version__)"
   /tmp/smoke/bin/qcog evidence
   ```
8. **Dry run on TestPyPI** (optional but cheap):

   ```bash
   python -m twine upload --repository testpypi dist/*
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple qcog
   ```
9. **Publish.** Either `python -m twine upload dist/*` with an API token, or - recommended - push a tag and let
   `.github/workflows/release.yml` publish with **trusted publishing**, so no token is stored anywhere. Set it up
   once at <https://pypi.org/manage/account/publishing/> with the owner, the repository name, the workflow
   `release.yml` and the environment `pypi`; TestPyPI needs its own entry. The workflow also verifies the
   metadata, installs the built wheel into a clean environment and imports it before publishing, so a broken
   wheel cannot reach PyPI.
10. **Keep releasing.** `pip install qcog` for users, `pip install -U qcog` to upgrade. For a documentation-only
    change, release a patch version; for a new model or a new option, a minor one.

## Two details worth knowing

* **Relative links in the README.** PyPI renders the README on its own, without your repository's file tree, so
  links such as `docs/packaging.md` do not resolve there (they work on GitHub). `twine check` does not complain
  about it. If that matters to you, use absolute `https://github.com/<you>/<repo>/blob/main/...` links in the
  README; the rest of the documentation can keep using relative links, since it is read on the site and on GitHub.
* **What ships in the wheel.** The package, its data files and the website stylesheet. The demos, the study, the
  tests and the generated site are *not* installed - they live in the repository - so `pip install qcog` gives a
  lean library.

## What was verified here

* `python -m pip install -e .` works from a checkout (perform it in a virtual environment).
* the source distribution builds and contains the package, the datasets, the website stylesheet and the licence:
  `python -c "import setuptools.build_meta as bm; print(bm.build_sdist('dist'))"` produced an 86-member archive.
* the metadata is valid: the build completed with no metadata error, and the console entry point and the extras
  appear in the generated `PKG-INFO`.
* building a *wheel* needs the `wheel` package, which is not installed in every environment (it was missing in
  the one used here); step 4 above covers it, and the release workflow installs it before building.

The remaining steps in the list above - the name check, the TestPyPI dry run and the final upload - need your
PyPI account, so they are the part only you can do.