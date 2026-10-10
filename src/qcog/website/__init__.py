"""Static website generator: the library's own tools, browsable without installing anything.

`python -m qcog.website` writes a site into `website/` containing a page per model (figures,
classical comparison, circuit drawings), the rendered notebook, the examples with their output,
the comparative study and the documentation.  It is plain HTML with one stylesheet and no
JavaScript, so GitHub Pages can serve it unchanged - see `.github/workflows/pages.yml` - and a
reader can also just open the files from disk.
"""

from .build import DEFAULT_REPO_URL, SITE_SECTIONS, build_site
from .markdown import to_html

__all__ = ["build_site", "SITE_SECTIONS", "DEFAULT_REPO_URL", "to_html"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m qcog.website``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="website", help="output directory (default: website)")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="repository URL used for source links")
    parser.add_argument("--models", nargs="+", default=None, help="build only these models")
    parser.add_argument("--no-figures", action="store_true", help="do not render figures")
    parser.add_argument("--no-examples", action="store_true", help="skip the example pages")
    parser.add_argument("--no-run-examples", action="store_true", help="do not execute the example scripts")
    parser.add_argument("--no-notebook", action="store_true", help="skip the notebook page")
    parser.add_argument("--no-docs", action="store_true", help="skip the documentation pages")
    parser.add_argument("--no-study", action="store_true", help="skip the comparative study")
    args = parser.parse_args(argv)
    summary = build_site(
        args.out,
        repo_url=args.repo_url,
        models=args.models,
        include_figures=not args.no_figures,
        include_examples=not args.no_examples,
        run_examples=not args.no_run_examples,
        include_notebook=not args.no_notebook,
        include_docs=not args.no_docs,
        include_study=not args.no_study,
    )
    print(f"site written to {args.out}")
    print(f"  pages   : {summary['pages']}")
    print(f"  figures : {summary['figures']}")
    for name, value in summary["sections"].items():
        print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())