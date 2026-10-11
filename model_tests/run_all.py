"""Run every model's test file and collect the result.

    python model_tests/run_all.py                 # all 23 models, figures and reports written
    python model_tests/run_all.py --fast          # fewer shots, no Aer comparison
    python model_tests/run_all.py --models question-order auctions
    python model_tests/run_all.py --no-save       # print only, write nothing

Each ``module_*/NN_*.py`` file tests one model of the catalogue: the shared battery in
``harness.py`` plus the checks that only make sense for that model.  The run writes one HTML report
per model (its figures, their descriptions, the numbers behind them and every check) and a summary
of all twenty-three into ``model_tests/out/``.
"""

from __future__ import annotations

import argparse
import html
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import import_file, print_model_summary, run_model  # noqa: E402

MODULE_FOLDERS = ("module_a", "module_b", "module_c", "module_d")


def discover(folder: Path) -> list[Path]:
    """The test files of one module folder, in catalogue order."""
    return sorted(path for path in folder.glob("*.py") if path.name[0].isdigit())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="fewer shots and no Aer comparison")
    parser.add_argument("--no-save", action="store_true", help="do not write figures or reports")
    parser.add_argument("--models", nargs="+", default=None, help="only these model slugs")
    parser.add_argument("--out", default=str(HERE / "out"), help="where the reports are written")
    parser.add_argument("--quiet", action="store_true", help="only the per-model summary lines")
    args = parser.parse_args(argv)

    files = [path for folder in MODULE_FOLDERS for path in discover(HERE / folder)]
    if args.models:
        wanted = set(args.models)
        files = [path for path in files if import_file(path).SLUG in wanted]
    if not files:
        print("no test files found", file=sys.stderr)
        return 2

    started = time.perf_counter()
    summaries: list[dict] = []
    print(f"model_tests: {len(files)} models", flush=True)
    for path in files:
        module = import_file(path)
        try:
            summary = run_model(
                module.SLUG,
                module.specific,
                module=module.MODULE,
                out_dir=args.out,
                fast=args.fast,
                quiet=bool(args.quiet),
                save=not args.no_save,
                title=module.TITLE,
            )
        except Exception as error:  # a broken test file must not hide the others
            print(f"ERROR {path.name}: {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            summaries.append(
                {
                    "slug": module.SLUG,
                    "title": module.TITLE,
                    "module": module.MODULE,
                    "checks": 0,
                    "passed": 0,
                    "failed": 1,
                    "failures": [{"name": f"the file {path.name} raised", "detail": f"{type(error).__name__}: {error}"}],
                    "notes": [],
                    "figures": 0,
                    "runs": 0,
                    "seconds": 0.0,
                    "report": "",
                }
            )
            continue
        summaries.append(summary)
        if args.quiet:
            print_model_summary(summary, verbose=True)

    elapsed = time.perf_counter() - started
    checks = sum(summary["checks"] for summary in summaries)
    passed = sum(summary["passed"] for summary in summaries)
    failed = sum(summary["failed"] for summary in summaries)
    notes = sum(len(summary["notes"]) for summary in summaries)
    figures = sum(summary["figures"] for summary in summaries)
    runs = sum(summary["runs"] for summary in summaries)
    print()
    print(
        f"{len(summaries)} models, {checks} checks ({passed} passed, {failed} failed), "
        f"{notes} notes, {figures} figures, {runs} runs, {elapsed:.1f}s"
    )
    if failed:
        print("failing checks:")
        for summary in summaries:
            for failure in summary["failures"]:
                print(f"  {summary['slug']}: {failure['name']} - {failure['detail']}")
    for summary in summaries:
        for note in summary["notes"]:
            print(f"  note {summary['slug']}: {note['name']} - {note['detail']}")

    if not args.no_save:
        summary_path = write_summary(Path(args.out), summaries, elapsed)
        print(f"\nsummary: {summary_path}")
        print(f"reports: {Path(args.out)}/<slug>/report.html")
    return 1 if failed else 0


def write_summary(out_dir: Path, summaries: list[dict], elapsed: float) -> Path:
    """A Markdown summary of the whole run, plus a small HTML index into the reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = sum(summary["checks"] for summary in summaries)
    failed = sum(summary["failed"] for summary in summaries)
    notes = sum(len(summary["notes"]) for summary in summaries)
    lines = [
        "# Model tests: every model in the catalogue",
        "",
        f"{len(summaries)} models, {checks} checks ({checks - failed} passed, {failed} failed), "
        f"{notes} notes, {sum(summary['figures'] for summary in summaries)} figures, "
        f"{sum(summary['runs'] for summary in summaries)} model runs, {elapsed:.1f} s.",
        "",
        "A **note** is not a failure: it is something a human should look at - a chart that is correct",
        "but shows nothing, a drawing the simulator declined, a sweep that disagrees with the run.",
        "",
        "| Model | Module | Checks | Figures | Runs | Time | Report |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        mark = "**FAIL**" if summary["failed"] else "pass"
        report = f"[{summary['slug']}]({summary['slug']}/report.html)" if summary["report"] else summary["slug"]
        lines.append(
            f"| {summary['title']} | {summary['module']} | {summary['passed']}/{summary['checks']} {mark} "
            f"| {summary['figures']} | {summary['runs']} | {summary['seconds']}s | {report} |"
        )
    lines += ["", "## Notes worth reading", ""]
    for summary in summaries:
        for note in summary["notes"]:
            lines.append(f"* **{summary['slug']}** - {note['name']}: {note['detail']}")
    if failed:
        lines += ["", "## Failing checks", ""]
        for summary in summaries:
            for failure in summary["failures"]:
                lines.append(f"* **{summary['slug']}** - {failure['name']}: {failure['detail']}")
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    index = out_dir / "index.html"
    body = "\n".join(
        f"<tr><td><a href='{summary['slug']}/report.html'>{html.escape(summary['title'])}</a></td>"
        f"<td>{summary['module']}</td><td>{summary['passed']}/{summary['checks']}</td>"
        f"<td>{summary['figures']}</td><td>{len(summary['notes'])}</td></tr>"
        for summary in summaries
    )
    index.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Model tests</title></head><body>"
        f"<h1>Model tests: {len(summaries)} models, {checks - failed}/{checks} checks passed</h1>"
        "<p>One report per model: its figures, their descriptions, the numbers behind them and every check.</p>"
        "<table><thead><tr><th>Model</th><th>Module</th><th>Checks</th><th>Figures</th><th>Notes</th></tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
