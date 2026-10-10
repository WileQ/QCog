"""Render every visualization the library can produce, and report.

    python scripts/check_figures.py                       # check everything, write nothing
    python scripts/check_figures.py --out out/figures     # also write the images
    python scripts/check_figures.py --models auctions     # one model
    python scripts/check_figures.py --no-sweeps           # faster: skip the parameter sweeps
    python scripts/check_figures.py --standalone-only     # just the figure functions

Exits non-zero if any figure fails to render or lacks its accessible description, so it
can be used as a check in a pipeline as well as by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qcog.viz import catalogue_checks, standalone_checks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="directory to write the images into")
    parser.add_argument("--models", nargs="+", default=None, help="model slugs to check")
    parser.add_argument("--no-sweeps", action="store_true", help="skip the models' parameter sweeps")
    parser.add_argument("--standalone-only", action="store_true", help="check only the figure functions")
    args = parser.parse_args(argv)

    rows = standalone_checks(args.out)
    if not args.standalone_only:
        rows += catalogue_checks(args.models, args.out, sweeps=not args.no_sweeps)

    width = max(len(row["name"]) for row in rows) + 2
    failures = 0
    for row in rows:
        print(f"{'PASS' if row['ok'] else 'FAIL'}  {row['name']:<{width}} {row['detail']}")
        failures += 0 if row["ok"] else 1
    print()
    print(f"{len(rows) - failures}/{len(rows)} visualizations rendered"
          + (f"; {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())