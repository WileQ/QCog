"""Build the static website.

    python scripts/build_website.py                    # the whole site into website/
    python scripts/build_website.py --models question-order auctions
    python scripts/build_website.py --no-run-examples   # faster: no example output captured

The site is generated from the library, so it always matches the code.  It is plain HTML with one
stylesheet and no JavaScript: GitHub Pages serves it unchanged (see
`.github/workflows/pages.yml`), and a reader can also open `website/index.html` from disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qcog.website import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))