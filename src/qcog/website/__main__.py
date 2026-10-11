"""``python -m qcog.website`` - build the static site from a checkout.

    python -m qcog.website                     # into website/
    python -m qcog.website --out out/site --models question-order auctions
    python -m qcog.website --no-run-examples   # faster: no example output captured
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
