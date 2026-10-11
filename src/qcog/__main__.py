"""``python -m qcog`` - the same interface as the ``qcog`` command.

Provided so that the library can be used without installing the console script,
which matters in restricted environments where only a checkout is available.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())