"""Allow the package to be executed directly: ``python -m sentinel``.

Delegates to the same entry point as ``app.py`` so both paths behave identically.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    # app.py lives beside the package at the repository root.
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import main

    sys.exit(main())
