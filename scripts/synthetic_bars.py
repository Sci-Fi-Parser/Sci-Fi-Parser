"""Thin shim — the generator lives in ``sci_fi_parser.accuracy.synthetic``.

Keeps ``python scripts/synthetic_bars.py …`` working (docs/CI/muscle memory)
without making ``scripts`` part of the importable package. The real CLI is in
``src/sci_fi_parser/accuracy/synthetic/cli.py``.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from sci_fi_parser.accuracy.synthetic.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
