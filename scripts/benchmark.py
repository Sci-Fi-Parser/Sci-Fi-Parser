"""Thin shim — the benchmark lives in ``sci_fi_parser.accuracy.benchmark``.

Keeps ``python scripts/benchmark.py …`` working without making ``scripts`` part
of the importable package. The real implementation is in
``src/sci_fi_parser/accuracy/benchmark.py``.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from sci_fi_parser.accuracy.benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
