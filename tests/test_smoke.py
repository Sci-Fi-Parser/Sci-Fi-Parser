"""Import-smoke tests for the accuracy package.

CI's ``uv run pytest`` doesn't set PYTHONPATH, so these double as a check that
``uv sync`` actually installs the project (i.e. ``sci_fi_parser`` is importable
as a real package, not just via the ``src/`` shim path).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_schema_module_imports():
    from sci_fi_parser.schema import (
        ChartData, Extractor, Point, Series, normalize_key, parse_chartdata,
    )

    # Round-trip: dict -> ChartData -> series_map.
    cd = ChartData(chart_type=None, confidence=None,
                   series=[Series(name="s", points=[Point(x="a", y=1.0)])])
    assert cd.series_map() == {("s", "a"): 1.0}

    # parse_chartdata strips code fences from messy VLM-style output.
    parsed = parse_chartdata(
        '```json\n{"chart_type": null, "series": [], "confidence": null}\n```')
    assert parsed.series == []

    # Extractor is runtime-checkable.
    class Dummy:
        name = "d"
        def extract(self, image_path):  # noqa: ARG002
            return ChartData(chart_type=None, series=[], confidence=None)
    assert isinstance(Dummy(), Extractor)

    # Fuzzy match key: case- and whitespace-insensitive.
    assert normalize_key("Region A", "X") == normalize_key("region a ", " x")


def test_chart_type_literal_constrained():
    """chart_type is a Literal — VLMs (via ollama format=) can't drift outside the set."""
    from pydantic import ValidationError
    from sci_fi_parser.schema import ChartData

    # Valid value: round-trips fine.
    cd = ChartData(chart_type="bar_chart", series=[], confidence=None)
    assert cd.chart_type == "bar_chart"

    # Invalid value: rejected.
    with pytest.raises(ValidationError):
        ChartData(chart_type="not_a_chart", series=[], confidence=None)


def test_schema_is_light():
    # The schema module must not pull in matplotlib / opencv / ollama; consumers
    # (cv extractor, main pipeline) should be able to import it cheaply.
    import subprocess as sp
    proc = sp.run(
        [sys.executable, "-c",
         "import sci_fi_parser.schema, sys; "
         "heavy = {'matplotlib', 'cv2', 'ollama'}; "
         "leaked = heavy & set(sys.modules); "
         "sys.exit(0 if not leaked else 1)"],
        check=False,
    )
    assert proc.returncode == 0, "sci_fi_parser.schema leaked a heavy import"


def test_synthetic_package_imports():
    from sci_fi_parser.accuracy.synthetic import cli, config, generate, output

    # Public API the cli + downstream consumers rely on.
    assert callable(cli.main)
    assert "simple" in config.CATALOG
    assert callable(generate.generate_series)
    assert callable(output.png_bytes)
    assert isinstance(output.SQLITE_DDL, str)


def test_benchmark_module_imports():
    from sci_fi_parser.accuracy import benchmark

    assert callable(benchmark.main)
    # Canonical schema + scoring entry points.
    assert hasattr(benchmark, "ChartData")
    assert callable(benchmark.score_chart)
    assert callable(benchmark.aggregate)


def test_accuracy_init_does_not_pull_matplotlib():
    # The package __init__ deliberately doesn't import `synthetic`, so a caller
    # that only needs `benchmark` shouldn't pay the matplotlib import cost.
    # Runs in a subprocess because module attribute state is process-global.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sci_fi_parser.accuracy, sys; "
         "sys.exit(0 if 'matplotlib' not in sys.modules else 1)"],
        check=False,
    )
    assert proc.returncode == 0, "importing sci_fi_parser.accuracy pulled in matplotlib"
