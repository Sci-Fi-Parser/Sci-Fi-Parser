"""Import-smoke tests for the accuracy package.

CI's ``uv run pytest`` doesn't set PYTHONPATH, so these double as a check that
``uv sync`` actually installs the project (i.e. ``sci_fi_parser`` is importable
as a real package, not just via the ``src/`` shim path).
"""

from __future__ import annotations

import subprocess
import sys


def test_schema_module_imports():
    from sci_fi_parser.schema import (
        ChartData, Extractor, Point, Series, parse_chartdata,
    )

    # Round-trip: dict -> ChartData -> series_map.
    cd = ChartData(series=[Series(name="s", points=[Point(x="a", y=1.0)])])
    assert cd.series_map() == {("s", "a"): 1.0}

    # parse_chartdata strips code fences from messy VLM-style output.
    parsed = parse_chartdata('```json\n{"series": []}\n```')
    assert parsed.series == []

    # Extractor is runtime-checkable.
    class Dummy:
        name = "d"
        def extract(self, image_path):  # noqa: ARG002
            return ChartData(series=[])
    assert isinstance(Dummy(), Extractor)


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
