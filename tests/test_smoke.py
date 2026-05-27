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


def test_vlm_profile_load(tmp_path):
    """The VLM profile is loaded from a TOML file; unknown keys are rejected."""
    from sci_fi_parser.accuracy.vlm_config import VLMProfile, load_profile

    p = tmp_path / "vlm.toml"
    p.write_text(
        'model = "test-model:1b"\n'
        'prompt = "hello"\n'
        'num_ctx = 1024\n',
        encoding="utf-8",
    )
    profile = load_profile(p)
    assert profile.model == "test-model:1b"
    assert profile.prompt == "hello"
    assert profile.num_ctx == 1024
    assert profile.num_gpu is None       # dataclass default
    assert isinstance(VLMProfile().model, str)  # built-in default is non-empty

    # Typos must raise — silent fallthrough would be a debugging nightmare.
    bad = tmp_path / "bad.toml"
    bad.write_text('modle = "oops"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(bad)


def test_vlm_comparison_config_load(tmp_path):
    """Comparison TOML parses [defaults] + [[model]] entries into VLMProfiles."""
    from sci_fi_parser.accuracy.vlm_compare import load_comparison_config

    cfg = tmp_path / "compare.toml"
    cfg.write_text(
        '[defaults]\n'
        'num_ctx = 1024\n'
        'prompt = "p"\n'
        '\n'
        '[[model]]\n'
        'name = "a"\n'
        'model = "qwen2.5vl:7b"\n'
        '\n'
        '[[model]]\n'
        'name = "b"\n'
        'model = "qwen2.5vl:3b"\n'
        'num_ctx = 4096\n',
        encoding="utf-8",
    )
    run, entries = load_comparison_config(cfg)
    assert run.data is None and run.pull is None  # no [run] table
    assert [e.name for e in entries] == ["a", "b"]
    # Defaults flow through; per-model override wins.
    assert entries[0].profile.num_ctx == 1024
    assert entries[1].profile.num_ctx == 4096
    assert all(e.profile.prompt == "p" for e in entries)

    # Empty file rejected.
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_comparison_config(empty)

    # Duplicate names rejected.
    dup = tmp_path / "dup.toml"
    dup.write_text(
        '[[model]]\nname = "x"\nmodel = "a"\n'
        '[[model]]\nname = "x"\nmodel = "b"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_comparison_config(dup)


def test_vlm_comparison_extends(tmp_path):
    """extends= loads a base VLMProfile; [defaults]/[[model]] layer on top."""
    from sci_fi_parser.accuracy.vlm_compare import load_comparison_config

    # Base profile with a custom prompt and num_ctx -- the values we want to
    # inherit downstream without duplicating them.
    base = tmp_path / "base.toml"
    base.write_text(
        'model = "qwen2.5vl:7b"\n'
        'prompt = "BASE PROMPT"\n'
        'num_ctx = 1024\n',
        encoding="utf-8",
    )

    cmp_cfg = tmp_path / "cmp.toml"
    cmp_cfg.write_text(
        'extends = "base.toml"\n'
        '[[model]]\nname = "a"\nmodel = "qwen2.5vl:7b"\n'
        '[[model]]\nname = "b"\nmodel = "qwen2.5vl:3b"\nnum_ctx = 4096\n',
        encoding="utf-8",
    )
    _, entries = load_comparison_config(cmp_cfg)
    # Both entries inherit the base prompt; the second overrides num_ctx.
    assert all(e.profile.prompt == "BASE PROMPT" for e in entries)
    assert entries[0].profile.num_ctx == 1024  # inherited
    assert entries[1].profile.num_ctx == 4096  # per-model override

    # [defaults] layers between extends and [[model]] -- so it wins over the
    # base but loses to per-model overrides. Verifies the merge order.
    cmp_layer = tmp_path / "cmp_layer.toml"
    cmp_layer.write_text(
        'extends = "base.toml"\n'
        '[defaults]\nnum_ctx = 2048\n'
        '[[model]]\nname = "a"\nmodel = "x"\n'
        '[[model]]\nname = "b"\nmodel = "y"\nnum_ctx = 8192\n',
        encoding="utf-8",
    )
    _, layered = load_comparison_config(cmp_layer)
    assert layered[0].profile.num_ctx == 2048
    assert layered[1].profile.num_ctx == 8192
    # Prompt still flows through from the base since [defaults] didn't override.
    assert layered[0].profile.prompt == "BASE PROMPT"

    # Missing extends file -> ValueError, not a silent fall-through to defaults.
    missing = tmp_path / "missing.toml"
    missing.write_text(
        'extends = "nope.toml"\n'
        '[[model]]\nname = "a"\nmodel = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_comparison_config(missing)


def test_vlm_comparison_resume_skip(tmp_path):
    """--resume reconstructs a row from an existing results.json instead of
    re-running. With resume off (or no prior file) the helper returns None."""
    import json as _json
    from sci_fi_parser.accuracy.vlm_compare import (
        CompareEntry, _maybe_resume_row,
    )
    from sci_fi_parser.accuracy.vlm_config import VLMProfile

    entry = CompareEntry(name="qwen-3b", profile=VLMProfile(model="qwen2.5vl:3b"))

    # No prior file -> None.
    assert _maybe_resume_row(entry, tmp_path, resume=True) is None

    # Write a minimal results.json that mimics what run_benchmark emits.
    row_dir = tmp_path / entry.name
    row_dir.mkdir()
    (row_dir / "results.json").write_text(
        _json.dumps({
            "extractor": entry.profile.model,
            "aggregate": {
                "n_charts": 40, "recall": 0.83, "precision": 0.87,
                "mean_pct": 7.31, "type_accuracy": 1.0,
            },
        }),
        encoding="utf-8",
    )

    # resume=False -> still None, even with the file present.
    assert _maybe_resume_row(entry, tmp_path, resume=False) is None

    # resume=True -> row dict carrying the aggregate, no error key.
    row = _maybe_resume_row(entry, tmp_path, resume=True)
    assert row is not None
    assert row["name"] == "qwen-3b"
    assert row["tag"] == "qwen2.5vl:3b"
    assert row["recall"] == 0.83
    assert row["mean_pct"] == 7.31
    assert "error" not in row


def test_vlm_comparison_run_table(tmp_path):
    """[run] table is parsed and validated; CLI-style values come back typed."""
    from pathlib import Path as _Path
    from sci_fi_parser.accuracy.vlm_compare import load_comparison_config

    cfg = tmp_path / "with_run.toml"
    cfg.write_text(
        '[run]\n'
        'data = "data/eval"\n'
        'out  = "reports/x"\n'
        'limit = 3\n'
        'seed  = 7\n'
        'pull  = "circular"\n'
        '[[model]]\nname = "m"\nmodel = "qwen2.5vl:7b"\n',
        encoding="utf-8",
    )
    run, entries = load_comparison_config(cfg)
    assert run.data == _Path("data/eval")
    assert run.out == _Path("reports/x")
    assert run.limit == 3
    assert run.seed == 7
    assert run.pull == "circular"
    assert len(entries) == 1

    # Unknown [run] key -> ValueError (typos must not silently slip through).
    bad_key = tmp_path / "bad_key.toml"
    bad_key.write_text(
        '[run]\ndatas = "x"\n[[model]]\nname = "m"\nmodel = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_comparison_config(bad_key)

    # Invalid pull mode -> ValueError.
    bad_pull = tmp_path / "bad_pull.toml"
    bad_pull.write_text(
        '[run]\npull = "yolo"\n[[model]]\nname = "m"\nmodel = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_comparison_config(bad_pull)


def test_vlm_comparison_pull_mode_resolution():
    """resolve_pull_mode encodes the CLI-vs-prompt-vs-error matrix."""
    from sci_fi_parser.accuracy.vlm_compare import resolve_pull_mode

    # No missing models -> mode is irrelevant; always returns "skip".
    assert resolve_pull_mode([], None, False) == "skip"
    assert resolve_pull_mode([], "prefetch", True) == "skip"

    # Explicit --pull wins, no prompt, with or without --yes.
    assert resolve_pull_mode(["a"], "prefetch", False) == "prefetch"
    assert resolve_pull_mode(["a"], "circular", True) == "circular"
    assert resolve_pull_mode(["a"], "skip", False) == "skip"

    # --yes without --pull AND missing models is a footgun -> SystemExit,
    # not a silent default. Catches scripts that forgot to choose a mode.
    with pytest.raises(SystemExit):
        resolve_pull_mode(["a"], None, True)


def test_vlm_comparison_preflight_format():
    """format_preflight renders LOCAL/MISSING + sizes + free disk + mode."""
    from sci_fi_parser.accuracy.vlm_compare import (
        ModelStatus, format_preflight,
    )

    statuses = [
        ModelStatus(tag="qwen2.5vl:7b", local=True, size_bytes=5_000_000_000),
        ModelStatus(tag="qwen2.5vl:7b-q8_0", local=False, size_bytes=None),
    ]
    out = format_preflight(statuses, mode="circular")
    assert "LOCAL" in out
    assert "MISSING" in out
    assert "qwen2.5vl:7b" in out and "qwen2.5vl:7b-q8_0" in out
    assert "(pull required)" in out
    assert "local total" in out
    assert "free disk" in out
    assert "mode:" in out and "circular" in out


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
