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
    from sci_fi_parser.accuracy.benchmark import normalize_key, series_map
    from sci_fi_parser.vlm.vlm_schema import (
        ChartData, Extractor, Point, Series, parse_chartdata,
    )

    # Round-trip: dict -> ChartData -> series_map.
    cd = ChartData(chart_type=None, confidence=None,
                   series=[Series(name="s", points=[Point(x="a", y=1.0)])])
    assert series_map(cd) == {("s", "a"): 1.0}

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
    from sci_fi_parser.vlm.vlm_schema import ChartData

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
         "import sci_fi_parser.vlm.vlm_schema, sys; "
         "heavy = {'matplotlib', 'cv2', 'ollama'}; "
         "leaked = heavy & set(sys.modules); "
         "sys.exit(0 if not leaked else 1)"],
        check=False,
    )
    assert proc.returncode == 0, "sci_fi_parser.vlm.vlm_schema leaked a heavy import"


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
    from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile

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


def test_score_chart_value_vs_identity():
    """Positional vs identity scoring disentangles three failure modes.

    Worked from the 4-bar Profit chart in the discussion:
    truth = Q1'20:445.08, Q2'20:504.55, Q3'20:553.50, Q4'20:995.50.

    - Case A: clean -> matched=4, both value_errors_pos and errors_pct small,
      misaligned=0.
    - Case B: heights perfectly read but x-labels shifted by one column;
      this is the case identity-scoring conflates with bad value-reading.
      value_errors_pos must stay small, misaligned must be > 0, and
      identity matched < 4 (so the existing recall column still flags it).
    - Case C: 10x scale (the qwen-3b bug). matched=4, misaligned=0, but
      both errors_pct and value_errors_pos are several times the axis span
      (errors normalize to the value_range, not to |true|).
    """
    from sci_fi_parser.accuracy.benchmark import score_chart
    from sci_fi_parser.vlm.vlm_schema import ChartData, Point, Series

    truth_entry = {
        "chart_type": "bar_chart",
        "series": [{"name": "Profit",
                    "points": [["Q1'20", 445.08], ["Q2'20", 504.55],
                               ["Q3'20", 553.50], ["Q4'20", 995.50]]}],
        # Drawn value-axis extent; errors are scored as a % of this span.
        "value_range": [0.0, 1050.0],
    }

    def _chart(name_value_pairs):
        return ChartData(
            chart_type="bar_chart",
            series=[Series(name="Profit",
                           points=[Point(x=cat, y=val)
                                   for cat, val in name_value_pairs])],
            confidence=0.95,
        )

    # Case A: clean
    pred_a = _chart([("Q1'20", 450), ("Q2'20", 500),
                     ("Q3'20", 550), ("Q4'20", 1000)])
    res_a = score_chart("img.png", truth_entry, pred_a)
    assert res_a.matched == 4
    assert res_a.misaligned == 0
    assert res_a.bar_count_err == 0
    assert max(res_a.errors_pct) < 5      # all under 5%
    assert max(res_a.value_errors_pos) < 5

    # Case B: heights right but labels shifted left by one column. The model
    # claims Q4'19/Q1'20/Q2'20/Q3'20 with the heights actually belonging to
    # Q1'20..Q4'20. Identity says only 3 match (the truth's Q4'20 vanishes,
    # the model's Q4'19 is extra). Positional pairing aligns by index, so
    # the values pair with their original truth values -- and all four
    # paired positions have label disagreement.
    pred_b = _chart([("Q4'19", 445.08), ("Q1'20", 504.55),
                     ("Q2'20", 553.50), ("Q3'20", 995.50)])
    res_b = score_chart("img.png", truth_entry, pred_b)
    assert res_b.matched == 3          # Q1/Q2/Q3 match by name
    assert res_b.misaligned == 4       # all four paired positions disagree
    assert res_b.n_paired_pos == 4
    assert res_b.bar_count_err == 0
    # All four positionally-paired values are bit-perfect -> ~0% error.
    assert max(res_b.value_errors_pos) < 1e-6
    # The identity-conditional errors look bad (different values lined up
    # to the same label) -- which is exactly the conflation we're fixing.
    assert max(res_b.errors_pct) > 10

    # Case C: identity right, values 10x too big.
    pred_c = _chart([("Q1'20", 4500.0), ("Q2'20", 5050.0),
                     ("Q3'20", 5530.0), ("Q4'20", 9950.0)])
    res_c = score_chart("img.png", truth_entry, pred_c)
    assert res_c.matched == 4
    assert res_c.misaligned == 0
    assert res_c.bar_count_err == 0
    # ~10x values against a 1050-wide span -> hundreds of % on every bar.
    assert min(res_c.errors_pct) > 300
    assert min(res_c.value_errors_pos) > 300


def test_vlm_comparison_resume_skip(tmp_path):
    """--resume reconstructs a row from an existing results.json instead of
    re-running. With resume off (or no prior file) the helper returns None."""
    import json as _json
    from sci_fi_parser.accuracy.vlm_compare import (
        CompareEntry, _maybe_resume_row,
    )
    from sci_fi_parser.vlm.vlm_config import VLMProfile

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
