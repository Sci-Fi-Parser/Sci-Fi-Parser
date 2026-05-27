"""Run the benchmark across multiple VLM profiles and render a leaderboard.

Reads ``config/vlm_comparison.toml`` (or a path given on the CLI), runs the
benchmark once per model, and emits a single comparison view linking out to
each model's individual report. Failed runs surface as a row in the
leaderboard rather than aborting the whole comparison.

Outputs under ``<out>/``:

* ``<model_name>/report.html`` + ``results.json`` — per-model benchmark.
* ``leaderboard.md`` — markdown table (good for terminals and git).
* ``leaderboard.html`` — self-contained HTML with links to per-model reports.

The contract: model entries are pulled from ``[[model]]`` arrays; each entry
may override any field of :class:`VLMProfile`. Shared values go in
``[defaults]``. The ``name`` field is the *display name* and used as the
report subdirectory — it does not have to match the ollama tag.

Top-level ``extends = "vlm.toml"`` (path relative to the referring TOML's
directory) loads that file as the base profile. The merge order is:
dataclass defaults -> ``extends`` file -> ``[defaults]`` -> ``[[model]]``.
This keeps the canonical prompt and tunables in *one* place (``vlm.toml``)
instead of duplicated across every comparison config.

A ``[run]`` table holds run-level defaults (``data``, ``out``, ``limit``,
``seed``, ``pull``) so a config can be self-contained. CLI flags override
``[run]`` values one-for-one. ``[run].data`` is required if ``--data`` is
not passed. ``[run]`` paths resolve against the cwd (same as CLI args);
``extends`` resolves against the TOML's own directory.

Before any run, a *preflight* lists each unique tag as LOCAL (with on-disk
size) or MISSING, plus free disk on the ollama models dir. If any tag is
missing, the runner asks ``--pull {prefetch,circular,skip}`` (interactively
when no flag was passed):

* ``prefetch`` -- pull every missing model up front, leave them after.
* ``circular`` -- pull, run, ``ollama rm`` per model; only removes what we
  pulled this run (preexisting locals are never touched).
* ``skip``     -- don't pull; missing models become explicit ERROR rows.

``--dry-run`` prints the preflight and exits. ``--yes`` skips the
"this will pull N models" confirmation; combined with a missing model and no
``--pull`` it errors out rather than guessing a mode.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import sys
import time
import tomllib
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path

from sci_fi_parser.accuracy.benchmark import run_benchmark
from sci_fi_parser.accuracy.vlm_config import VLMProfile, load_profile


@dataclass(slots=True)
class CompareEntry:
    name: str
    profile: VLMProfile


@dataclass(slots=True)
class RunConfig:
    """Defaults for a comparison run, parsed from the TOML's ``[run]`` table.

    Every field is optional -- ``None`` means "no opinion, let the CLI or a
    hard-coded default decide". Relative paths are resolved against the user's
    cwd at the time the runner is invoked (same as the CLI flag).
    """
    data: Path | None = None
    out: Path | None = None
    limit: int | None = None
    seed: int | None = None
    pull: str | None = None


_RUN_FIELDS = {f.name for f in fields(RunConfig)}


def _slice_profile_fields(d: dict, path: Path) -> dict:
    """Drop ``name`` and reject keys that aren't on :class:`VLMProfile`."""
    profile_fields = {f.name for f in fields(VLMProfile)}
    unknown = set(d) - profile_fields - {"name"}
    if unknown:
        raise ValueError(
            f"unknown key(s) in {path}: {sorted(unknown)}; "
            f"valid: {sorted(profile_fields | {'name'})}")
    return {k: v for k, v in d.items() if k in profile_fields}


def _load_extends_base(raw: dict, path: Path) -> dict:
    """If top-level ``extends = "..."`` is set, load that file as a base.

    Path is resolved against the *referring* TOML's parent dir (matches every
    other include convention -- compare/smoke configs can sit alongside the
    profile they reference). Returns the base profile as a dict of every
    :class:`VLMProfile` field; ``{}`` if no ``extends`` key.
    """
    if "extends" not in raw:
        return {}
    ext_path = (path.parent / raw["extends"]).resolve()
    if not ext_path.exists():
        raise ValueError(
            f"{path}: extends -> {ext_path} (file not found)")
    base = load_profile(ext_path)
    return {f.name: getattr(base, f.name) for f in fields(VLMProfile)}


def _parse_run_table(raw_run: dict, path: Path) -> RunConfig:
    """Validate the ``[run]`` table and coerce path-typed fields."""
    unknown = set(raw_run) - _RUN_FIELDS
    if unknown:
        raise ValueError(
            f"unknown [run] key(s) in {path}: {sorted(unknown)}; "
            f"valid: {sorted(_RUN_FIELDS)}")
    pull = raw_run.get("pull")
    if pull is not None and pull not in _MODE_CHOICES:
        raise ValueError(
            f"[run].pull in {path} must be one of {list(_MODE_CHOICES)}, "
            f"got {pull!r}")
    return RunConfig(
        data=Path(raw_run["data"]) if "data" in raw_run else None,
        out=Path(raw_run["out"]) if "out" in raw_run else None,
        limit=raw_run.get("limit"),
        seed=raw_run.get("seed"),
        pull=pull,
    )


def load_comparison_config(path: Path) -> tuple[RunConfig, list[CompareEntry]]:
    """Parse the comparison TOML.

    Returns ``(run_config, entries)``. The optional ``[run]`` table holds
    run-level defaults; ``[defaults]`` is the per-model base profile;
    per-model ``[[model]]`` entries override both. Unknown keys at any level
    raise :class:`ValueError`.
    """
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    run = _parse_run_table(raw.get("run", {}), path)
    base = _load_extends_base(raw, path)
    defaults = {**base, **_slice_profile_fields(raw.get("defaults", {}), path)}
    entries: list[CompareEntry] = []
    seen: set[str] = set()
    for raw_entry in raw.get("model", []):
        if "name" not in raw_entry:
            raise ValueError(f"every [[model]] in {path} needs a 'name' field")
        name = raw_entry["name"]
        if name in seen:
            raise ValueError(f"duplicate model name {name!r} in {path}")
        seen.add(name)
        overrides = _slice_profile_fields(raw_entry, path)
        entries.append(CompareEntry(
            name=name, profile=VLMProfile(**{**defaults, **overrides})))
    if not entries:
        raise ValueError(f"{path}: no [[model]] entries defined")
    return run, entries


# --------------------------------------------------------------------------- #
# Ollama admin helpers (list / show / pull / delete) via stdlib urllib.
# We avoid the `ollama` Python package here so this module doesn't track its
# evolving response shapes -- the HTTP endpoints have been stable.
# --------------------------------------------------------------------------- #
def _ollama_url(path: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return f"{host}{path}"


def _ollama_request(method: str, path: str, body: dict | None = None,
                    *, timeout: float = 30.0, stream: bool = False):
    """Return a response object (caller-closed if streaming) or the parsed JSON.

    Raises :class:`OSError` if the daemon is unreachable.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        _ollama_url(path), method=method, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    resp = urllib.request.urlopen(req, timeout=timeout)  # pylint: disable=consider-using-with
    if stream:
        return resp
    try:
        raw = resp.read()
        return json.loads(raw) if raw else {}
    finally:
        resp.close()


def _list_local_models() -> dict[str, int]:
    """Map of locally-pulled tag -> size in bytes. Empty dict if daemon down."""
    try:
        data = _ollama_request("GET", "/api/tags", timeout=5)
    except (OSError, urllib.error.URLError):
        return {}
    out: dict[str, int] = {}
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            out[name] = int(m.get("size") or 0)
    return out


def _handle_pull_event(tag: str, raw: bytes, last_status: str) -> str:
    """Parse one pull-stream event; print status if it changed. Returns new last."""
    try:
        evt = json.loads(raw)
    except json.JSONDecodeError:
        return last_status
    if "error" in evt:
        raise RuntimeError(f"ollama pull {tag}: {evt['error']}")
    status = evt.get("status", "")
    if status and status != last_status:
        print(f"    {status}", flush=True)
        return status
    return last_status


def _pull_model(tag: str) -> None:
    """Stream-pull a model, printing the status line as it changes."""
    print(f"  pulling {tag} ...", flush=True)
    resp = _ollama_request("POST", "/api/pull",
                           {"name": tag, "stream": True},
                           timeout=None, stream=True)
    try:
        last_status = ""
        for raw in resp:
            last_status = _handle_pull_event(tag, raw, last_status)
    finally:
        resp.close()


def _delete_model(tag: str) -> None:
    """Remove a model from the local ollama store."""
    _ollama_request("DELETE", "/api/delete", {"name": tag}, timeout=30)


# --------------------------------------------------------------------------- #
# Preflight: classify each unique tag as local/missing, render a report.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ModelStatus:
    tag: str
    local: bool
    size_bytes: int | None  # known only for local models


def _human_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} PB"  # unreachable, keeps linter happy


def _ollama_models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


def _disk_free(path: Path) -> int:
    """Free bytes on the FS holding `path`; 0 if the path doesn't exist."""
    probe = path if path.exists() else path.parent
    if not probe.exists():
        probe = Path.home()
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def inspect_preflight(entries: list[CompareEntry]) -> list[ModelStatus]:
    """One ModelStatus per *unique* model tag, in first-seen order."""
    local = _list_local_models()
    seen: dict[str, ModelStatus] = {}
    for e in entries:
        tag = e.profile.model
        if tag in seen:
            continue
        seen[tag] = ModelStatus(tag=tag, local=tag in local,
                                size_bytes=local.get(tag))
    return list(seen.values())


def format_preflight(statuses: list[ModelStatus],
                     mode: str | None = None) -> str:
    """Human-readable summary printed before the run."""
    name_w = max((len(s.tag) for s in statuses), default=10) + 2
    lines = ["Preflight:"]
    local_total = 0
    for s in statuses:
        if s.local:
            local_total += s.size_bytes or 0
            sz = _human_bytes(s.size_bytes)
            label = "LOCAL"
        else:
            sz = "(pull required)"
            label = "MISSING"
        lines.append(f"  {s.tag.ljust(name_w)} {label:<8} {sz}")
    free = _disk_free(_ollama_models_dir())
    lines.append("")
    lines.append(f"  local total: {_human_bytes(local_total)}")
    lines.append(f"  free disk:   {_human_bytes(free)}  "
                 f"({_ollama_models_dir()})")
    if mode:
        lines.append(f"  mode:        {mode}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Interactive prompts (mode + confirm). Refuse to block when not on a TTY.
# --------------------------------------------------------------------------- #
_MODE_CHOICES = ("prefetch", "circular", "skip")


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_mode(missing: list[str]) -> str:
    """Ask the user how to handle missing models. Returns a mode or aborts."""
    if not _is_interactive():
        raise SystemExit(
            f"missing models {missing} but stdin is not a TTY. "
            "Pass --pull {prefetch,circular,skip}.")
    print("\nMissing models need a decision:")
    for t in missing:
        print(f"  - {t}")
    print("\n  [p] prefetch  pull all missing up front, keep them")
    print("  [c] circular  pull -> run -> remove, one model at a time")
    print("  [s] skip      leave missing models as error rows")
    print("  [a] abort")
    while True:
        ch = input("> ").strip().lower()
        if ch in ("p", "prefetch"):
            return "prefetch"
        if ch in ("c", "circular"):
            return "circular"
        if ch in ("s", "skip"):
            return "skip"
        if ch in ("a", "abort", "q", "quit"):
            raise SystemExit("aborted by user")
        print("  pick p / c / s / a")


def _confirm_yn(prompt: str) -> bool:
    if not _is_interactive():
        raise SystemExit(
            f"{prompt} -- not a TTY; pass --yes to confirm non-interactively.")
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


# --------------------------------------------------------------------------- #
# Leaderboard rendering
# --------------------------------------------------------------------------- #
def _identity(v):
    return v


def _val(v: float | None, fmt: str = ".2f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)


def _pct(v: float | None) -> str:
    """Plain '% of true value' number, already in percent units."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.2f}%"


def _pct100(v: float | None) -> str:
    """0..1 fraction rendered as a percent (recall / precision / type_acc)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def _secs(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f} s"


_LEADER_COLS = [
    ("Model", "name", _identity),
    ("Tag", "tag", html.escape),
    ("Mean err", "mean_pct", _pct),
    ("Median", "median_pct", _pct),
    ("p95", "p95_pct", _pct),
    ("Recall", "recall", _pct100),
    ("Precision", "precision", _pct100),
    ("Type acc", "type_accuracy", _pct100),
    ("Mean conf", "mean_confidence", _val),
    ("Mean time", "mean_sec", _secs),
    ("Total time", "total_sec", _secs),
]


def _row_dict(name: str, tag: str, agg: dict | None,
              error: str | None) -> dict:
    """Combine the per-model aggregate into the columns the renderer wants."""
    if error is not None:
        return {"name": name, "tag": tag, "error": error}
    return {"name": name, "tag": tag, **agg}


def _render_md(rows: list[dict]) -> str:
    headers = [c[0] for c in _LEADER_COLS]
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        if "error" in r:
            err = f"ERROR: {r['error']}"
            cells = [r["name"], r["tag"], err] + [""] * (len(headers) - 3)
        else:
            cells = [fmt(r.get(key)) for _, key, fmt in _LEADER_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _html_cell(key: str, link: str, row: dict, fmt) -> str:
    """One leaderboard cell. ``name`` becomes a link; ``tag`` gets <code>."""
    if key == "name":
        return f"<td>{link}</td>"
    if key == "tag":
        return f"<td><code>{html.escape(row['tag'])}</code></td>"
    return f"<td>{fmt(row.get(key))}</td>"


def _html_row(row: dict) -> str:
    name_esc = html.escape(row["name"])
    link = f"<a href='{name_esc}/report.html'>{name_esc}</a>"
    if "error" in row:
        cells = [f"<td>{link}</td>", f"<td><code>{row['tag']}</code></td>",
                 f"<td colspan='{len(_LEADER_COLS)-2}' style='color:#a00'>"
                 f"ERROR: {html.escape(row['error'])}</td>"]
    else:
        cells = [_html_cell(key, link, row, fmt)
                 for _, key, fmt in _LEADER_COLS]
    return "<tr>" + "".join(cells) + "</tr>"


def _render_html(rows: list[dict], reports_dir: Path) -> str:
    headers = "".join(f"<th>{h}</th>" for h, *_ in _LEADER_COLS)
    body = "\n".join(_html_row(r) for r in rows)
    return f"""<!doctype html><meta charset="utf-8">
<title>VLM comparison leaderboard</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1a1a1a}}
 h1{{font-size:20px}}
 table{{border-collapse:collapse;font-size:13px}}
 th,td{{border-bottom:1px solid #eee;padding:6px 10px;text-align:right;
        white-space:nowrap}}
 th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
 th{{background:#f4f6f8}}
 a{{color:#1a5490;text-decoration:none}} a:hover{{text-decoration:underline}}
</style>
<h1>VLM comparison leaderboard</h1>
<p>Each row is one model in <code>{html.escape(str(reports_dir))}</code>.
Click the model name to see its full per-chart report.
Error = |predicted − true| as a percentage of the true value.</p>
<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>
"""


def write_leaderboard(rows: list[dict], out: Path) -> None:
    (out / "leaderboard.md").write_text(_render_md(rows) + "\n", encoding="utf-8")
    (out / "leaderboard.html").write_text(_render_html(rows, out), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _safe_run(entry: CompareEntry, data: Path, out: Path,
              seed: int, limit: int | None) -> tuple[dict | None, str | None]:
    """Run one model. Return (aggregate, error_str). Catches everything."""
    try:
        agg = run_benchmark(
            data=data, out=out, extractor_name="ollama",
            profile=entry.profile, seed=seed, limit=limit,
            print_summary=False,
        )
        return agg, None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        traceback.print_exc()
        return None, f"{type(exc).__name__}: {exc}"


def _print_progress(idx: int, total: int, entry: CompareEntry) -> None:
    print(f"\n[{idx}/{total}] {entry.name}  ({entry.profile.model})", flush=True)


def resolve_pull_mode(missing: list[str], pull: str | None,
                      assume_yes: bool) -> str:
    """Decide the pull mode given CLI flags + the set of missing tags.

    No missing models -> mode is irrelevant; return ``"skip"``.
    Explicit ``--pull`` wins. ``--yes`` without ``--pull`` and missing
    models is a footgun -> error out. Otherwise prompt the user.
    """
    if not missing:
        return "skip"
    if pull is not None:
        return pull
    if assume_yes:
        raise SystemExit(
            "missing models present and --yes given but no --pull mode; "
            "pick --pull {prefetch,circular,skip}.")
    return _prompt_mode(missing)


def _ensure_pulled(tag: str, local: set[str], pulled_by_us: set[str]) -> None:
    """If `tag` isn't local, pull it and record that we did so."""
    if tag in local:
        return
    _pull_model(tag)
    local.add(tag)
    pulled_by_us.add(tag)


def _maybe_drop(tag: str, mode: str, pulled_by_us: set[str],
                still_needed: set[str], local: set[str]) -> None:
    """In circular mode, remove a tag we pulled once no later entry needs it."""
    if mode != "circular" or tag not in pulled_by_us or tag in still_needed:
        return
    try:
        _delete_model(tag)
        local.discard(tag)
        pulled_by_us.discard(tag)
        print(f"  removed {tag}")
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        print(f"  WARNING: could not remove {tag}: {exc}", file=sys.stderr)


def _maybe_resume_row(entry: CompareEntry, out: Path,
                      resume: bool) -> dict | None:
    """If a previous run already produced ``out/<name>/results.json``, rebuild
    the leaderboard row from it instead of re-running. Returns ``None`` when
    resume is off or there's no prior file (caller runs the benchmark).
    """
    if not resume:
        return None
    rj = out / entry.name / "results.json"
    if not rj.exists():
        return None
    try:
        data = json.loads(rj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    agg = data.get("aggregate")
    if not agg:
        return None
    print(f"  ↺ resumed from {rj}")
    return _row_dict(entry.name, entry.profile.model, agg, None)


def _process_entry(entry: CompareEntry, data: Path, out: Path,
                   seed: int, limit: int | None, mode: str,
                   local: set[str], pulled_by_us: set[str],
                   resume: bool) -> dict:
    """Build one leaderboard row: resume, skip, or actually run."""
    resumed = _maybe_resume_row(entry, out, resume)
    if resumed is not None:
        return resumed
    tag = entry.profile.model
    if tag not in local and mode == "skip":
        return _row_dict(entry.name, tag, None,
                         f"model {tag!r} not pulled (mode=skip)")
    if mode == "circular":
        _ensure_pulled(tag, local, pulled_by_us)
    agg, err = _safe_run(entry, data, out / entry.name, seed, limit)
    return _row_dict(entry.name, tag, agg, err)


def _run_entries(entries: list[CompareEntry], data: Path, out: Path,
                 seed: int, limit: int | None, mode: str,
                 local: set[str], resume: bool = False) -> list[dict]:
    """Run each entry under the chosen pull mode, building leaderboard rows.

    The leaderboard is written after every row so a crash mid-loop still
    leaves a usable partial report on disk.
    """
    rows: list[dict] = []
    pulled_by_us: set[str] = set()
    if mode == "prefetch":
        for tag in {e.profile.model for e in entries if e.profile.model not in local}:
            _ensure_pulled(tag, local, pulled_by_us)
    for i, entry in enumerate(entries, 1):
        _print_progress(i, len(entries), entry)
        row = _process_entry(entry, data, out, seed, limit, mode,
                             local, pulled_by_us, resume)
        rows.append(row)
        write_leaderboard(rows, out)
        still_needed = {e.profile.model for e in entries[i:]}
        _maybe_drop(entry.profile.model, mode, pulled_by_us,
                    still_needed, local)
    return rows


def _merge_run(run: RunConfig, *, data: Path | None, out: Path | None,
               seed: int | None, limit: int | None,
               pull: str | None) -> tuple[Path, Path, int, int | None, str | None]:
    """CLI values override TOML ``[run]`` values; missing -> hard default.

    ``data`` is required: if neither side provides it we raise ``SystemExit``
    rather than fall back to a sentinel.
    """
    data_v = data or run.data
    if data_v is None:
        raise SystemExit(
            "no dataset specified: pass --data or set [run].data in the TOML")
    out_v = out or run.out or Path("reports/comparison")
    seed_v = seed if seed is not None else (run.seed if run.seed is not None else 0)
    limit_v = limit if limit is not None else run.limit
    pull_v = pull or run.pull
    return data_v, out_v, seed_v, limit_v, pull_v


def _finalize_and_report(rows: list[dict], out: Path, started: float) -> None:
    write_leaderboard(rows, out)
    elapsed = time.perf_counter() - started
    print(f"\n  comparison done in {elapsed:.1f} s")
    print(f"  leaderboard -> {out / 'leaderboard.html'}")
    print(f"               {out / 'leaderboard.md'}\n")
    print(_render_md(rows))


def run_comparison(config_path: Path,
                   data: Path | None = None, out: Path | None = None,
                   seed: int | None = None, limit: int | None = None,
                   pull: str | None = None, assume_yes: bool = False,
                   dry_run: bool = False, resume: bool = False) -> None:
    run_cfg, entries = load_comparison_config(config_path)
    data, out, seed, limit, pull = _merge_run(
        run_cfg, data=data, out=out, seed=seed, limit=limit, pull=pull)
    statuses = inspect_preflight(entries)
    missing = [s.tag for s in statuses if not s.local]
    print(format_preflight(statuses, mode=pull))
    if dry_run:
        return
    mode = resolve_pull_mode(missing, pull, assume_yes)
    if mode in ("prefetch", "circular") and missing and not assume_yes:
        if not _confirm_yn(f"\nThis will pull {len(missing)} model(s). Proceed?"):
            raise SystemExit("aborted by user")
    out.mkdir(parents=True, exist_ok=True)
    local = {s.tag for s in statuses if s.local}
    started = time.perf_counter()
    rows = _run_entries(entries, data, out, seed, limit, mode, local,
                        resume=resume)
    _finalize_and_report(rows, out, started)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="synthetic dataset dir (images/ + labels.jsonl); "
                         "overrides [run].data in the TOML")
    ap.add_argument("--config", type=Path,
                    default=Path("config/vlm_comparison.toml"),
                    help="comparison config TOML")
    ap.add_argument("--out", type=Path, default=None,
                    help="parent dir; each model gets a subdir + a leaderboard.* "
                         "is written here (overrides [run].out)")
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides [run].seed (default: 0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap each model to the first N images (overrides "
                         "[run].limit)")
    ap.add_argument("--pull", choices=_MODE_CHOICES, default=None,
                    help="how to handle missing models: prefetch (pull all up "
                         "front, keep), circular (pull -> run -> remove per "
                         "model), or skip (leave them as error rows). "
                         "If omitted and any model is missing, you'll be "
                         "prompted interactively.")
    ap.add_argument("--yes", "-y", dest="assume_yes", action="store_true",
                    help="skip the 'this will download N model(s)' prompt; "
                         "requires --pull when models are missing")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the preflight summary and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip any model whose <out>/<name>/results.json "
                         "already exists; useful after a crash")
    args = ap.parse_args()
    run_comparison(args.config, args.data, args.out,
                   seed=args.seed, limit=args.limit,
                   pull=args.pull, assume_yes=args.assume_yes,
                   dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
