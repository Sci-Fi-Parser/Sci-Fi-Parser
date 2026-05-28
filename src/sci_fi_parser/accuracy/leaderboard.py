"""Cross-model leaderboard rendering (markdown + HTML).

Split out of :mod:`sci_fi_parser.accuracy.vlm_compare` so the comparison runner
(config loading, daemon orchestration, CLI) stays free of presentation code.
Mirrors the benchmark / report split: one renderer per output artefact.

Inputs are plain dicts (``{"name", "tag", **aggregate}``) rather than typed
result objects -- a leaderboard row may come from a fresh ``aggregate()`` or
from a resumed ``results.json``, and both share the same dict shape.
"""

from __future__ import annotations

import html
import math
from pathlib import Path


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


def _int_or_dash(v) -> str:
    """Integer count, or '—' for None / NaN (e.g. older results.json)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return str(int(v))


# Column order leads with value-side metrics (the user's stage-of-project
# priority: did the heights come out right?) followed by the label-side
# diagnostic (`Misalign`, `Bar Δ`) and identity-side context (`Recall`,
# `Type acc`). Old rows whose results.json predates the value_* fields will
# render as "—" for those columns -- _pct/_pct100/_val/_int_or_dash all
# tolerate None / NaN.
_LEADER_COLS = [
    ("Model", "name", _identity),
    ("Tag", "tag", html.escape),
    ("Val mean", "value_mean_pct", _pct),
    ("Val max", "value_max_pct", _pct),
    ("Val median", "value_median_pct", _pct),
    ("Misalign", "misalignment_pct", _pct100),
    ("Bar Δ", "bar_count_err_total", _int_or_dash),
    ("Recall", "recall", _pct100),
    ("Type acc", "type_accuracy", _pct100),
    ("Mean conf", "mean_confidence", _val),
    ("Mean time", "mean_sec", _secs),
    ("Total time", "total_sec", _secs),
]


def row_dict(name: str, tag: str, agg: dict | None,
             error: str | None) -> dict:
    """Combine the per-model aggregate into the columns the renderer wants."""
    if error is not None:
        return {"name": name, "tag": tag, "error": error}
    return {"name": name, "tag": tag, **agg}


def render_md(rows: list[dict]) -> str:
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
    (out / "leaderboard.md").write_text(render_md(rows) + "\n", encoding="utf-8")
    (out / "leaderboard.html").write_text(_render_html(rows, out), encoding="utf-8")
