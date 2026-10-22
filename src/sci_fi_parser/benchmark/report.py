"""Self-contained HTML report for an end-to-end benchmark run."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to a buffer, never a window
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from PIL import Image  # noqa: E402

from sci_fi_parser.benchmark.scoring import ChartScore  # noqa: E402

_CSS = """
 body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1a1a1a}
 h1{font-size:20px}
 h2{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
 .stats{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
 .stat{background:#f4f6f8;border-radius:10px;padding:12px 16px;min-width:90px;text-align:center}
 .stat .v{font-size:22px;font-weight:700} .stat .k{font-size:12px;color:#666}
 .tables{display:flex;flex-wrap:wrap;gap:24px}
 .grid{display:flex;flex-wrap:wrap;gap:14px}
 .card{border:1px solid #e3e3e3;border-radius:10px;padding:10px;width:340px}
 .card img{width:100%;border-radius:6px} .meta{font-size:12px;margin-top:6px}
 .hist{max-width:880px;width:100%;border:1px solid #eee;border-radius:8px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 td,th{border-bottom:1px solid #eee;padding:4px 8px;text-align:right}
 td:first-child,th:first-child{text-align:left}
 #t th{cursor:pointer;background:#f4f6f8;position:sticky;top:0}
"""

_SORT_JS = r"""
document.querySelectorAll('#t th').forEach((h,i)=>h.onclick=()=>{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows];
 const num=h.dataset.num==='1', dir=h.dataset.d=h.dataset.d==='1'?'':'1';
 const val=c=>num?+(c.dataset.v??0):c.textContent;
 rows.sort((a,b)=>{const x=val(a.cells[i]),y=val(b.cells[i]);
   return (x>y?1:x<y?-1:0)*(dir?-1:1);});
 rows.forEach(r=>tb.appendChild(r));});
"""


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}%"


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _thumbnail(path: Path, width: int = 360) -> str:
    try:
        image = Image.open(path)
    except OSError:
        return "<div>image missing</div>"
    image.thumbnail((width, width))
    return f'<img src="data:image/png;base64,{_png_b64(image)}">'


def _histogram(scores: list[ChartScore]) -> str:
    """Signed deviation (pred - true) / data range of every matched point, clamped to ±100 %."""

    deviations = [
        max(-100.0, min(100.0, (p.pred - p.true) / s.span * 100.0))
        for s in scores
        for p in s.points
        if p.pred is not None
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=110)
    ax.hist(deviations, bins=40, range=(-100, 100), color="#6366f1")
    ax.axvline(0, color="#94a3b8", lw=1, ls=":")
    ax.set_xlabel("deviation (pred − true) / data range, %")
    ax.set_ylabel("points")
    ax.set_title(f"n = {len(deviations)} matched points", fontsize=9, loc="right")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return f'<img class="hist" src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">'


def _summary_cards(summary: dict[str, Any]) -> str:
    cards = [
        ("Charts", summary["charts"]),
        ("No prediction", summary["no_prediction"]),
        ("Mean error", _pct(summary["mean_pct"])),
        ("Median", _pct(summary["median_pct"])),
        ("p95", _pct(summary["p95_pct"])),
        ("≤5%", _share(summary["within_5pct"])),
        ("Recall", _share(summary["recall"])),
        ("Precision", _share(summary["precision"])),
        ("Missed", summary["missed"]),
        ("Extra", summary["extra"]),
        ("Type acc", _share(summary["type_accuracy"])),
    ]
    return "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in cards
    )


def _breakdown_table(key: str, groups: dict[str, dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(value)}</td><td>{s['charts']}</td><td>{_pct(s['mean_pct'])}</td>"
        f"<td>{_share(s['recall'])}</td><td>{_share(s['precision'])}</td></tr>"
        for value, s in groups.items()
    )
    return (
        f"<table style='max-width:520px'><tr><th>{html.escape(key)}</th><th>charts</th>"
        f"<th>mean err</th><th>recall</th><th>precision</th></tr>{rows}</table>"
    )


def _rank(score: ChartScore) -> float:
    """Higher is worse: value error plus a penalty for missed and extra points."""

    if score.mean_pct is None or score.max_pct is None:
        return float("inf")
    label_loss = (score.missed + score.extra) / ((len(score.points) + score.extra) or 1) * 5.0
    return score.mean_pct + score.max_pct + label_loss


def _card(score: ChartScore) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(p.series)}</td><td>{html.escape(p.x)}</td><td>{p.true:.2f}</td>"
        f"<td>{'—' if p.pred is None else f'{p.pred:.2f}'}</td></tr>"
        for p in score.points
    )
    return (
        f'<div class="card">{_thumbnail(score.image_path)}<div class="meta">'
        f"<b>{html.escape(score.image_path.name)}</b> · type {score.type_pred} (true {score.type_true})<br>"
        f"mean {_pct(score.mean_pct)} · max {_pct(score.max_pct)} · "
        f"missed {score.missed} · extra {score.extra}"
        f"<table><tr><th>series</th><th>x</th><th>true</th><th>pred</th></tr>{rows}</table></div></div>"
    )


def _table_row(score: ChartScore) -> str:
    def num(value: float | int | None, text: str) -> str:
        return f'<td data-v="{-1 if value is None else value}">{text}</td>'

    return (
        f"<tr><td>{html.escape(score.image_path.name)}</td><td>{score.type_true}</td>"
        f"<td>{score.type_pred}</td>{num(len(score.points), str(len(score.points)))}"
        f"{num(score.matched, str(score.matched))}{num(score.missed, str(score.missed))}"
        f"{num(score.extra, str(score.extra))}{num(score.mean_pct, _pct(score.mean_pct))}"
        f"{num(score.max_pct, _pct(score.max_pct))}</tr>"
    )


def write_report(
    path: Path,
    run: dict[str, Any],
    summary: dict[str, Any],
    groups: dict[str, dict[str, dict[str, Any]]],
    scores: list[ChartScore],
) -> None:
    """Write the HTML report for one run to ``path``."""

    model = html.escape(run["vlm"]["model"])
    breakdown_html = "".join(_breakdown_table(key, rows) for key, rows in groups.items())
    cards = "".join(_card(s) for s in sorted(scores, key=lambda s: (_rank(s), s.image_path.name)))
    rows = "".join(_table_row(s) for s in scores)
    path.write_text(
        f"""<!doctype html><meta charset="utf-8">
<title>Benchmark · {model}</title>
<style>{_CSS}</style>
<h1>Benchmark — <code>{model}</code></h1>
<p>{html.escape(run["dataset"])} data <code>{html.escape(run["data"])}</code> ·
classification {"on" if run["stages"]["classify"] else "from truth"} ·
OCR/CV {"on" if run["stages"]["ocr"] else "off"} · {html.escape(run["started_at"])}</p>
<div class="stats">{_summary_cards(summary)}</div>
<p>Error = |predicted − true| as a percentage of the truth data range, over matched points only.
Recall = matched / true points. Precision = matched / predicted points.</p>
{f'<h2>Breakdowns</h2><div class="tables">{breakdown_html}</div>' if groups else ""}
<h2>Deviation distribution</h2>
{_histogram(scores)}
<h2>Charts, best to worst</h2>
<div class="grid">{cards}</div>
<h2>All charts <small>(click a header to sort)</small></h2>
<table id="t"><thead><tr><th>image</th><th>type</th><th>predicted</th>
<th data-num="1">points</th><th data-num="1">matched</th><th data-num="1">missed</th>
<th data-num="1">extra</th><th data-num="1">mean err</th><th data-num="1">max err</th></tr></thead>
<tbody>{rows}</tbody></table>
<script>{_SORT_JS}</script>
""",
        encoding="utf-8",
    )
