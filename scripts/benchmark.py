"""Accuracy benchmark for chart extractors against synthetic ground truth.

Pipeline role -- the MEASUREMENT LAYER
--------------------------------------
Synthetic charts (scripts/synthetic_bars.py) have values known *by construction*,
so for any extractor we can compute exact per-chart error. Aggregated, those
errors become the empirical **error margin** we attach to real-world extractions
(which have no ground truth) before they go to SQL.

One canonical schema, many extractors
-------------------------------------
Every extractor -- VLM, CV+OCR pipeline, chart-specialized model -- is adapted to
a single :class:`ChartData` schema, so comparisons are apples-to-apples. Add an
extractor by implementing ``Extractor.extract(image_path) -> ChartData``.

Note: pure OCR is *not* a standalone value extractor (it reads text, not data
points) -- benchmark it as part of a CV+OCR pipeline.

Usage
-----
    .venv/bin/python scripts/benchmark.py --data train_data/synthetic --out reports/run1
    .venv/bin/python scripts/benchmark.py --data /tmp/sweep --extractor noisy-oracle
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import html
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ValidationError


# --------------------------------------------------------------------------- #
# Canonical schema -- the standardization target every extractor must speak
# --------------------------------------------------------------------------- #
class Point(BaseModel):
    x: str | float          # category label (bars) or numeric x
    y: float


class Series(BaseModel):
    name: str = "series"
    points: list[Point]


class ChartData(BaseModel):
    """The one format. VLMs are constrained to emit it; OCR/CV are mapped into it."""

    series: list[Series]
    confidence: float | None = None

    def series_map(self) -> dict[tuple[str, str], float]:
        """Flatten to {(series_name, category): value} for matching."""
        out: dict[tuple[str, str], float] = {}
        for s in self.series:
            for p in s.points:
                out[(s.name, str(p.x))] = float(p.y)
        return out


def parse_chartdata(raw: str | dict) -> ChartData:
    """Validate (and lightly repair) extractor output into ChartData.

    Real VLM output is messy -- code fences, prose, trailing commas. This is the
    repair half of the standardization layer; constrained decoding is the other.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if "```" in text:                       # strip ```json ... ``` fences
            text = text.split("```")[1]
            text = text[4:] if text.lstrip().lower().startswith("json") else text
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        raw = json.loads(text)
    return ChartData.model_validate(raw)


def truth_to_map(series: list[dict]) -> dict[tuple[str, str], float]:
    """Ground-truth label2 series -> {(series_name, category): value}."""
    out: dict[tuple[str, str], float] = {}
    for s in series:
        for cat, val in s["points"]:
            out[(s["name"], str(cat))] = float(val)
    return out


# --------------------------------------------------------------------------- #
# Extractor interface (runtime-agnostic) + a test double + an Ollama skeleton
# --------------------------------------------------------------------------- #
class Extractor(Protocol):
    name: str

    def extract(self, image_path: Path) -> ChartData:
        ...


class NoisyOracle:
    """TEST double: returns true values perturbed by noise + occasional miss/extra.

    Lets us exercise the harness and report without a real model. Noise is scaled
    by the value-axis span, matching the %-of-span error metric. Not a real model.
    """

    def __init__(self, truth_by_image: dict, rng: np.random.Generator,
                 rel_noise: float = 0.03, miss_p: float = 0.05, extra_p: float = 0.06):
        self.name = "noisy-oracle"
        self._truth = truth_by_image
        self._rng = rng
        self._rel_noise = rel_noise
        self._miss_p = miss_p
        self._extra_p = extra_p

    def extract(self, image_path: Path) -> ChartData:
        entry = self._truth[image_path.name]
        lo, hi = entry["value_range"]
        span = abs(hi - lo) or 1.0
        out_series = []
        for s in entry["series"]:
            pts = []
            for cat, val in s["points"]:
                if self._rng.random() < self._miss_p:
                    continue  # simulate a missed bar
                noisy = float(val) + self._rng.normal(0, self._rel_noise * span)
                pts.append(Point(x=str(cat), y=round(noisy, 3)))
            if self._rng.random() < self._extra_p:  # simulate a hallucinated bar
                pts.append(Point(x="GHOST", y=round(self._rng.uniform(lo, hi), 3)))
            out_series.append(Series(name=s["name"], points=pts))
        conf = float(np.clip(self._rng.normal(0.9, 0.05), 0, 1))
        return ChartData(series=out_series, confidence=conf)


class OllamaVLM:
    """Skeleton: local VLM via Ollama with schema-ENFORCED JSON output.

    Ollama's ``format=`` accepts a JSON schema and guarantees the reply conforms
    to it -- standardization-at-source for the VLM path. Needs ``pip install
    ollama`` and the model pulled (``ollama pull <model>``).
    """

    PROMPT = ("Extract the data from this bar chart. Return ONLY JSON: every series "
              "with its name and a list of {\"x\": category, \"y\": value} points.")

    def __init__(self, model: str = "qwen2.5vl:7b"):
        self.name = model
        self._model = model

    def extract(self, image_path: Path) -> ChartData:
        import ollama  # local import: optional dependency
        resp = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": self.PROMPT, "images": [str(image_path)]}],
            format=ChartData.model_json_schema(),   # <- guarantees schema-valid JSON
        )
        return parse_chartdata(resp["message"]["content"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ChartResult:
    image: str
    n_true: int
    n_pred: int
    matched: int
    missed: int                       # true bars the extractor did not return
    extra: int                        # predicted bars with no matching (series,cat)
    errors_pct: list[float] = field(default_factory=list)  # per matched bar, % of value span
    truth: dict = field(default_factory=dict)
    pred: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def mean_pct(self) -> float:
        return mean(self.errors_pct) if self.errors_pct else float("nan")

    @property
    def max_pct(self) -> float:
        return max(self.errors_pct) if self.errors_pct else float("nan")


def score_chart(image: str, entry: dict, pred: ChartData) -> ChartResult:
    """Match predicted bars to true bars by (series, category); error = |Δ| as % of span."""
    lo, hi = entry["value_range"]
    span = abs(hi - lo) or 1.0
    tmap = truth_to_map(entry["series"])
    pmap = pred.series_map()

    errors, matched = [], 0
    for key, tv in tmap.items():
        if key in pmap:
            matched += 1
            errors.append(abs(pmap[key] - tv) / span * 100.0)
    extra = sum(1 for k in pmap if k not in tmap)
    return ChartResult(image, len(tmap), len(pmap), matched, len(tmap) - matched,
                       extra, errors, tmap, pmap, entry.get("meta", {}))


def aggregate(results: list[ChartResult]) -> dict:
    all_err = np.array([e for r in results for e in r.errors_pct], dtype=float)
    total_true = sum(r.n_true for r in results) or 1
    total_pred = sum(r.n_pred for r in results) or 1
    total_matched = sum(r.matched for r in results)
    return {
        "n_charts": len(results),
        "n_bars_true": int(total_true),
        "mean_pct": float(all_err.mean()) if all_err.size else float("nan"),
        "median_pct": float(np.median(all_err)) if all_err.size else float("nan"),
        "p95_pct": float(np.percentile(all_err, 95)) if all_err.size else float("nan"),
        "recall": total_matched / total_true,
        "precision": total_matched / total_pred,
        "missed_total": sum(r.missed for r in results),
        "extra_total": sum(r.extra for r in results),
        "within_1pct": float((all_err <= 1).mean()) if all_err.size else float("nan"),
        "within_5pct": float((all_err <= 5).mean()) if all_err.size else float("nan"),
    }


def group_summary(results: list[ChartResult], key: str) -> list[tuple]:
    """(group, n_charts, mean %err, recall) rows grouped by a meta key."""
    groups: dict = defaultdict(list)
    for r in results:
        groups[r.meta.get(key)].append(r)
    rows = []
    for k in sorted(groups, key=lambda x: (x is None, x)):
        rs = groups[k]
        errs = [e for r in rs for e in r.errors_pct]
        tt = sum(r.n_true for r in rs) or 1
        rows.append((k, len(rs), mean(errs) if errs else float("nan"),
                     sum(r.matched for r in rs) / tt))
    return rows


# --------------------------------------------------------------------------- #
# HTML report (dependency-free templating + base64 thumbnails)
# --------------------------------------------------------------------------- #
def _thumb_b64(path: Path, width: int = 260) -> str:
    from PIL import Image
    im = Image.open(path)
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pct(x: float) -> str:
    return "-" if x != x else f"{x:.2f}%"  # x!=x catches NaN


def _group_table(title: str, rows: list[tuple]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{n}</td>"
        f"<td>{_pct(err)}</td><td>{rec*100:.1f}%</td></tr>"
        for k, n, err, rec in rows
    )
    return (f"<table class='full' style='max-width:520px'><thead><tr><th>{title}</th>"
            f"<th>charts</th><th>mean err</th><th>recall</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _detail_card(r: ChartResult, img_dir: Path) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{html.escape(c)}</td><td>{tv:.2f}</td>"
        f"<td>{(f'{r.pred[(s, c)]:.2f}' if (s, c) in r.pred else '—')}</td></tr>"
        for (s, c), tv in r.truth.items()
    )
    return f"""
    <div class="card">
      <img src="data:image/png;base64,{_thumb_b64(img_dir / r.image)}"/>
      <div class="meta">
        <b>{html.escape(r.image)}</b> · {html.escape(str(r.meta.get('type', '')))}
        · L{r.meta.get('density_level', '?')} · labels={r.meta.get('labels_on')}<br/>
        mean {_pct(r.mean_pct)} · max {_pct(r.max_pct)} · missed {r.missed} · extra {r.extra}
        <table class="kv"><tr><th>series</th><th>cat</th><th>true</th><th>pred</th></tr>{rows}</table>
      </div>
    </div>"""


def write_html(path: Path, extractor: str, agg: dict, results: list[ChartResult],
               img_dir: Path, n_show: int = 6) -> None:
    ok = [r for r in results if r.errors_pct]
    best = sorted(ok, key=lambda r: r.mean_pct)[:n_show]
    worst = sorted(ok, key=lambda r: r.mean_pct, reverse=True)[:n_show]

    cards = [
        ("Charts", agg["n_charts"]), ("Mean error", _pct(agg["mean_pct"])),
        ("Median", _pct(agg["median_pct"])), ("p95", _pct(agg["p95_pct"])),
        ("Recall", f"{agg['recall']*100:.1f}%"), ("Precision", f"{agg['precision']*100:.1f}%"),
        ("≤1%", f"{agg['within_1pct']*100:.0f}%"), ("≤5%", f"{agg['within_5pct']*100:.0f}%"),
        ("Missed", agg["missed_total"]), ("Extra", agg["extra_total"]),
    ]
    summary = "".join(f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>'
                      for k, v in cards)

    table_rows = "".join(
        f"<tr><td>{html.escape(r.image)}</td><td>{html.escape(str(r.meta.get('type','')))}</td>"
        f"<td>{r.meta.get('density_level','')}</td><td>{r.meta.get('labels_on')}</td>"
        f"<td>{r.n_true}</td><td>{r.matched}</td><td>{r.missed}</td><td>{r.extra}</td>"
        f'<td data-v="{0 if r.mean_pct != r.mean_pct else r.mean_pct}">{_pct(r.mean_pct)}</td>'
        f'<td data-v="{0 if r.max_pct != r.max_pct else r.max_pct}">{_pct(r.max_pct)}</td></tr>'
        for r in results
    )
    by_type = _group_table("by type", group_summary(results, "type"))
    by_density = _group_table("by density level", group_summary(results, "density_level"))
    by_labels = _group_table("by labels-on", group_summary(results, "labels_on"))

    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Extractor benchmark · {html.escape(extractor)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1a1a1a}}
 h1{{font-size:20px}} h2{{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}}
 .stats{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}
 .stat{{background:#f4f6f8;border-radius:10px;padding:12px 16px;min-width:90px;text-align:center}}
 .stat .v{{font-size:22px;font-weight:700}} .stat .k{{font-size:12px;color:#666}}
 .tables{{display:flex;flex-wrap:wrap;gap:24px}}
 .grid{{display:flex;flex-wrap:wrap;gap:14px}}
 .card{{border:1px solid #e3e3e3;border-radius:10px;padding:10px;width:320px}}
 .card img{{width:100%;border-radius:6px}} .meta{{font-size:12px;margin-top:6px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 .kv td,.kv th{{border-bottom:1px solid #eee;padding:2px 6px;text-align:right}}
 .kv td:first-child,.kv th:first-child{{text-align:left}}
 table.full th,table.full td{{border-bottom:1px solid #eee;padding:6px 8px;text-align:right}}
 table.full th:first-child,table.full td:first-child{{text-align:left}}
 table.full th{{cursor:pointer;background:#f4f6f8;position:sticky;top:0}}
</style>
<h1>Extractor benchmark — <code>{html.escape(extractor)}</code></h1>
<div class="stats">{summary}</div>
<p>Error = |predicted − true| as a percentage of the chart's value-axis span (scale-free).
Recall = bars found / true bars. Precision = correct (series,category) bars / predicted.</p>

<h2>Breakdowns</h2>
<div class="tables">{by_type}{by_density}{by_labels}</div>

<h2>Worst {len(worst)}</h2><div class="grid">{''.join(_detail_card(r, img_dir) for r in worst)}</div>
<h2>Best {len(best)}</h2><div class="grid">{''.join(_detail_card(r, img_dir) for r in best)}</div>

<h2>All charts <small>(click a header to sort)</small></h2>
<table class="full" id="t"><thead><tr>
 <th>image</th><th>type</th><th>L</th><th>labels</th><th>true</th><th>matched</th>
 <th>missed</th><th>extra</th><th>mean err</th><th>max err</th></tr></thead>
 <tbody>{table_rows}</tbody></table>
<script>
document.querySelectorAll('#t th').forEach((h,i)=>h.onclick=()=>{{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows];
 const num=i>=4, dir=h.dataset.d=h.dataset.d==='1'?'':'1';
 rows.sort((a,b)=>{{const x=a.cells[i],y=b.cells[i];
   const va=num?+(x.dataset.v??x.textContent.replace(/[^0-9.\\-]/g,'')||0):x.textContent;
   const vb=num?+(y.dataset.v??y.textContent.replace(/[^0-9.\\-]/g,'')||0):y.textContent;
   return (va>vb?1:va<vb?-1:0)*(dir?-1:1);}});
 rows.forEach(r=>tb.appendChild(r));}});
</script>
""", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def load_truth(data_dir: Path) -> dict:
    """image name -> {series, value_range, meta} from labels.jsonl."""
    truth = {}
    with (data_dir / "labels.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            l2 = rec["label2"]
            truth[rec["image"]] = {
                "series": l2["series"],
                "value_range": l2["value_range"],
                "meta": rec.get("meta", {}),
            }
    return truth


def build_extractor(name: str, truth: dict, rng: np.random.Generator) -> Extractor:
    if name == "noisy-oracle":
        return NoisyOracle(truth, rng)
    if name.startswith("ollama:"):
        return OllamaVLM(name.split(":", 1)[1])
    raise SystemExit(f"unknown extractor {name!r} (try: noisy-oracle, ollama:<model>)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="synthetic dir (images/ + labels.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("reports/latest"))
    ap.add_argument("--extractor", default="noisy-oracle",
                    help="noisy-oracle | ollama:<model> (extend build_extractor for more)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N charts")
    args = ap.parse_args()

    truth = load_truth(args.data)
    images = sorted(truth)[: args.limit] if args.limit else sorted(truth)
    rng = np.random.default_rng(args.seed)
    extractor = build_extractor(args.extractor, truth, rng)
    img_dir = args.data / "images"

    results: list[ChartResult] = []
    for name in images:
        entry = truth[name]
        try:
            pred = extractor.extract(img_dir / name)
        except (ValidationError, json.JSONDecodeError, KeyError, Exception) as exc:  # noqa: BLE001
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
            pred = ChartData(series=[])
        results.append(score_chart(name, entry, pred))

    agg = aggregate(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({
        "extractor": extractor.name, "aggregate": agg,
        "by_type": group_summary(results, "type"),
        "by_density": group_summary(results, "density_level"),
        "per_chart": [{"image": r.image, "meta": r.meta, "mean_pct": r.mean_pct,
                       "max_pct": r.max_pct, "matched": r.matched,
                       "missed": r.missed, "extra": r.extra} for r in results],
    }, indent=2, default=lambda o: None if isinstance(o, float) and o != o else o),
        encoding="utf-8")
    write_html(args.out / "report.html", extractor.name, agg, results, img_dir)

    print(f"\n  extractor : {extractor.name}")
    print(f"  charts    : {agg['n_charts']}  ({agg['n_bars_true']} bars)")
    print(f"  mean/med/p95 error : {_pct(agg['mean_pct'])} / {_pct(agg['median_pct'])} / {_pct(agg['p95_pct'])}")
    print(f"  recall/precision   : {agg['recall']*100:.1f}% / {agg['precision']*100:.1f}%")
    print(f"  within 1% / 5%     : {agg['within_1pct']*100:.0f}% / {agg['within_5pct']*100:.0f}%")
    print(f"  missed/extra bars  : {agg['missed_total']} / {agg['extra_total']}")
    print("\n  mean error by type:")
    for k, n, err, rec in group_summary(results, "type"):
        print(f"    {str(k):22s} {_pct(err):>8}  (recall {rec*100:.0f}%, {n} charts)")
    print(f"\n  report -> {args.out / 'report.html'}\n  json   -> {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
