"""Scoring for typed benchmark truth."""

from __future__ import annotations

from sci_fi_parser.benchmark import benchmark
from sci_fi_parser.benchmark.truth import ChartTruth
from sci_fi_parser.schema import ImageSet
from sci_fi_parser.vlm.vlm_schema import ChartData, parse_chartdata


def score_chart_values(
    image_id: str,
    truth: ChartTruth,
    pred: ChartData,
    seconds: float = 0.0,
    metadata: dict | None = None,
) -> benchmark.ChartResult:
    lo, hi = truth.data_range
    span = abs(hi - lo) or 1.0
    tmap = {(s.name, benchmark._cat_key(p.x)): float(p.y) for s in truth.series for p in s.points}
    pmap = benchmark._align_series_names(tmap, benchmark.series_map(pred))
    pnorm = {benchmark.normalize_key(s, c): k for k in pmap for s, c in [k]}

    errors, matched = [], 0
    for tkey, tv in tmap.items():
        pkey = pnorm.get(benchmark.normalize_key(*tkey))
        if pkey is not None:
            matched += 1
            errors.append(benchmark._pct_of_span(pmap[pkey], tv, span))
    matched_pkeys = {pnorm[benchmark.normalize_key(*k)] for k in tmap if benchmark.normalize_key(*k) in pnorm}
    extra = sum(1 for k in pmap if k not in matched_pkeys)
    errors_pos, misaligned, n_paired = benchmark._positional_score(tmap, pmap, span)
    result = benchmark.ChartResult(
        image_id,
        len(tmap),
        len(pmap),
        matched,
        len(tmap) - matched,
        extra,
        errors,
        value_errors_pos=errors_pos,
        misaligned=misaligned,
        n_paired_pos=n_paired,
        bar_count_err=abs(len(pmap) - len(tmap)),
        span=span,
        truth=tmap,
        pred=pmap,
        meta=metadata or {},
        seconds=seconds,
        type_true=truth.chart_type,
        type_pred=pred.chart_type,
        confidence=pred.confidence,
    )
    return result


def score_vlm_outputs(
    image_set: ImageSet,
    truth_by_image_id: dict[str, ChartTruth],
) -> list[benchmark.ChartResult]:
    results: list[benchmark.ChartResult] = []
    for image_id, record in image_set.items():
        truth = truth_by_image_id.get(image_id)
        if truth is None:
            continue
        try:
            pred = parse_chartdata(image_set.get_vlm_result(image_id))
        except Exception:  # pylint: disable=broad-exception-caught
            pred = ChartData(chart_type=None, series=[], confidence=None)
        seconds = float(record.get("metadata", {}).get("vlm", {}).get("seconds", 0.0))
        metadata = record.get("metadata", {}).get("benchmark", {})
        results.append(
            score_chart_values(
                image_set.get_image_path(image_id).name,
                truth,
                pred,
                seconds=seconds,
                metadata=metadata,
            )
        )
    return results


aggregate_value_results = benchmark.aggregate
