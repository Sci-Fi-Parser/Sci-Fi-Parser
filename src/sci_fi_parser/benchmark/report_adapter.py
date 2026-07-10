"""Adapters to the existing draw_lap report contract."""

from __future__ import annotations

from sci_fi_parser.benchmark import benchmark


def value_results_to_draw_lap_charts(results: list[benchmark.ChartResult]) -> list[dict]:
    charts = []
    for result in results:
        meta = result.meta or {}
        charts.append(
            {
                "image": result.image,
                "preset": str(meta.get("preset", meta.get("type", ""))),
                "density": meta.get("density", ""),
                "labels_on": meta.get("labels_on"),
                "type_true": result.type_true,
                "type_pred": result.type_pred,
                "type_matched": result.type_matched,
                "n_true": result.n_true,
                "matched": result.matched,
                "missed": result.missed,
                "extra": result.extra,
                "mean_pct": result.mean_pct,
                "max_pct": result.max_pct,
                "errors_pct": list(result.errors_pct),
                "span": result.span,
                "seconds": result.seconds,
                "truth": [[s, c, tv] for (s, c), tv in result.truth.items()],
                "pred": [[s, c, pv] for (s, c), pv in result.pred.items()],
            }
        )
    return charts


def value_results_to_breakdowns(results: list[benchmark.ChartResult]) -> list[dict]:
    return [
        {"title": "by preset", "rows": benchmark.group_summary(results, "preset")},
        {"title": "by density", "rows": benchmark.group_summary(results, "density")},
        {"title": "by labels-on", "rows": benchmark.group_summary(results, "labels_on")},
    ]
