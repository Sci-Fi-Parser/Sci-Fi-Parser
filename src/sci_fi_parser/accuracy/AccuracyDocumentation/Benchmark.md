# Benchmark (gentle version)

> Senior version: [../Benchmark.md](../Benchmark.md).

The `benchmark` command is the **scorer**. It takes a synthetic dataset
plus an extractor, runs the extractor on every chart, compares what came
back to the ground truth, and writes a nice HTML report.

## Running it

```bash
benchmark --data train_data/synthetic --out reports/run1
```

By default, it uses the `noisy-oracle` extractor — a fake one that just
perturbs the truth by a few percent. Why? So you can **sanity-check the
whole pipeline** before plugging in a real model. If the noisy oracle
gives you a ~3% mean error, scoring + report + JSON are all wired up
correctly. If it gives you 0% or 100%, something's broken in the harness.

To use the real VLM:

```bash
benchmark --data train_data/synthetic --out reports/run1 --extractor ollama
```

This requires a local ollama server running and the model pulled
(`ollama pull qwen2.5vl:7b`).

### Command-line flags

| Flag | Default | Meaning |
|---|---|---|
| `--data <dir>` | required | The dataset directory (must contain `labels.jsonl` and `images/`) |
| `--out <dir>` | `reports/latest` | Where to write the report and results.json |
| `--extractor` | `noisy-oracle` | Which extractor to run: `noisy-oracle`, `ollama`, or `ollama:<model-tag>` |
| `--vlm-config <path>` | — | A specific VLM TOML to load (overrides `config/vlm.toml`) |
| `--seed` | `0` | RNG seed (only affects the noisy oracle) |
| `--limit` | — | Only run the first N images (handy for quick smoke tests) |

## The "extractor" concept

An extractor is any object that:

1. Has a `name` attribute (a string used in the report)
2. Has an `extract(image_path)` method that returns a `ChartData`

That's it. We have three built-in extractors:

| Name | Where | What it does |
|---|---|---|
| `noisy-oracle` | `benchmark.py:NoisyOracle` | Reads the truth, adds Gaussian noise, sometimes drops a bar, sometimes adds a phantom bar. Pretends to be an imperfect extractor. **For sanity checks only.** |
| `ollama` | `vlm/vlm.py:OllamaVLM` | The real one. Sends the chart image to a local ollama server, asks the model to extract values, returns the result. |
| `ollama:<tag>` | same | Same as `ollama` but with the model tag overridden one-shot. |

You can add your own — see [Extending.md](Extending.md).

## A worked example

Let's say there's one chart. The truth is:

```python
# From labels.jsonl
{
  "label1": "bar_chart",
  "label2": {
    "series": [{"name": "Revenue ($M)",
                "points": [["Q1", 100], ["Q2", 150], ["Q3", 0]]}],
    ...
  }
}
```

The extractor (the VLM) returns:

```python
ChartData(
    chart_type="bar_chart",
    confidence=0.9,
    series=[
        Series(name="series", points=[          # ← note: not "Revenue ($M)"
            Point(x="q1 ", y=110),              # close to truth (100), trailing space, lowercase
            Point(x="Q2",  y=140),              # close to truth (150)
            Point(x="Q4",  y=200),              # phantom! Truth has no Q4
        ])
    ]
)
```

What `score_chart` does, step by step:

1. **Builds the truth map**: `{("Revenue ($M)", "Q1"): 100, ("Revenue ($M)", "Q2"): 150, ("Revenue ($M)", "Q3"): 0}`

2. **Builds the prediction map**: `{("series", "q1 "): 110, ("series", "Q2"): 140, ("series", "Q4"): 200}`

3. **Single-series rename.** Both sides have exactly one series with
   different names. The scorer rekeys the prediction to use the truth's
   series name: `{("Revenue ($M)", "q1 "): 110, ...}`

   *Why?* Single-series charts often don't have a legend; the VLM has
   nothing to read off and falls back to `"series"` or the y-axis title.
   Demanding strict name match would throw away every correct value.

4. **Match keys (case-insensitive, trimmed).** `"Q1"` matches `"q1 "`
   because both normalise to `"q1"`. `"Q2"` matches `"Q2"` directly.
   `"Q3"` doesn't appear in the prediction → that's a **miss**.
   `"Q4"` doesn't appear in the truth → that's **extra** (hallucinated).

5. **Per-bar error.**
   - Q1: `|110 - 100| / 100 = 10%`
   - Q2: `|140 - 150| / 150 = 6.67%`
   - Q3: missed (no error percentage)
   - Q4: extra (no error percentage)

6. **Stats for this chart:**
   - `matched = 2`, `missed = 1`, `extra = 1`
   - `mean_pct = 8.33%`, `max_pct = 10%`
   - `n_true = 3`, `n_pred = 3`
   - `confidence = 0.9`

## What the report shows

The HTML report has these sections, top to bottom:

### 1. Summary cards (the strip at the top)

A row of stat boxes. Hover any one for its exact definition. The full
list:

| Card | What it means |
|---|---|
| **Charts** | How many charts were scored |
| **Mean error** | Average per-bar error across the whole run (% of true value) |
| **Median** | Middle per-bar error. Less sensitive to outliers |
| **p95** | 95th percentile. "95% of bars had error at most this much" |
| **Recall** | "Of the bars that should have been found, how many did the extractor return and match?" 100% = no misses |
| **Precision** | "Of the bars the extractor returned, how many matched a true bar?" 100% = no hallucinations |
| **≤1%** | Fraction of matched bars whose error is ≤ 1% |
| **≤5%** | Same, but ≤ 5% |
| **Type acc** | Fraction of charts where the extractor's chart_type matched truth |
| **Mean conf** | Average self-reported confidence (only for VLMs) |
| **Missed** | Total bars never returned across all charts |
| **Extra** | Total hallucinated bars across all charts |
| **Mean time** | Average seconds per chart |
| **p95 time** | 95% of charts finished within this many seconds |
| **Total time** | Sum of all per-chart times |

### 2. Breakdowns

Three small tables side by side: rows grouped by preset, by density, and
by labels-on/off. Each row shows the count, mean error, recall, and mean
time for that group.

This is how you see things like: "ah, accuracy collapses past density 20"
or "labels-on is 5% better than labels-off".

### 3. Worst / Best panes

The 6 worst and 6 best charts of the run (by a composite score — see
"How charts get ranked" below). Each one shows a thumbnail plus a small
table of (series, category, truth, predicted) values.

This is the most useful pane day-to-day. Look at the worst — does the VLM
have a systematic problem? (Confused by missing labels? Trouble with
small bars near zero?) Look at the best — what kind of charts is it
nailing?

### 4. All-charts table (sortable)

A full per-chart table. Click any column header to sort by it. Useful for
"which charts had the longest extraction time?" or "which charts had a
specific preset?"

## How charts get ranked (Worst / Best)

The composite score is:

```
score = mean_pct + max_pct + label_loss
```

where `label_loss = (missed + extra) / (n_true + extra) × 5.0`.

Higher score = worse.

Why this formula?

- **Mean alone** would wash out a single 50%-off bar in an otherwise good
  chart. We want to surface that bar.
- **Max alone** would miss the case where every bar is mediocre but none
  catastrophically wrong.
- Adding both catches both kinds of failure.
- The **label loss** ensures that an extractor that gets all values right
  but extras-out a bunch of phantom bars still drops in ranking.
- Charts with **zero matches** (recall = 0%) get score `+∞`, so total
  failures bubble to the very top of "Worst". That's where they belong.

## `results.json` (the machine-readable companion)

Same numbers as the report, but as JSON. Useful for CI ("don't merge if
mean error went up") and trend tracking ("plot mean error over time").

```jsonc
{
  "extractor": "qwen2.5vl:7b",
  "aggregate": { /* the summary card values */ },
  "by_preset":  [["simple", 10, 5.1, 0.98, 42.1], ...],
  "by_density": [[4, 6, 3.2, 1.0, 30.5], ...],
  "per_chart": [
    {"image": "simple_s0_d04_on_480.png",
     "meta": {...},
     "mean_pct": 4.2, "max_pct": 11.1,
     "matched": 4, "missed": 0, "extra": 0,
     "type_true": "bar_chart", "type_pred": "bar_chart", "type_matched": true,
     "confidence": 0.91}
  ]
}
```

Note: NaN (not-a-number) values get written as `null` because JavaScript /
JSON can't represent literal NaN.

## What happens when the extractor crashes

VLMs sometimes time out. Sometimes their JSON is malformed beyond repair.
What does the benchmark do?

The runner wraps `extractor.extract(...)` in a `try/except`. If it
crashes, you see in the terminal:

```
  ! simple_s0_d20_off_480.png: TimeoutError: ollama timed out
```

The prediction is substituted with an empty `ChartData(chart_type=None,
series=[], confidence=None)`, and the run continues. That chart gets:
`n_pred=0`, every true bar `missed`, no contribution to error stats.

Why this design? A single bad image shouldn't kill a 2-hour run.

## Fuzzy matching: how `normalize_key` works

When the scorer matches a predicted bar to a true bar, it doesn't require
exact strings. It does this:

```python
def normalize_key(series, category):
    return (series.strip().casefold(), category.strip().casefold())
```

Both sides go through this. So:
- `"Region A"` matches `"region a"`
- `"Region A"` matches `"Region A "` (trailing space)
- `"Region A"` matches `"REGION A"`

This is forgiving for case and whitespace, but **not** for spelling. The
VLM has to get `Region A` literally — `Region 1` would not match.

For integer-valued numeric categories, there's also a normalisation in
`_cat_key`: a JSON number `2018.0` is matched as `"2018"` to a string
truth label. Stops a silly type mismatch from being scored as a miss.

## The "% of true value" choice

Error is computed as `|pred - true| / |true| × 100`. That gives the
researcher-friendly read: pred 400 vs true 300 = 33%.

Alternative formulas you might wonder about:
- **% of axis span** — looks artificially small. (300 vs 400 on a
  1000-scale chart would be 10%, but the bar is wrong by a third.)
- **Absolute units** — can't cross-compare charts at different scales.

What if the true value is 0? You can't divide by zero. We bound it: if
true=0 and pred=0, error is 0%. If true=0 and pred≠0, error is 100%.

## Common questions

- **"My mean error is 50%. Is that good?"** Depends on the chart type and
  what you're measuring against. The noisy oracle gives ~3% by default.
  A well-tuned VLM on labels-on charts can hit 5-15%. Labels-off is much
  harder.

- **"Why is precision 100% but recall 90%?"** The extractor got every bar
  it returned right (no hallucinations), but missed 10% of the true bars.

- **"What does p95 mean?"** "95% of values are at or below this number."
  It's a measure of the tail — outliers. If mean is 5% but p95 is 80%,
  most bars are fine but some are very wrong.

## Sources

- `src/sci_fi_parser/accuracy/benchmark.py` (the whole file)
- `src/sci_fi_parser/schema.py` (`Extractor`, `normalize_key`,
  `parse_chartdata`)
- `src/sci_fi_parser/vlm/vlm.py` (`OllamaVLM`)
