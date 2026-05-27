# Decisions (gentle version)

> Senior version: [../Decisions.md](../Decisions.md).

This file explains **why** the code is shaped the way it is. Each decision
is one section. If you read code and think "why didn't they just do X?",
the answer is probably here.

The format for each decision: what was decided, why, where in the code,
and what we gave up to get it.

These aren't immutable. If we ever change one, the entry gets updated, not
deleted — the history is the whole point.

---

## 1. Every extractor returns the same `ChartData` shape

**Decided:** Whatever the extractor does internally, its output is a
pydantic `ChartData` (with `chart_type`, `series`, `confidence`).

**Why:** So we can compare them apples-to-apples. If one extractor
returned a tuple of arrays and another returned a dict of lists, the
scorer would need a custom case for each. With one shape, the scorer
only knows about `ChartData`.

**Where:** `schema.py` (definition); `accuracy/vlm.py:OllamaVLM`,
`accuracy/benchmark.py:NoisyOracle` (the existing implementations).

**Trade-off:** New extractors might need to do translation work at their
boundary. For example, an OCR pipeline that natively returns "list of
text strings" has to be mapped to series/points. That's unavoidable
once you've chosen a shared shape — it's just a question of where the
mapping lives. We chose: at the extractor's boundary, not in the scorer.

---

## 2. `chart_type` can only be from a fixed list

**Decided:** `ChartType = Literal["bar_chart", "grouped_bar_chart",
"stacked_bar_chart", "horizontal_bar_chart", "line_chart"]`.

**Why:** Ollama has a feature called "constrained decoding" — you can
give it a JSON schema and the model is **forced** to emit values that
match. A Literal becomes an `enum` in the schema, meaning the VLM
literally cannot output `"pie chart"` or `"bar_chart "` (extra space).
That kills a whole class of silent typo bugs.

**Where:** `schema.py:30`.

**Trade-off:** Adding a new chart type is a code change, not a config
change. We accept that — chart types are a small closed set in
practice. The test
`tests/test_smoke.py::test_chart_type_literal_constrained` makes sure
pydantic rejects invalid values.

---

## 3. `chart_type` and `confidence` are required, but can be `None`

**Decided:** Both fields are typed `... | None` (not `... = None`). In
pydantic, that means they're "required but nullable".

**Why:** With constrained decoding, the model looks at which fields are
required and decides whether to emit them. If they aren't required, the
model **silently omits them** when it isn't sure — and we get no signal
that it was uncertain. By making them required, we force an explicit
`null` when the model can't decide. That's much more useful.

**Where:** `schema.py:65`.

**Trade-off:** None we've noticed.

---

## 4. `schema.py` lives at the package root, not under `accuracy/`

**Decided:** The shared `ChartData` shape lives at
`src/sci_fi_parser/schema.py`, **not** under `accuracy/`. It imports
only `json`, `pathlib`, `typing`, `pydantic`. Nothing heavy.

**Why:** Future code outside the accuracy layer needs `ChartData` — the
main pipeline does, and so will a future CV+OCR extractor. If
`ChartData` lived under `accuracy/`, importing it would tend to pull in
sibling modules (matplotlib, opencv). That's a ~1 second startup hit
every time. Keeping `schema.py` separate and small avoids it.

**Where:** `schema.py`. Enforced by
`tests/test_smoke.py::test_schema_is_light`, which actually runs Python
in a subprocess and checks that `matplotlib`, `cv2`, `ollama` did not
get loaded.

**Trade-off:** Slightly weird directory layout — `schema.py` looks like
a "root concern" file. It is. Worth it.

---

## 5. `accuracy/__init__.py` is empty (almost)

**Decided:** The `__init__.py` for the accuracy package doesn't import
any submodules.

**Why:** Same as decision #4. If `__init__` did
`from . import synthetic`, then *any* code that touches the accuracy
package would pull in matplotlib. We want
`import sci_fi_parser.accuracy.benchmark` to stay cheap.

**Where:** `accuracy/__init__.py`. Enforced by
`tests/test_smoke.py::test_accuracy_init_does_not_pull_matplotlib`.

**Trade-off:** Callers have to write
`import sci_fi_parser.accuracy.X` rather than
`from sci_fi_parser.accuracy import X`. Fine — the docstring tells
them so.

---

## 6. VLM extractor is in its own file

**Decided:** `OllamaVLM` was extracted from `benchmark.py` into
`accuracy/vlm.py`.

**Why:** If you only want the extractor (because you're writing
production pipeline code that *uses* the VLM, not because you're
benchmarking), you shouldn't have to import the benchmark machinery
(HTML report, ranking, etc.). Now: `from sci_fi_parser.accuracy.vlm
import OllamaVLM`.

**Where:** Commit `e742552`.

**Trade-off:** One more file. Negligible.

---

## 7. VLM settings come from a TOML file, not Python

**Decided:** Model name, prompt, ollama options live in
`config/vlm.toml`. There's a precedence: CLI flag > `--vlm-config` >
`config/vlm.toml` > built-in default. Environment variables can
override individual fields.

**Why:** Researchers swap models and prompts constantly. Editing
Python invites diffs, merge conflicts, and "don't commit my local
prompt" mishaps. TOML is just data — easy to edit, easy to swap,
easy to share. The CLI override covers "try one model once"; env
vars cover "let me bump context just this once".

**Where:** `accuracy/vlm_config.py`, `accuracy/vlm.py`,
`accuracy/benchmark.py:_resolve_profile`. Config at `config/vlm.toml`.

**Trade-off:** Four possible sources of truth sounds scary, but the
precedence is documented in [Configuration.md](Configuration.md) and
the function `_resolve_profile` resolves to one final profile cleanly.

---

## 8. We dropped the `docling` dependency

**Decided:** `docling` (a PDF parsing library) was removed from
`pyproject.toml`. `uv.lock` shrank by ~1.7k lines.

**Why:** It was carried from an earlier exploration and was no longer
in the call graph. Smaller lockfile = faster `uv sync`, fewer
transitive dependency CVEs to chase.

**Where:** Commit `2b32d77`.

**Trade-off:** If we want it back later, it's a one-line add. No data
lost.

---

## 9. The synthetic generator's default mode is a "density series"

**Decided:** When you run `synthetic-bars` with no special flags, it
samples **one** style per chart type and walks through `density_steps`
holding that style fixed.

**Why:** It's a **controlled experiment**. We want to know "how does
accuracy degrade as charts get more crowded?". If we randomised
everything (style **and** density), we couldn't separate the density
effect from style noise. By holding style constant and varying only
density, the effect we're measuring is isolated.

**Where:** `accuracy/synthetic/generate.py:generate_series`.

**Trade-off:** A density series concentrates lots of charts on one
style. If that style is weirdly pathological, the report can
over-state degradation specific to it. Mitigated by `per_type > 1`,
which generates multiple independent styles.

---

## 10. Every density step is drawn twice (labels off + labels on)

**Decided:** For every density step in a series, we render **two**
charts with the same data: one with value labels printed on the bars,
one without.

**Why:** Lets us measure "with help" vs "without help" accuracy on
identical content. VLMs do dramatically better when values are
written on bars. Quantifying that gap tells us how much of the score
is reading vs. inferring from axis scale.

**Where:** `accuracy/synthetic/generate.py:_emit_density`. The
benchmark report breaks down by `meta.labels_on`.

**Trade-off:** Doubles image count. Acceptable.

---

## 11. Synthetic labels come from us, not from a model

**Decided:** When we draw a chart, the "right answer" is the values we
passed to matplotlib. We do **not** run an LLM to read the chart we
just drew.

**Why:** The existing real-data labels (in
`train_data/dataset.sqlite3`) were produced by a Gemma → Haiku → Opus
pipeline. If we benchmarked a Gemma-family model against those
labels, we'd partly be measuring "does this model agree with prior
models?" rather than "is it correct?". Synthetic data, by
construction, has zero of that circularity.

**Where:** Whole `accuracy/synthetic/` package; rationale in
`synthetic/__init__.py` docstring.

**Trade-off:** Synthetic charts lack the long tail of real-world
weirdness (handwritten annotations, axis breaks, broken JPEGs). We'll
need real data again once an extractor is dialled in. Synthetic and
real can coexist.

---

## 12. `cv2` is imported only when needed

**Decided:** Inside `accuracy/synthetic/output.py`, the `import cv2`
statements live **inside** the functions that use them, not at the
top of the file.

**Why:** Plain chart generation (no augmentation, no overlay) doesn't
need opencv. We declare opencv as a `pip install '.[cv]'` extra. By
lazy-importing it, users who don't ask for `--augment` or `--overlay`
can install just the core deps.

**Where:** `accuracy/synthetic/output.py:augment`,
`:make_overlay`, `:write_overlay`. Extras declared in
`pyproject.toml`.

**Trade-off:** Lazy imports are slightly slower on first call. We do
this once per chart at most — irrelevant.

---

## 13. `value_range` gets clipped to start at zero

**Decided:** When the benchmark loads `labels.jsonl`, it clips the
lower bound of `value_range` to 0 if it was negative.

**Why:** matplotlib's y-axis on positive-only charts ends up with a
small pad below zero (`ylim = (-0.02 × hi, 1.05 × hi)`). The noisy
oracle scales its noise by the axis span. Without the clip, the
span is artificially larger, the noise is proportionally smaller,
and the oracle looks too good.

**Where:** `accuracy/benchmark.py:load_truth`.

**Trade-off:** If we ever generate charts where negative values are
legitimate (rare today), this clip would flatten them. Worth a
comment if/when it matters.

---

## 14. Single-series charts get a series-name "free pass"

**Decided:** If both the truth and the prediction have **exactly one**
series, the scorer rekeys the prediction's series to use the truth's
series name (so values can still be matched).

**Why:** Single-series charts often have no legend — there's nothing
for the VLM to read as a series name. It'll fall back to `"series"`
or the y-axis title. Demanding strict series-name match would mark
every value as a miss, even though they were all correct.

**Where:** `accuracy/benchmark.py:_align_series_names`.

**Trade-off:** Multi-series alignment stays strict. So if the VLM
mislabels series 0 vs series 1 in a grouped chart, those values are
scored as misses, not partial matches. That's intentional — when
series are real and distinct, order/name matter, and a "smart"
alignment heuristic would couple matching to the metric.

---

## 15. Error is `% of the true value`

**Decided:** Per-bar error is `|pred - true| / |true| × 100`.

**Why:** That's the natural read for a researcher: "pred 400 vs true
300 is a 33% error". Alternative formulas are worse:
- `% of axis span` — artificially small (300 vs 400 on a 1000-scale
  chart would be 10%, but the bar is wrong by a third).
- absolute units — can't cross-compare charts.

**Where:** `accuracy/benchmark.py:_pct_of_true`. Bounded at 100% when
the truth is 0 (so we don't divide by zero).

**Trade-off:** Tiny non-zero truths blow up. (true=0.01, pred=0.02 →
100%.) Synthetic data uses sensible scales; for real data, revisit.

---

## 16. Best/Worst ranking uses `mean + max + label_loss`

**Decided:** The "Worst" pane in the report sorts charts by:
```
mean_pct + max_pct + (missed + extra) / (n_true + extra) × 5.0
```
Higher is worse. Charts with zero matches get `+infinity`.

**Why:**
- **Mean alone** washes out single bad bars.
- **Max alone** misses chronically mediocre charts.
- Summing both surfaces both failure modes.
- The label loss makes hallucinations cost something.
- `+infinity` for total failures forces them to the top of "Worst",
  which is where they belong.

**Where:** `accuracy/benchmark.py:_rank_score`.

**Trade-off:** The `× 5.0` is judgement. Tweak if it ever feels wrong.

---

## 17. The HTML report has no dependencies

**Decided:** `report.html` is a single self-contained file with
inline CSS and JavaScript and base64-embedded thumbnails. No flask,
no jinja, no separate `assets/` folder.

**Why:** The report gets emailed, dropped on shared drives, opened
on phones. Anything with assets-on-disk breaks the first time
someone moves the file. Self-contained > tidy.

**Where:** `accuracy/benchmark.py:_CSS`, `:_SORT_JS`, `:write_html`,
`:_thumb_b64`.

**Trade-off:** File size grows with thumbnail count. Default is 6
worst + 6 best, so it's small (a few MB). For bigger panes we'd
need to cache.

---

## 18. `reports/` is gitignored

**Decided:** Benchmark outputs (`reports/**`) are not tracked in
git.

**Why:** It's a derived artifact. Tracking it bloats the repo and
creates merge conflicts every run.

**Where:** Commit `2be68c8`; `.gitignore`.

**Trade-off:** Old reports must be archived elsewhere if you want
to diff them historically. CI can attach them as build artifacts.

---

## How to read this file going forward

- If you're about to refactor something, scan the relevant sections
  first. Some changes look like cleanup but actually break a
  constraint that's hard to see.
- If you disagree with a decision, that's fine — but **update the
  entry** when you change the code. Don't delete it. The history is
  part of the doc.
- If a new design choice gets made and is non-obvious, add an entry.
  Future-you and your teammates will thank you.

## Sources

- `src/sci_fi_parser/schema.py`, `accuracy/{vlm,vlm_config,benchmark}.py`,
  `accuracy/synthetic/*`
- `config/vlm.toml`, `config/synthetic_bars.toml`
- `tests/test_smoke.py`
- Commits `c3c71f0`, `e742552`, `fd68746`, `2b32d77`, `2be68c8`
