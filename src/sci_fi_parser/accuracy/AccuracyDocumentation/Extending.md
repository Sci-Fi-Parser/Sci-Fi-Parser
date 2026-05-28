# Extending the accuracy layer (gentle version)

> Senior version: [../Extending.md](../Extending.md).

This guide walks through the most common extensions you might want to
add: a new extractor, a new chart type, a new metric, or a new VLM
backend.

## Before you change anything

Two rules the codebase enforces — don't break them by accident:

1. **`schema.py` must not import matplotlib, cv2, or ollama.** Why this
   matters is in [Decisions.md](Decisions.md) #4. The test that catches
   it is `tests/test_smoke.py::test_schema_is_light`.

2. **`accuracy/__init__.py` must not transitively pull matplotlib.**
   Same reason, different scope. Test:
   `test_accuracy_init_does_not_pull_matplotlib`.

If either test goes red after a change you made, the test is right and
your refactor is wrong. Read the matching Decision entry before trying
to "fix" the test.

## Adding a new extractor

This is the most common extension. Say you want to try out
PaddleOCR + a bar-detection model to read charts.

### The contract

The contract is a `Protocol` in `schema.py`:

```python
@runtime_checkable
class Extractor(Protocol):
    name: str                                           # ← a string label
    def extract(self, image_path: Path) -> ChartData:   # ← the workhorse
        ...
```

That's the whole contract. Have a `name` attribute and an `extract`
method. Python's `runtime_checkable` decorator means you can even do
`isinstance(your_extractor, Extractor)` and it just works.

### A worked example

Here's a stub CV+OCR extractor. You'd put this in a new file:

```python
# src/sci_fi_parser/accuracy/cv_ocr.py
"""CV+OCR extractor: detect bars with computer vision, read axis with OCR."""

from pathlib import Path
from sci_fi_parser.schema import ChartData, Series, Point


class CvOcrExtractor:
    name = "cv-ocr"                                  # ← shows up in reports

    def __init__(self, ocr_engine, bar_detector):
        self._ocr = ocr_engine                       # ← injected so tests can fake it
        self._det = bar_detector

    def extract(self, image_path: Path) -> ChartData:
        # 1. Detect bars in the image → list of bounding boxes
        bars = self._det.detect(image_path)
        # 2. Read y-axis ticks with OCR → {value: pixel_position}
        axis = self._ocr.read_ticks(image_path)
        # 3. Read x-axis category labels
        categories = self._ocr.read_categories(image_path)
        # 4. For each bar, calibrate its pixel height to a value
        points = [
            Point(x=cat, y=self._calibrate(box, axis))
            for cat, box in zip(categories, bars)
        ]
        # 5. Pack into the shared ChartData shape
        return ChartData(
            chart_type=None,                          # we don't classify the chart
            series=[Series(name="series", points=points)],
            confidence=None,                          # we don't self-rate
        )

    def _calibrate(self, box, axis):
        # ... map a bbox's top pixel through the axis tick map → numeric value
        ...
```

Three rules to follow:

1. **Use `None` for fields you can't fill.** `chart_type=None` and
   `confidence=None` are perfectly legal. The benchmark handles them —
   `type_accuracy` only counts charts where both sides have a value;
   `mean_confidence` only averages non-None values.

2. **Use `"series"` as the series name for single-series charts.**
   That's the canonical default. It triggers the
   "single-series free pass" in the scorer (see
   [Decisions.md](Decisions.md) #14).

3. **Don't reach into pydantic internals.** Build with the public
   constructors: `ChartData(...)`, `Series(...)`, `Point(...)`.
   Pydantic will validate types and raise if you got them wrong.

### Plug it into the runner

The benchmark runner has a small dispatch in
`accuracy/benchmark.py:build_extractor`:

```python
def build_extractor(name, truth, rng, profile=None):
    if name == "noisy-oracle":
        return NoisyOracle(truth, rng)
    if name == "ollama":
        return OllamaVLM(profile=profile)
    if name.startswith("ollama:"):
        return OllamaVLM(profile=profile, model_override=name.split(":", 1)[1])
    if name == "cv-ocr":                                       # ← your addition
        from sci_fi_parser.accuracy.cv_ocr import CvOcrExtractor  # lazy import!
        return CvOcrExtractor(
            ocr_engine=..., bar_detector=...,                  # wire these up
        )
    raise SystemExit(f"unknown extractor {name!r} ...")
```

Note the **lazy import**: `from sci_fi_parser.accuracy.cv_ocr import ...`
is inside the `if` branch, not at the top of the file. That way, when
someone picks the `noisy-oracle` or `ollama` extractor, they don't pay
the cost of importing paddleocr.

Also update the `--extractor` help string in `_parse_args` to mention
your new name.

### Run it

```bash
benchmark --data train_data/synthetic --out reports/cv-ocr-run \
          --extractor cv-ocr
```

If your extractor has its own settings (model paths, etc.), consider
following the `VLMProfile` pattern — a TOML config plus a `--cv-config`
flag that resolves to a path.

### Test it

Add a minimal import test to `tests/test_smoke.py`:

```python
def test_cv_ocr_imports():
    from sci_fi_parser.accuracy.cv_ocr import CvOcrExtractor
    assert callable(CvOcrExtractor)
```

If your extractor pulls heavy dependencies, declare them as a project
**extra** in `pyproject.toml` (like the existing `cv` extra), so people
who don't need it can skip installation.

## Adding a new chart type

If the new type is similar to existing ones (a new variant of bar or
line), add it to `CATALOG` in `synthetic/config.py`:

```python
CATALOG: dict[str, dict] = {
    ...,
    "wide-grouped": {                                 # ← new alias
        "chart_type": "grouped_bar_chart",            # ← from the ChartType literal
        "family": "bar",                              # ← which render path
        "orientation": "v",
        "stacked": False,
        "color": "series",
        "series": "multi",
    },
}
```

The fields:

| Field | What it controls |
|---|---|
| `chart_type` | The string returned as `label1` and to the extractor's `chart_type`. Must be one of the `ChartType` values in `schema.py`. |
| `family` | `"bar"` or `"line"` — picks the matplotlib render path. |
| `orientation` | `"v"` (vertical) or `"h"` (horizontal). Horizontal triggers `barh`. |
| `stacked` | `True` or `False`. Vertical bars only. |
| `color` | `"single"`, `"colormap"` (gradient across bars), or `"series"` (one per series). |
| `series` | An int, or the string `"multi"` (sample from `n_series` range). |

If the new chart type needs a **new family** (e.g. pie, scatter), you'll
need to:

1. Add the new literal to `ChartType` in `schema.py`. This is a code
   change because `ChartType` is a `Literal` (see Decisions #2).
2. Add a render function in `synthetic/render.py` (e.g.
   `_draw_scatter`) and wire it into `_draw`'s dispatch.
3. Update `_collect_geometry` if the new family has different geometry
   primitives (e.g. circles vs rectangles).

## Adding a new metric

A metric is "a number computed from the per-chart scoring results that
appears in the report".

Three places to edit (in `accuracy/benchmark.py`):

1. **Compute it.** In `aggregate(results)`, add a key to the returned
   dict:
   ```python
   def aggregate(results):
       ...
       return {
           ...,
           "my_metric": my_compute(results),
       }
   ```

2. **Surface it in the report.** Either in `_summary_cards` (a card
   in the top strip with a hover tooltip):
   ```python
   ("My metric", f"{agg['my_metric']:.2f}",
    "Plain-English description for the tooltip."),
   ```
   Or in `_group_table` if it's a per-group number.

3. **Surface it in `_print_summary`** (the stdout summary) if it's a
   headline number.

Per-chart vs aggregate metrics:

- **Per chart** (computed for each chart, then averaged): add a field
  to `ChartResult`, compute it in `score_chart`, then aggregate it in
  `aggregate`. If you want to expose it in the sortable all-charts
  table, also add it to `_chart_row`.
- **Aggregate only** (a single number from all charts): just compute
  it in `aggregate`.

**Don't skip the hover tooltip.** The `_summary_cards` cards take a
`description` string that becomes the `title=` tooltip. Researchers
hover. Write the formula in plain English.

## Adding a new VLM backend (not ollama)

Say you want to use the Anthropic API instead of ollama.

Mirror the shape of `OllamaVLM` in a new file:

```python
# src/sci_fi_parser/accuracy/anthropic_vlm.py
import os
from pathlib import Path
from sci_fi_parser.vlm.vlm_config import VLMProfile
from sci_fi_parser.schema import ChartData, parse_chartdata


class AnthropicVLM:
    def __init__(self, profile: VLMProfile | None = None,
                 model_override: str | None = None):
        profile = profile or VLMProfile()
        self.name = model_override or profile.model
        self._prompt = profile.prompt
        # ... whatever client init you need ...

    def extract(self, image_path: Path) -> ChartData:
        import anthropic   # ← lazy import, inside extract
        # ... send image + self._prompt to the API ...
        raw = response["content"]               # API-specific
        return parse_chartdata(raw)             # the schema's "messy input" handler
```

Three rules again:

1. **Lazy-import the client.** Putting `import anthropic` at module top
   would mean everyone who imports the `vlm` module pays the cost,
   even if they're using ollama. Lazy inside `extract` is the
   convention.

2. **Always run output through `parse_chartdata`.** Real model output
   is messy — code fences, leading prose, trailing whitespace.
   `parse_chartdata` strips fences, locates the outer `{...}`, and
   pydantic-validates. Use it.

3. **Reuse `VLMProfile`** for the config object. If your backend
   needs new fields (e.g. an API key env var name), add them to the
   dataclass in `vlm_config.py`. Don't create a parallel dataclass —
   the comparison runner (when it lands) will expect one shape.

Then register it in `build_extractor` the same way ollama is:

```python
if name == "anthropic":
    from sci_fi_parser.accuracy.anthropic_vlm import AnthropicVLM
    return AnthropicVLM(profile=profile)
if name.startswith("anthropic:"):
    return AnthropicVLM(profile=profile, model_override=name.split(":", 1)[1])
```

## Common pitfalls

Things that will burn you if you don't watch for them:

- **Eager-importing your new extractor's heavy deps in
  `accuracy/__init__.py`.** Breaks the matplotlib-light invariant.
  Test fails. Don't do this.

- **Subclassing `ChartData` to add extractor-specific extras.** If
  you do, the report won't know about your extras (it only knows
  `ChartData`). Either argue for the extras in
  [Decisions.md](Decisions.md) and add them to `ChartData` itself
  (high bar), or keep them in your extractor's internals.

- **Returning a `ChartData` you built bypassing pydantic.** If the
  client returns partial JSON, run it through `parse_chartdata`,
  which validates. Building a dict and casting to `ChartData` without
  validation can hide bugs.

- **Computing your own accuracy stats in the extractor.** Don't. The
  point of the canonical schema is one comparable number per
  extractor. The scorer does the math.

## A checklist for any extension

Before opening a PR, go through this:

- [ ] My new code follows the `ChartData` contract.
- [ ] My new code passes through `parse_chartdata` if it receives
      raw model output.
- [ ] Heavy imports are inside functions, not at module top.
- [ ] Heavy deps are declared as `[project.optional-dependencies]`
      extras in `pyproject.toml`.
- [ ] `tests/test_smoke.py::test_schema_is_light` still passes.
- [ ] `tests/test_smoke.py::test_accuracy_init_does_not_pull_matplotlib`
      still passes.
- [ ] I added a smoke test for my new module.
- [ ] If I added a config knob, I documented it in
      [Configuration.md](Configuration.md) and the senior
      [../Configuration.md](../Configuration.md).
- [ ] If I made a non-obvious design choice, I added an entry to
      [Decisions.md](Decisions.md) and the senior
      [../Decisions.md](../Decisions.md).

## Sources

- `src/sci_fi_parser/schema.py`
- `src/sci_fi_parser/accuracy/{vlm, vlm_config, benchmark}.py`
- `src/sci_fi_parser/accuracy/synthetic/{config, render, style}.py`
- `tests/test_smoke.py`
- `pyproject.toml`
