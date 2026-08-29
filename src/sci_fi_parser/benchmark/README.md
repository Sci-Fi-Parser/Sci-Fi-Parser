# `sci_fi_parser.accuracy` — measurement layer

Tools for generating chart datasets with **ground-truth labels by construction**
and scoring any extractor against them. Two CLI commands and a pluggable
`Extractor` protocol.

The CLIs (`synthetic-bars`, `benchmark`) are installed by `uv sync` via
`[project.scripts]` — no `python -m …` prefix needed.

## `synthetic-bars` — generate ground-truth chart datasets

Charts whose data is known *by construction* (no model-labelled images, no
circularity risk). Default mode is a *controlled density series*: per enabled
type the style is fixed once and only the bar count varies across
`density_steps`; each density is rendered as a matched **labels-off / labels-on
pair** sharing identical data. Labels are auto-fitted so they never overlap.

```bash
synthetic-bars --config config/synthetic_bars.toml --overlay --sqlite
```

Pick types, density ladder, and resolutions in
[../../../config/synthetic_bars.toml](../../../config/synthetic_bars.toml):
`output_types`, `density_steps`, `resolutions`. Run with `--preview` first to
render a small sample of every catalog type.

| Flag | Effect |
|---|---|
| `--preview` | one sample per catalog type into `preview/` |
| `--random N` | random mode instead: N fully-random charts |
| `--augment` | add JPEG/noise/blur realism (labels stay exact) |
| `--overlay` | write `_debug/overlay_*.png` to eyeball label accuracy |
| `--sqlite` | also write `dataset.sqlite3` (mirrors the real schema + geometry) |
| `--refresh` | delete prior outputs in `--out` before generating |
| `--seed N` | reproducibility |

Outputs in `--out`:
- `images/<type>_s<k>_d<NN>_{off,on}_<res>.png` (series) or `random_NNNNN_<res>.png`
- `labels.jsonl` — per chart: `label1` (chart type), `label2` (axes +
  per-series points + `value_range`), `geometry` (bboxes + tick positions, or
  `null`), `meta`
- `_debug/` overlays (with `--overlay`)
- `dataset.sqlite3` (with `--sqlite`)

Tuning: every knob is documented inline in `GenConfig`
([synthetic/config.py](synthetic/config.py)) and mirrored with comments in the
TOML.

## `benchmark` — score any extractor against ground truth

```bash
benchmark --data train_data/synthetic --out reports/run1
```

The default extractor is `noisy-oracle` (a test double that perturbs truth) so
the harness runs without a model server. For a real model, install
[ollama](https://ollama.com) locally, `ollama pull qwen2.5vl:7b`, then use
`--extractor vlm` to pick up the model declared in
[../../../config/vlm.toml](../../../config/vlm.toml) — or override per-run
with `--extractor vlm:<tag>`. The extractor talks to any OpenAI-compatible
chat-completions endpoint (`base_url` in the profile); ollama's `/v1` compat
layer is the default.

**Metrics**
- Error = `|predicted − true|` as **% of the true value**. Hover any card in
  the HTML report for the exact definition.
- **Recall** = bars found / true bars. **Precision** = correct
  (series, category) bars / predicted.
- **Type acc** = fraction of charts where the extractor's `chart_type` matched
  truth. **Mean conf** = average VLM self-reported confidence (not necessarily
  calibrated).

**Outputs** in `--out`:
- `report.html` — self-contained: summary cards (hover for tooltips), breakdowns
  by preset / density / labels, ranked best/worst charts with thumbnails,
  sortable per-chart table
- `results.json` — machine-readable per-chart rows for CI / trend tracking

## Configuring the VLM extractor

The active model, prompt, and endpoint live in
[../../../config/vlm.toml](../../../config/vlm.toml). Edit that file to swap
models permanently, or override on the CLI for one-offs:

```bash
benchmark --data ... --extractor vlm:qwen2.5vl:7b-q8_0   # one-shot
benchmark --data ... --vlm-config config/vlm_q8.toml      # A/B test
```

Resolution order: CLI flag > `config/vlm.toml` in cwd > built-in defaults.

## Adding a new extractor

Anything implementing the `Extractor` protocol from
[../schema.py](../schema.py) plugs in:

```python
from pathlib import Path
from sci_fi_parser.schema import ChartData


class MyExtractor:
    name = "my-extractor"

    def extract(self, image_path: Path) -> ChartData: ...
```

Register it in `build_extractor` ([benchmark.py](benchmark.py)). `ChartData`
is the single canonical schema every extractor must return: the VLM is
constrained to emit it (via `response_format: json_schema`), CV+OCR pipelines
are mapped into it.

> Note: pure OCR is *not* a standalone extractor (it reads text, not data
> points) — pair it with CV.

## Layout

```
accuracy/
  __init__.py
  benchmark.py      # `benchmark` CLI: runs an extractor, writes report
  vlm_compare.py    # `benchmark-compare` CLI: runs N models, builds leaderboard
  synthetic/        # ground-truth chart generator
```

The VLM extractor itself (`ChatCompletionsVLM` + `VLMProfile` + `load_profile`) lives
in the sibling package [../vlm/](../vlm/) — separated from this package so
non-benchmark callers (e.g. the runtime pipeline) can import the model
client without pulling in the measurement machinery.

Config files live at repo-root [config/](../../../config/), separate from the
package source so users edit data, not code.
