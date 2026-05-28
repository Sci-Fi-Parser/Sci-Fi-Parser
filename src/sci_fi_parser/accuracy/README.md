# `sci_fi_parser.accuracy` — measurement layer

Tools for generating chart datasets with **ground-truth labels by construction**
and scoring any extractor against them. Three CLI commands
(`synthetic-bars`, `benchmark`, `benchmark-compare`) and a pluggable
`Extractor` protocol.

The CLIs (`synthetic-bars`, `benchmark`, `benchmark-compare`) are installed by
`uv sync` via `[project.scripts]`, but only resolve on PATH inside the project
venv. Pick one:

```bash
.venv/bin/benchmark --data train_data/synthetic        # explicit launcher
source .venv/bin/activate && benchmark --data ...      # activate for the session
uv run benchmark --data train_data/synthetic           # uv-native
```

A bare `benchmark` in a fresh shell will produce `command not found`.

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
the harness runs without an ollama server. For a real model:

```bash
ollama pull qwen2.5vl:7b              # one-time, downloads weights
ollama serve &                        # daemon must be running -- check with `pgrep ollama`
benchmark --data ... --extractor ollama         # uses vlm.toml's model
benchmark --data ... --extractor ollama:qwen2.5vl:7b-q8_0   # one-off override
```

GPU is used automatically when Ollama detects CUDA/Metal — no flag needed.
`num_gpu` in [vlm.toml](../../../config/vlm.toml) (or `BENCH_NUM_GPU=N`) caps
the number of *model layers* offloaded to GPU; lower it if VRAM is tight.

If the daemon isn't running you'll see one
`ConnectionError: Failed to connect to Ollama` per chart and the run will
finish with 0% recall everywhere.

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

The active model, prompt, and ollama options live in
[../../../config/vlm.toml](../../../config/vlm.toml). Edit that file to swap
models permanently, or override on the CLI for one-offs:

```bash
benchmark --data ... --extractor ollama:qwen2.5vl:7b-q8_0   # one-shot
benchmark --data ... --vlm-config config/vlm_q8.toml         # A/B test
BENCH_NUM_GPU=18 benchmark --data ... --extractor ollama    # env override
```

Resolution order: CLI flag > `config/vlm.toml` in cwd > built-in defaults.

## `benchmark-compare` — run multiple VLMs and build a leaderboard

Same scoring pipeline as `benchmark`, but runs once per model listed in a
comparison TOML and emits a single `leaderboard.html` (+ `leaderboard.md`)
linking out to each model's per-run report.

```bash
benchmark-compare --config config/vlm_comparison.toml --data train_data/synthetic
benchmark-compare --config config/vlm_smoke.toml --dry-run        # just preflight
```

Before any model runs, a *preflight* lists each unique tag as LOCAL / MISSING
plus free disk. If anything is missing it asks `--pull
{prefetch,circular,skip}` (or prompts interactively):

- `prefetch` — pull all missing models up front, keep them after.
- `circular` — pull → run → `ollama rm`, one model at a time. Only removes
  models *this run* pulled; preexisting locals are never touched.
- `skip`     — leave missing models as ERROR rows.

Use `--resume` after a crash to skip models that already have a `results.json`.

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
is the single canonical schema every extractor must return: VLMs are
constrained to emit it (via `format=ChartData.model_json_schema()` in ollama),
CV+OCR pipelines are mapped into it.

> Note: pure OCR is *not* a standalone extractor (it reads text, not data
> points) — pair it with CV.

## Layout

```
accuracy/
  __init__.py
  benchmark.py     # `benchmark` CLI: NoisyOracle + runner that glues scoring + report
  scoring.py      # per-chart metrics + aggregation (no I/O, no presentation)
  report.py       # HTML report rendering for a single benchmark run
  assets/         # report.css + report.sort.js, loaded via importlib.resources
  vlm.py          # OllamaVLM extractor (talks to the ollama daemon)
  vlm_config.py   # VLMProfile + load_profile (TOML -> object)
  vlm_compare.py  # `benchmark-compare` CLI: config + orchestration + interactive prompts
  leaderboard.py  # markdown + HTML leaderboard rendering
  ollama_api.py   # Ollama daemon HTTP client + preflight (no model logic)
  synthetic/      # ground-truth chart generator
```

Config files live at repo-root [config/](../../../config/), separate from the
package source so users edit data, not code.
