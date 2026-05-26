# Synthetic data + extractor benchmark

Two dev tools that form the **measurement layer** of the pipeline:

1. **`synthetic_bars.py`** — generate bar charts whose data is known *by
   construction* (no model/human labels, no circularity).
2. **`benchmark.py`** — run any extractor (VLM / OCR+CV / chart model) against
   that ground truth and score it; the aggregate error is the empirical **error
   margin** we later attach to real extractions in SQL.

All commands assume the project venv: prefix with `.venv/bin/python`.

---

## 1. Generate synthetic charts

**Default mode is a controlled density series.** For each enabled type the style is
fixed once (colours, titles, figure, DPI, value range) and only the bar count varies
across `density_steps`; each density is rendered as a matched **off/on pair** (labels
hidden vs shown) sharing identical data. Labels are auto-fitted (rotated +
font-shrunk) so they never overlap — like a real chart.

```bash
.venv/bin/python scripts/synthetic_bars.py --config scripts/synthetic_bars.toml --overlay --sqlite
```

- Pick types, density ladder, and resolutions in `synthetic_bars.toml`:
  `output_types = ["simple", "2-line", …]`, `density_steps = [...]`,
  `resolutions = [240, 480, 720]`. Run with `--preview` first to see a small
  sample of every catalog type.
- Catalog aliases: `simple multicolor grouped stacked horizontal line 2-line
  3-line multiline`.
- `resolutions` = output image heights in px (aspect preserved); each chart is
  rendered at every resolution, value appended to the filename.
- **Total images = types × per_type × densities × 2 (off/on) × resolutions.**
  Default (`["simple"]`, `per_type=1`, one resolution) → **20 images**.
- `meta.pair_id` links each off/on twin; `meta.series_id` groups a density ramp;
  `meta.density` is the bar count — use these to plot accuracy-vs-density.

Useful flags:

| Flag | Effect |
|------|--------|
| `--preview` | render one small sample of every catalog type into `preview/` |
| `--random N` | random mode instead: N fully-random charts (variety/volume) |
| `--augment` | add JPEG/noise/blur realism (labels stay exact) |
| `--overlay` | write `_debug/overlay_*.png` to eyeball label accuracy |
| `--sqlite` | write `dataset.sqlite3` mirroring the real schema + geometry/meta |
| `--refresh` | delete prior `images/`, `_debug/`, `labels.jsonl`, `dataset.sqlite3` in `--out` first |
| `--config FILE` | override knobs from TOML (see `synthetic_bars.toml`) |
| `--seed N` | reproducibility |

**Outputs** in `--out`:
- `images/<type>_s<k>_d<NN>_{off,on}_<res>.png` (series) or `random_NNNNN_<res>.png`.
- `labels.jsonl` — per chart: `label1` (type), `label2` (axes + per-series points +
  `value_range`), `geometry` (bboxes + value-axis ticks, or `null`), `meta`.
- `_debug/` overlays (with `--overlay`), `dataset.sqlite3` (with `--sqlite`).

**Tuning:** every knob is documented inline in `GenConfig` and mirrored with
comments in `synthetic_bars.toml`. Within a series the style is held constant so
density is the only variable; vary the ranges across series to fight the
sim-to-real gap.

---

## 2. Benchmark an extractor

```bash
.venv/bin/python scripts/benchmark.py --data train_data/synthetic --out reports/run1
```

Default extractor is `noisy-oracle` (a test double that perturbs the truth) so
the harness runs with no model installed. Swap in a real one with
`--extractor ollama:<model>` (needs `pip install ollama` + the model pulled).

**Metrics**
- Error = `|predicted − true|` as **% of the value-axis span** (scale-free across
  the 1→1e6 value ranges).
- **Recall** = bars found / true bars; **precision** = correct (series,category)
  bars / predicted. Detection errors (missed / hallucinated bars) are reported
  separately from value errors.

**Outputs** in `--out`:
- `report.html` — self-contained (open in a browser): summary cards, breakdowns
  by type / density / labels, best & worst charts with thumbnails, sortable table.
- `results.json` — per-chart rows for CI / trend tracking.

### Adding a real extractor

Implement one method and register it in `build_extractor`:

```python
class MyExtractor:
    name = "my-extractor"
    def extract(self, image_path: Path) -> ChartData: ...
```

`ChartData` is the single canonical schema every extractor must return; VLMs are
constrained to emit it (e.g. Ollama `format=ChartData.model_json_schema()`),
OCR+CV pipelines are mapped into it. Note: **pure OCR is not a standalone
extractor** (it reads text, not data points) — pair it with CV.

---

## Where the code lives

These two `scripts/` entries are thin shims; the real code is the importable
package `sci_fi_parser.accuracy`:

- `scripts/synthetic_bars.py` → `src/sci_fi_parser/accuracy/synthetic/`
- `scripts/benchmark.py`      → `src/sci_fi_parser/accuracy/benchmark.py`

The pipeline imports the package directly; the shims just keep the documented
CLI paths working.

## Dependencies
Runtime deps (`matplotlib`, `numpy`, `pillow`, `pydantic`) are pinned in
`pyproject.toml`. `opencv-python` (overlays) and `ollama` (VLM extractor) are
optional extras — install with `pip install -e ".[cv,ollama]"` if you need them.
