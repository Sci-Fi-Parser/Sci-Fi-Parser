# Benchmarks

Developer tooling for measuring extraction against annotated chart images. It is
not part of normal parsing. Two commands are installed with the package:

| Command | Module | Measures |
|---|---|---|
| `bench-pipeline` | `bench_pipeline.py` | End-to-end: pipeline stages + VLM, scored `ChartData` |
| `bench-ocr-cv` | `ocr_cv.py` | OCR/CV component metrics, no VLM |

## Datasets

Both commands read a directory with chart images in `images/` and ground truth in
one of two layouts (`--dataset`):

- `synthetic` (default): `truth.jsonl`, one record per image, written by the
  synthetic generator. Values, and optionally pixel geometry, are known by
  construction.
- `benetech`: `annotations/<image stem>.json` in the
  [Benetech](https://www.kaggle.com/competitions/benetech-making-graphs-accessible)
  format. `scripts/filter_benetech_vertical_bars.py` copies one chart type out of
  the extracted Benetech set.

Generate a synthetic dataset from the repository root:

```bash
uv run python -m scripts.synthetic.cli --config scripts/synthetic/synthetic_bars.toml --out train_data/synthetic
```

`--preview` renders one sample of every chart type, `--overlay` draws the truth
over each image, `--augment` adds JPEG/noise/blur. `scripts/synthetic/ocr_cv_stage.toml`
is a fixed vertical-bar set with full geometry for `bench-ocr-cv`. Every knob is
documented in `scripts/synthetic/config.py`.

## `bench-pipeline`

```bash
bench-pipeline --data train_data/synthetic --out reports/run1
bench-pipeline --data train_data/benetech --dataset benetech --model qwen3-vl:8b --limit 50
```

Images go through the same stage functions as `parse_folder`
(`start_classification`, `start_ocr`, `start_vlm`), so prompts and stage
behaviour match a real parsing run. Only charts the pipeline would send to the
VLM (classified as bar or line charts) get a prediction; the rest are scored as
"no prediction".

| Flag | Effect |
|---|---|
| `--vlm-config PATH` | VLM profile TOML (default: the packaged `config/vlm.toml`) |
| `--model NAME` | Override the profile's model, e.g. to compare models |
| `--no-classify` | Take the chart class from the truth instead of the classifier |
| `--no-ocr` | Send no OCR/CV context to the VLM |
| `--limit N` | Only the first N images, sorted by file name |

The VLM needs an OpenAI-compatible chat completions endpoint (see
`docs/design_notes/vlm.md`). To compare models or prompts, run once per
`--model` or `--vlm-config` into separate `--out` directories.

### Metrics

Truth and predicted points are matched on (series name, x label), ignoring case
and surrounding whitespace. When both sides have a single series, its name is
ignored.

- **Error**: `|predicted − true|` as a percentage of the truth data range,
  reported as mean, median, p95 and the share within 5 %. It only covers
  matched points, so a run that misses many points can still show a low
  error. Always read it together with recall and precision.
- **Recall**: matched / true points. **Precision**: matched / predicted points.
- **Missed / extra**: true points without a match / predicted points without a match.
- **Type accuracy**: share of charts whose predicted `chart_type` equals the
  truth; charts without a prediction count as wrong.
- **No prediction**: charts that were not sent to the VLM or whose extraction failed.

### Outputs

- `results.json`: `run` (start time, package version, dataset, enabled stages,
  VLM model, endpoint and prompt, seconds per stage), `summary`, `breakdowns`
  by synthetic `preset` / `density` / `labels_on`, and one row per chart.
- `report.html`: self-contained report with summary cards, breakdowns, the
  distribution of signed deviations, chart cards from best to worst with
  thumbnails, and a sortable per-chart table.

## `bench-ocr-cv`

```bash
bench-ocr-cv --data train_data/ocr_cv_stage --ocr-mode both --out reports/ocr_cv
```

Runs bar detection and axis analysis on vertical bar charts and scores the
intermediate OCR/CV results. `--ocr-mode oracle` feeds the synthetic truth OCR
tokens to the analysis, so role and calibration errors can be told apart from
OCR errors; `detected` runs PaddleOCR; `both` runs each. Oracle mode needs
synthetic data with full geometry.

Metrics per mode:

- **Role P/R/F1**: Y-tick and X-label role assignment of OCR tokens matched to truth tokens.
- **Y-tick recall**: true Y-axis ticks recognised with the correct text.
- **Calibration**: fit rate, success rate (normalized MAE < 0.01), and mean
  tick value error of the fitted linear Y-axis, normalised by the tick span.
- **Bars**: precision/recall of detected bar boxes at IoU ≥ 0.5, and exact bar
  count accuracy. Benetech data only has the bar count.

Outputs `results.json` (summary per mode plus per-sample records) and a small `report.html`.
