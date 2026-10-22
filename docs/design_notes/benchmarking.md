# Benchmarking

Benchmarking is developer tooling for measuring extraction against annotated ground truth; it is not part of normal end-user parsing. Usage, metrics, and outputs are documented in `src/sci_fi_parser/benchmark/README.md`.

There are two benchmarks. `bench-pipeline` (`bench_pipeline.py`) is end-to-end: it loads synthetic `images/` with `truth.jsonl` or Benetech `images/` with `annotations/`, runs the same stage functions as `parse_folder` (classification, OCR/CV, VLM), scores the resulting `ChartData`, and writes `results.json` and an HTML report. `bench-ocr-cv` (`ocr_cv.py`) scores the OCR/CV stage on its own, without a VLM. Both share the dataset loader in `dataset.py`.

The end-to-end benchmark is relatively useful for detecting accidental OCR/CV regressions on the annotated Benetech dataset. Component metrics are still needed to explain a regression, but final VLM value error, recall, type accuracy, and missed/extra points show whether an OCR/CV change helps the actual pipeline. Because the benchmark calls the pipeline stage functions directly, the VLM receives exactly the prompt and OCR/CV context that a real parsing run sends.

## Weaknesses and future work

- Most users cannot run a meaningful benchmark because annotated chart data is required. The commands could move out of the installed package into `scripts/`, next to the synthetic generator.
- Runs record their dataset, stages, VLM model, endpoint, prompt, and package version, but not a dataset checksum, and define no pass/fail regression thresholds. Stored results should not be compared without checking that these inputs match.
- Value error only covers matched points. Missing points can therefore make value error look good, so it must be reviewed together with recall and precision.
- The VLM stage only reports total time; there is no per-chart timing.
- Comparing OCR/CV context variants (none, selected evidence, full JSON) currently means toggling `--no-ocr` or changing `format_ocr_output`; there is no built-in A/B mode.
