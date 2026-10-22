# Benchmarking

Benchmarking is developer tooling for measuring extraction against annotated ground truth; it is not part of normal end-user parsing. `bench_pipeline.py` is the main entry point, exposed as `bench-pipeline`. It loads synthetic `images/` with `truth.jsonl` or Benetech `images/` with `annotations/`, optionally runs classification and OCR/CV, runs an extractor, scores the resulting `ChartData`, and writes `results.json` and an HTML report.

The end-to-end benchmark with VLM extraction is relatively useful for detecting accidental OCR/CV regressions on the annotated Benetech dataset. Component metrics are still needed to explain a regression, but final VLM value, recall, type, and bar-count results show whether an OCR/CV change helps the actual pipeline.

## Weaknesses and future work

- The implementation is difficult to follow and is split between the staged pipeline and older helpers in `benchmark.py`. Some modules are partly deprecated: `benchmark.py` is mostly legacy but is still reused for extractors, aggregation, and reports.
- After some main pipeline refactors, the benchmark doesn't follow the same structure anymore.
- Most users cannot run a meaningful benchmark because annotated chart data is required. The command orchestration and dataset-generation utilities could therefore move outside the installed package into `scripts/`. A focused rewrite is also reasonable, but should preserve the useful scoring and end-to-end regression contracts.
- Runs do not define pass/fail regression thresholds or record enough dataset, prompt, model, and revision provenance for strict reproducibility. Stored comparison results should not be reused without verifying those inputs.
- Value error only covers matched or positionally paired values. Missing bars can therefore make value error look good, so it must be reviewed together with recall and bar-count error.
- OCR/CV context benchmarks should use the same prompt formatting as the runtime pipeline. Context-free and different structured-context variants should be compared explicitly before changing what is sent to the VLM.
