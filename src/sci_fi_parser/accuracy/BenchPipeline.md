# Bench Pipeline Guide

This guide explains how `bench_pipeline.py` works, what each part calls, and how to run and test it.

`bench_pipeline.py` is a staged benchmark wrapper. It keeps the intended extraction pipeline shape visible, but it delegates the actual scoring and report generation to `sci_fi_parser.accuracy.benchmark`.

The current active flow is:

```text
bench-pipeline CLI
-> main()
-> _parse_args()
-> _resolve_profile()
-> run_pipeline()
-> load_inputs()
-> optional bench_ocr_and_cv()
-> bench_vlm_output()
-> benchmark.run_benchmark()
-> report.html + results.json
```

## Required Dataset Layout

The benchmark currently expects a synthetic dataset directory with this structure:

```text
train_data/synthetic/
  images/
    chart_1.png
    chart_2.png
    ...
  labels.jsonl
```

`labels.jsonl` contains the ground-truth chart data. The benchmark compares extractor output against these labels.

## File Purpose

`src/sci_fi_parser/accuracy/bench_pipeline.py` exists to benchmark the project pipeline in stages.

It currently supports these stages:

- Load synthetic benchmark inputs.
- Optionally run the OCR/CV stage.
- Pass OCR text into the VLM prompt as extra context.
- Run the existing benchmark/scoring layer.

Two intended stages are still placeholders:

- `bench_image_extraction()`
- `bench_classification()`

They currently return `None` and are not used by `run_pipeline()`.

## `bench_ocr_and_cv(inputs)`

Purpose:

- Run the current OCR/CV stage against all images.
- Store formatted OCR/CV output in an `OCRSet`.
- Return that `OCRSet` so VLM benchmarking can use it as prompt context.

## `bench_vlm_output(...)`

Purpose:

- Prepare optional OCR prompt text.
- Run the existing benchmark.
- Return the aggregate benchmark metrics.

## `run_pipeline(...)`

Purpose:

- This is the public Python entry point for the staged benchmark pipeline.

Flow:

- Load benchmark inputs.
- If `ocr_cv=True`, run OCR/CV and collect prompt context.
- Run the final benchmark/scoring stage.
- Return aggregate metrics as a dictionary.

Calls:

- `load_inputs(data, limit)`
- `bench_ocr_and_cv(inputs)` when `ocr_cv=True`
- `bench_vlm_output(...)`

This function does not call the placeholder stages.

## `_parse_args()`

Purpose:

- Define and parse the command-line interface.

CLI options:

- `--data`: required synthetic dataset directory.
- `--out`: report/output directory. Defaults to `reports/pipeline`.
- `--extractor`: extractor to benchmark. Defaults to `noisy-oracle`.
- `--vlm-config`: optional VLM profile TOML file.
- `--seed`: random seed for deterministic noisy-oracle behavior.
- `--limit`: only benchmark the first `N` sorted charts.
- `--ocr-cv`: run OCR/CV first and pass its text to the VLM prompt.

## `main()`

```python
def main() -> None:
    args = _parse_args()
    run_pipeline(
        data=args.data,
        out=args.out,
        extractor_name=args.extractor,
        profile=_resolve_profile(args.vlm_config),
        seed=args.seed,
        limit=args.limit,
        ocr_cv=args.ocr_cv,
    )
```

Purpose:

- Command-line entry point.

This function is connected to the console command in `pyproject.toml`:

```toml
[project.scripts]
bench-pipeline = "sci_fi_parser.accuracy.bench_pipeline:main"
```

That means after installing the project and having some synthetic data, you can run:

```bash
bench-pipeline --data train_data/synthetic
```


You can generate some synthetic data for example with this:
```bash
uv run synthetic-bars --random 10 --out train_data/synthetic --refresh
```
```

```
With `uv`, use:

```bash
uv run bench-pipeline --data train_data/synthetic
```

## Output Files

The benchmark writes output into the `--out` directory.

Main files:

- `report.html`: human-readable benchmark report with summary cards, charts, and per-chart details.
- `results.json`: machine-readable benchmark results.

Example:

```text
reports/pipeline/
  report.html
  results.json
```

## Run With OCR/CV Context

Use `--ocr-cv` to run the current OCR/CV stage before benchmarking:

```bash
uv run bench-pipeline --data train_data/synthetic --out reports/pipeline-ocr --extractor noisy-oracle --limit 5 --ocr-cv
```

This changes the flow to:

```text
load_inputs()
-> bench_ocr_and_cv()
-> _ocr_suffix()
-> benchmark.run_benchmark(..., prompt_suffixes=...)
```

With `noisy-oracle`, the OCR prompt suffix is accepted by the same extractor interface, but the oracle does not need OCR to produce predictions. This command is mainly useful to verify OCR/CV wiring.

For real VLM extractors, OCR text becomes extra prompt context.
