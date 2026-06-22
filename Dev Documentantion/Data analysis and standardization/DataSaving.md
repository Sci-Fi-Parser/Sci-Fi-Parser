# Chart Storage Strategy

# Design Principles

The pipeline produces multiple layers of information:

* Extracted image metadata
* OCR output
* VLM output
* Raw model responses
* Structured chart data

These serve different purposes and should not all be stored in the same format.

The recommended approach is:

1. Preserve the complete extraction artifact as raw JSON.
2. Create normalized analytical tables for querying.
3. Store analytical tables in Parquet format.
4. Query analytical tables using DuckDB.

---

# Storage Layers

## Layer 1: Raw Extraction Archive

Store the complete extraction output exactly as produced by the pipeline.

Example:

```json
{
  "chart_id": "b219ad15-c988-4459-9942-52ad814e71d4",
  "pdf_id": "4974d4ed-4c99-4336-aa3b-18bd1acd9ffc",
  "page_number": 6,
  "image_path": "temp/extracted_images/b219ad15-c988-4459-9942-52ad814e71d4.png",
  "ocr": {...},
  "vlm": {...},
  "metadata": {...}
}
```

Recommended format:

```text
raw/
    chart_runs.jsonl
```

One JSON object per extracted chart.

### Benefits

* Full reproducibility
* Preserves original OCR output
* Preserves original model responses
* Allows future reprocessing without rerunning models
* Supports auditing and debugging

---

# Layer 2: Analytical Tables

Flatten extracted chart data into normalized tables.

---

## Table: charts

One row per chart.

| Column               | Description                   |
| -------------------- | ----------------------------- |
| chart_id             | Unique chart identifier       |
| pdf_id               | Source PDF                    |
| page_number          | Page number                   |
| image_path           | Extracted image path          |
| chart_type           | Bar, line, scatter, pie, etc. |
| confidence           | Model confidence              |
| model                | VLM model used                |
| extraction_timestamp | Extraction time               |
| pipeline_version     | Pipeline version              |

Example:

| chart_id | chart_type        |
| -------- | ----------------- |
| abc123   | line_chart        |
| def456   | grouped_bar_chart |

---

## Table: series

One row per chart series.

| Column       | Description              |
| ------------ | ------------------------ |
| series_id    | Unique series identifier |
| chart_id     | Parent chart             |
| series_index | Position in chart        |
| series_name  | Series label             |

Example:

| series_id | chart_id | series_name      |
| --------- | -------- | ---------------- |
| 1         | def456   | BOJ Holdings     |
| 2         | def456   | Outstanding JGBs |

---

## Table: points

One row per data point.

| Column      | Description                         |
| ----------- | ----------------------------------- |
| point_id    | Unique point identifier             |
| chart_id    | Parent chart                        |
| series_id   | Parent series                       |
| point_index | Position within series              |
| x_raw       | Original x-axis value               |
| x_numeric   | Numeric representation if available |
| x_type      | year, category, date, numeric       |
| y           | Y value                             |
| y_unit      | %, USD, count, etc.                 |

Example:

| chart_id | series_name  | x_raw | y  |
| -------- | ------------ | ----- | -- |
| def456   | BOJ Holdings | 2000  | 1  |
| def456   | BOJ Holdings | 2010  | 5  |
| def456   | BOJ Holdings | 2017  | 46 |

---

# Recommended File Formats

## Raw Data

```text
JSONL
```

Reason:

* Human-readable
* Easy debugging
* Supports append-only workflows
* Preserves full extraction context

Directory:

```text
raw/chart_runs.jsonl
```

---

## Analytical Data

```text
Parquet
```

Reason:

* Columnar storage
* Compression
* Fast analytical queries
* Compatible with DuckDB, Pandas, Polars, Spark, BigQuery

Directory:

```text
tables/
    charts.parquet
    series.parquet
    points.parquet
```

---

# Query Layer

Use DuckDB as the primary query engine.

Example:

```sql
SELECT chart_type,
       COUNT(*)
FROM charts
GROUP BY chart_type;
```

Example:

```sql
SELECT p.x_raw,
       p.y
FROM points p
JOIN series s
    ON p.series_id = s.series_id
WHERE s.series_name = 'Outstanding JGBs';
```

Example:

```sql
SELECT AVG(y)
FROM points
WHERE chart_id = 'def456';
```

---

# Directory Structure

```text
data/

├── raw/
│   └── chart_runs.jsonl
│
├── tables/
│   ├── charts.parquet
│   ├── series.parquet
│   └── points.parquet
│
├── images/
│   └── extracted_chart_images/
│
└── metadata/
    └── extraction_runs.json
```

---

# Future Extensions

The schema can be expanded to include:

## OCR Labels Table

```text
ocr_labels
```

Stores:

* OCR text
* Bounding boxes
* OCR confidence

---

## Classification Table

```text
classifications
```

Stores:

* Chart classifier outputs
* Model confidence
* Predicted chart categories

---

## Provenance Table

```text
provenance
```

Stores:

* Source PDF
* Page number
* Extraction timestamp
* Model versions
* Pipeline versions

---

# Final Recommendation

Use a dual-storage architecture:

### Reproducibility Layer

Store complete extraction artifacts as:

```text
JSONL
```

### Analytics Layer

Store normalized tables as:

```text
Parquet
```

Query analytical data using:

```text
DuckDB
```

This provides:

* Full reproducibility
* Efficient storage
* Fast SQL querying
* Easy scaling
* Compatibility with future data platforms
