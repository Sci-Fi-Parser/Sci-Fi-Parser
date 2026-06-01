# Architecture of data output of Scifi-Parser

Since the goal is to build a distributable pip package that can be easily run locally,
the default option should be lightweight. Give the user optionalities for more complex
data saving and analysis options.

Current idea:

EXTRACTION → JSONL
ANALYTICS EXPORT → PARQUET

Save ALL produced data to local folder using JSONL (JSON lines) data format.
When moving to analytics we will convert JSONL to Parquet dataformat using PyArrow.
Analytics will be done locally using DuckDB
You can run SQL directly on Parquet files without importing into a database.

output/ (output folder)
  manifest.json          (A small metadata file)
  documents.jsonl        (Save original documents)
  pages.jsonl            (Save pages of documents)
  charts.jsonl           (Save charts from pages)
  datapoints.jsonl       (Save datapoints from charts)   
  runs.jsonl             (Metadata from runs)
  errors.jsonl           (Save possible errors encountered)
  evidence/
    chart_crops/         (Full chart png files) These are used for debugging and traceability
    overlays/            (OCR outputs as png files)

PDFs
  ↓
Extraction
  ↓
JSONL records (safe incremental writes)
  ↓
Parquet export
  ↓
DuckDB / pandas / Spark analytics


How this is implemented throughout the pipeline:

-We create a writer object at the beginning of the pipeline and pass
 it along the pipeline through all the steps (extraction, ocr, VLM, etc.)
 all data saving will be done through this writer object and it has full control
 of the output folders. 


Batch of pdfs is given to Scifiparser and also the writer object:

with DatasetWriter("output") as writer: 
    for pdf_path in pdfs:
        result = process_pdf(pdf_path, writer)

These functions will be called in the different sections of the pipeline
writer.save_page_image(page_image, document_id, page_number)
writer.save_chart_crop(chart_image, chart_id)
writer.write_chart(chart_record)
writer.write_datapoint(datapoint_record)
writer.write_error(error_record)


main.py creates writer
↓
writer is passed through pipeline
↓
each step saves its own evidence + records
↓
writer owns folder structure and file-writing details



documents.jsonl

{"document_id":"doc_001","path":"fund.pdf"}


charts.jsonl

{"chart_id":"chart_001","document_id":"doc_001","page":5}


datapoints.jsonl

{"chart_id":"chart_001","year":2019,"return":0.11}
{"chart_id":"chart_001","year":2020,"return":-0.02}
{"chart_id":"chart_001","year":2021,"return":0.15}


