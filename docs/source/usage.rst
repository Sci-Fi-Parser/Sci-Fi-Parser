Usage
=====

The package exposes a single high-level entry point for end-to-end parsing:

.. code-block:: python

   from sci_fi_parser import parse_folder

   result = parse_folder(
       input_dir="path/to/pdfs",
       output_dir="output",
       extracted_image_dir="output/extracted_images",
   )

   print(result.summary())

:func:`parse_folder() <sci_fi_parser.api.parse_folder>` walks a PDF file or a flat directory of PDFs, processing each
PDF one at a time through the full pipeline: it extracts chart images, classifies them, runs OCR / computer vision,
applies the VLM stage, and finally saves a hash of the PDF to cache.
If ``output_dir`` is provided, the parsed dataset is written to JSONL and Parquet outputs.
Alternatively, the output can be saved using the returned object with :meth:`ParseResult.save() <sci_fi_parser.api.ParseResult.save>`

A ``cache.db`` is stored for each output directory. When :func:`parse_folder() <sci_fi_parser.api.parse_folder>` is run again with the same output location,
it reuses cached results and skips PDFs that have already been processed.
Using a different output directory creates a different cache. Note that if no ``output_dir`` is provided, the default location for the cache will be ``temp/``.

The returned :class:`ParseResult <sci_fi_parser.api.ParseResult>` gives access to the parsed
image and PDF collections:

.. code-block:: python

   image_set = result.images
   pdf_set = result.pdfs

   for image_id, record in image_set.items():
       print(image_id, record)

Saved output is organized as append-only JSONL records and derived Parquet tables:

* ``raw/image_set.jsonl`` for the image record stream
* ``raw/pdf_set.jsonl`` for the PDF record stream
* ``tables/charts.parquet`` for one row per chart
* ``tables/series.parquet`` for one row per series
* ``tables/points.parquet`` for one row per extracted point
* ``tables/pdfs.parquet`` for one row per PDF

The VLM config file has these variables:

.. code-block:: toml
    
    model = "qwen2.5vl:3b" # Ollama only
    base_url = "http://localhost:11434/v1" # VLM endpoint base url
    api_key_env = "" # API key, if any
    prompt = "" # VLM prompt if any
