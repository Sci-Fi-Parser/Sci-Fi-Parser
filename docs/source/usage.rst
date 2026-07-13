Usage
=====

The package exposes a single high-level entry point for end-to-end parsing:

.. code-block:: python

   from sci_fi_parser import parse_folder

   result = parse_folder(
       "path/to/pdfs",
       output_dir="output",
       extracted_image_dir="temp/extracted_images",
       vlm_config="config/vlm.toml",
   )

   print(result.summary())

``parse_folder`` walks a PDF file or a flat directory of PDFs, extracts chart
images, classifies them, runs OCR / computer vision, and finally applies the
VLM stage. If ``output_dir`` is provided, the parsed dataset is written to JSONL
and Parquet outputs.

The returned :class:`sci_fi_parser.api.ParseResult` gives access to the parsed
image and PDF collections:

.. code-block:: python

   image_set = result.images
   pdf_set = result.pdfs

   for image_id, record in image_set.items():
       print(image_id, record)

Output written by :meth:`sci_fi_parser.api.ParseResult.save` is organized as:

* ``raw/image_set.jsonl`` for the full record stream
* ``tables/charts.parquet`` for one row per chart
* ``tables/series.parquet`` for one row per series
* ``tables/points.parquet`` for one row per extracted point

For batch runs, keep the extraction folder separate from the final output
directory. The extraction folder is temporary working storage and is not meant
to be checked in or treated as a final artifact.