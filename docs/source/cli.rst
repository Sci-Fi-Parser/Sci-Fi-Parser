# Command Line Interface

Sci-Fi Parser installs several command-line entry points via
`[project.scripts]` in `pyproject.toml`.

## Parsing PDFs

The main entry point for the package is:

.. code-block:: bash

scifi-parser INPUT

where `INPUT` is either a PDF file or a directory containing PDF files.

By default, the parser writes its outputs to an `output` directory and runs
the full pipeline:

* Image extraction
* Chart classification
* OCR
* VLM extraction

Example:

.. code-block:: bash

scifi-parser train_data/small_pdfs

## Custom output directory

Use `--output` (or `-o`) to specify a custom output directory:

.. code-block:: bash

scifi-parser train_data/small_pdfs --output results

## Disabling pipeline stages

Individual stages of the pipeline can be disabled:

Disable chart classification:

.. code-block:: bash

scifi-parser train_data/small_pdfs --no-classify

Disable OCR:

.. code-block:: bash

scifi-parser train_data/small_pdfs --no-ocr

Disable VLM extraction:

.. code-block:: bash

scifi-parser train_data/small_pdfs --no-vlm

Multiple stages may be disabled simultaneously:

.. code-block:: bash

scifi-parser train_data/small_pdfs --no-classify --no-ocr

## Displaying help

To see all available options:

.. code-block:: bash

scifi-parser --help

## Other command-line tools

Sci-Fi Parser also installs several utilities for benchmarking and synthetic
dataset generation:

.. code-block:: bash

bench-pipeline
synthetic-bars

The benchmark package has its own documentation in
`src/sci_fi_parser/benchmark/README.md`, which describes the benchmarking
workflow, configuration, and output formats in detail.
