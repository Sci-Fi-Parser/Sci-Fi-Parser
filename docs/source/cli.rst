Command Line Interface
======================

Sci-Fi Parser installs several command-line entry points via
``[project.scripts]`` in ``pyproject.toml``.

Parsing PDFs
------------

The main entry point for the package is:

.. code-block:: bash

   parse-folder INPUT

where ``INPUT`` is either a PDF file or a directory containing PDF files.

By default, the parser writes its outputs to an ``output`` directory and runs
the full pipeline:

* Image extraction
* Chart classification
* OCR
* VLM extraction

Example:

.. code-block:: bash

   parse-folder train_data/small_pdfs

Custom output directory
-----------------------

Use ``--output`` (or ``-o``) to specify a custom output directory:

.. code-block:: bash

   parse-folder train_data/small_pdfs --output results

Disabling pipeline stages
-------------------------

Disable chart classification:

.. code-block:: bash

   parse-folder train_data/small_pdfs --no-classify

Disable OCR:

.. code-block:: bash

   parse-folder train_data/small_pdfs --no-ocr

Disable VLM extraction:

.. code-block:: bash

   parse-folder train_data/small_pdfs --no-vlm

Multiple stages may be disabled simultaneously:

.. code-block:: bash

   parse-folder train_data/small_pdfs --no-classify --no-ocr

Displaying help
---------------

To see all available options:

.. code-block:: bash

   parse-folder --help

Other command-line tools
------------------------

Sci-Fi Parser also installs two developer benchmarks that score the pipeline
against annotated chart images:

.. code-block:: bash

   bench-pipeline --data train_data/synthetic
   bench-ocr-cv --data train_data/synthetic

``bench-pipeline`` runs the pipeline stages and a VLM end to end and scores the
extracted data; ``bench-ocr-cv`` scores the OCR/CV stage alone. Synthetic
datasets are generated with ``scripts/synthetic``. Datasets, flags, metrics and
outputs are described in

``src/sci_fi_parser/benchmark/README.md``
