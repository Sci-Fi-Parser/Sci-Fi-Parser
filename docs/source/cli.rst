Command Line Interface
=====================

The project installs a few command-line entry points via ``[project.scripts]``
in ``pyproject.toml``.

Parse a folder of PDFs with the default pipeline:

.. code-block:: bash

   parse-chart

This script reads the input, extraction, output, and VLM settings from the
constants defined in :mod:`sci_fi_parser.main`. It is intended for quick local
runs and simple smoke testing.

Benchmark-related commands are also installed:

.. code-block:: bash

   benchmark
   benchmark-compare
   bench-pipeline

For synthetic dataset generation, the project also exposes:

.. code-block:: bash

   synthetic-bars

The benchmark package has its own README in
``src/sci_fi_parser/benchmark/README.md`` with the detailed workflow, CLI
flags, and output format. This page only captures the installed entry points so
the Sphinx docs stay compact.