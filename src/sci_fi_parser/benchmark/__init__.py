"""Accuracy-metrics layer.

Two cooperating pieces, coupled by a data contract (``labels.jsonl``), not by code:

* :mod:`sci_fi_parser.accuracy.synthetic` -- generates charts whose values are known
  *by construction*, so they carry exact ground truth.
* :mod:`sci_fi_parser.accuracy.benchmark` -- scores any extractor against that ground
  truth and turns the per-chart errors into an empirical error margin.

Import the subpackage you need directly; this package intentionally pulls in neither,
so e.g. importing ``benchmark`` does not drag in matplotlib.
"""
