"""Select matplotlib's headless backend before pyplot is imported anywhere.

Importing this module first (it is the first import in the package ``__init__``)
guarantees ``Agg`` is active before any submodule imports ``matplotlib.pyplot`` or
creates a figure, so the generator runs without a display.
"""

import matplotlib

matplotlib.use("Agg")
