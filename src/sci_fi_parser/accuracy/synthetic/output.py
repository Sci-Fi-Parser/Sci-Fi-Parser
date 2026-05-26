"""Optional post-processing and writers: augmentation, debug overlays, PNG/SQLite.

``cv2`` (opencv) is imported lazily inside the functions that need it, so importing
this module -- and the generator package -- never requires opencv unless you actually
use ``--augment`` or ``--overlay``.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY, source TEXT, source_ref TEXT,
    label1 TEXT, label2 TEXT, geometry TEXT, meta TEXT,
    img BLOB, mime_type TEXT, width INTEGER, height INTEGER
);
"""


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add JPEG/noise/blur realism only -- no spatial transforms, so labels stay exact."""
    import cv2  # pylint: disable=import-outside-toplevel
    out = image
    if rng.random() < 0.6:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rng.random() < 0.4:
        noise = rng.normal(0, rng.uniform(3, 12), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def make_overlay(image: np.ndarray, label: dict) -> np.ndarray:
    """Draw ground-truth boxes/points/ticks onto the image to verify pixel accuracy."""
    import cv2  # pylint: disable=import-outside-toplevel
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    geo = label.get("geometry")
    if geo is None:
        return out
    horizontal = geo.get("orientation") == "h"
    for it in geo["items"]:
        if "bbox_px" in it:
            x0, y0, x1, y1 = (int(round(v)) for v in it["bbox_px"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        else:
            x, y = (int(round(v)) for v in it["point_px"])
            cv2.circle(out, (x, y), 4, (0, 255, 0), -1)
    pa = [int(round(v)) for v in geo["plot_area_px"]]
    for tick in geo["value_ticks"]:
        c = int(round(tick["px"]))
        if horizontal:
            cv2.line(out, (c, pa[3] - 6), (c, pa[3] + 6), (0, 0, 255), 2)
        else:
            cv2.line(out, (pa[0] - 6, c), (pa[0] + 6, c), (0, 0, 255), 2)
    cv2.rectangle(out, (pa[0], pa[1]), (pa[2], pa[3]), (255, 0, 0), 1)
    return out


def write_overlay(path, image: np.ndarray, label: dict) -> None:
    """Render the debug overlay for one chart and write it to ``path``."""
    import cv2  # pylint: disable=import-outside-toplevel
    cv2.imwrite(str(path), make_overlay(image, label))


def png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes (for the SQLite ``img`` BLOB)."""
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()
