"""Geometric clustering of vector drawing bounding boxes."""

from collections.abc import Iterable

Box = tuple[float, float, float, float]


def _are_neighbors(a: Box, b: Box, x_tolerance: float, y_tolerance: float) -> bool:
    """Check whether two boxes overlap once ``a`` is grown by the tolerances."""
    return (
        a[0] - x_tolerance < b[2]
        and b[0] - x_tolerance < a[2]
        and a[1] - y_tolerance < b[3]
        and b[1] - y_tolerance < a[3]
    )


def cluster_boxes(boxes: Iterable[Box], x_tolerance: float, y_tolerance: float) -> list[Box]:
    """Merge boxes that lie within the tolerances of each other.

    Boxes are merged transitively: growing a cluster can bring further boxes into range,
    so merging repeats until no cluster changes.

    Args:
        boxes: Bounding boxes as ``(left, bottom, right, top)``.
        x_tolerance: Horizontal distance below which boxes are considered neighbors.
        y_tolerance: Vertical distance below which boxes are considered neighbors.

    Returns:
        The merged bounding boxes, in no particular order.
    """
    remaining = sorted(set(boxes))
    changed = True

    while changed:
        changed = False
        clusters: list[Box] = []

        while remaining:
            cluster = remaining.pop()
            merged = True
            while merged:
                merged = False
                rest = []
                for box in remaining:
                    if _are_neighbors(cluster, box, x_tolerance, y_tolerance):
                        cluster = (
                            min(cluster[0], box[0]),
                            min(cluster[1], box[1]),
                            max(cluster[2], box[2]),
                            max(cluster[3], box[3]),
                        )
                        merged = True
                        changed = True
                    else:
                        rest.append(box)
                remaining = rest
            clusters.append(cluster)

        remaining = clusters

    return remaining
