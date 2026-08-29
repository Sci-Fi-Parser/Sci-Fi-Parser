from sci_fi_parser.image_extraction.vector_clustering import cluster_boxes


def test_empty_input() -> None:
    assert cluster_boxes([], 75.0, 75.0) == []


def test_keeps_disjoint_boxes_apart() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (500.0, 500.0, 510.0, 510.0)]

    assert sorted(cluster_boxes(boxes, 75.0, 75.0)) == sorted(boxes)


def test_merges_boxes_within_tolerance() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]

    assert cluster_boxes(boxes, 75.0, 75.0) == [(0.0, 0.0, 30.0, 30.0)]


def test_tolerance_is_per_axis() -> None:
    """Boxes near enough horizontally but far apart vertically stay separate."""
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 500.0, 30.0, 510.0)]

    assert len(cluster_boxes(boxes, 75.0, 5.0)) == 2


def test_merges_transitively() -> None:
    """A grown cluster reaches boxes that were out of range of the original."""
    boxes = [(0.0, 0.0, 10.0, 10.0), (80.0, 0.0, 90.0, 10.0), (160.0, 0.0, 170.0, 10.0)]

    assert cluster_boxes(boxes, 75.0, 75.0) == [(0.0, 0.0, 170.0, 10.0)]


def test_deduplicates_identical_boxes() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 100

    assert cluster_boxes(boxes, 75.0, 75.0) == [(0.0, 0.0, 10.0, 10.0)]


def test_clustering_is_idempotent() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0), (500.0, 500.0, 510.0, 510.0)]

    once = cluster_boxes(boxes, 75.0, 75.0)

    assert sorted(cluster_boxes(once, 75.0, 75.0)) == sorted(once)
