def serialize_bbox(bbox):
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
        "right": bbox.right,
        "bottom": bbox.bottom,
    }


def serialize_bar_candidate(bar_candidate):
    return {
        "bbox": serialize_bbox(bar_candidate.bbox)
    }


def serialize_ocr_result(ocr_res):
    return {
        "labels": ocr_res["labels"],
        "confidence": [float(x) for x in ocr_res["confidence"]],
        "bbox": ocr_res["bbox"].tolist(),
    }


def serialize_matched(matched):
    return [
        {
            "bar_candidate": serialize_bar_candidate(bar_candidate),
            "ocr_boxes": [box.tolist() for box in ocr_boxes],
        }
        for bar_candidate, ocr_boxes in matched
    ]