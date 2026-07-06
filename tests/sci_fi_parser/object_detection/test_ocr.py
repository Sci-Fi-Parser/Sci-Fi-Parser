from unittest.mock import MagicMock, patch

from sci_fi_parser.object_detection.ocr import Ocr


@patch("sci_fi_parser.object_detection.ocr.PaddleOCR")
def test_constructor_passes_expected_kwargs(mock_paddleocr):
    Ocr("some/path.png")
    mock_paddleocr.assert_called_once_with(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )


@patch("sci_fi_parser.object_detection.ocr.PaddleOCR")
def test_run_ocr_returns_extracted(mock_paddleocr_cls):
    fake_result = MagicMock()
    fake_result.get.side_effect = lambda k: {
        "rec_texts": ["hello"],
        "rec_scores": [0.99],
        "rec_boxes": [[0, 0, 10, 10]],
    }.get(k)

    mock_paddleocr_cls.return_value.predict.return_value = [fake_result]

    ocr = Ocr("some/path.png")
    result = ocr.run_ocr()

    assert result == {
        "labels": ["hello"],
        "confidence": [0.99],
        "bbox": [[0, 0, 10, 10]],
    }


@patch("sci_fi_parser.object_detection.ocr.PaddleOCR")
def test_read_image_works(_):
    ocr = Ocr()
    path = "some/path.png"
    ocr.read_image(path)

    assert ocr.input_path == path
