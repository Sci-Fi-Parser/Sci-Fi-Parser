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

    assert result.schema_version == "ocr-token-v1"
    assert len(result.tokens) == 1
    assert result.tokens[0].original_text == "hello"
    assert result.tokens[0].confidence == 0.99
    assert result.tokens[0].box.model_dump() == {
        "left": 0.0,
        "top": 0.0,
        "right": 10.0,
        "bottom": 10.0,
    }


@patch("sci_fi_parser.object_detection.ocr.PaddleOCR")
def test_read_image_works(_):
    ocr = Ocr()
    path = "some/path.png"
    ocr.read_image(path)

    assert ocr.input_data == path


@patch("sci_fi_parser.object_detection.ocr.PaddleOCR")
def test_run_ocr_combines_multiple_paddle_results(mock_paddleocr_cls):
    first = MagicMock()
    first.get.side_effect = lambda key: {
        "rec_texts": ["10"],
        "rec_scores": [0.9],
        "rec_boxes": [[8, 2, 2, 6]],
    }.get(key)
    second = MagicMock()
    second.get.side_effect = lambda key: {
        "rec_texts": ["20"],
        "rec_scores": [0.8],
        "rec_boxes": [[10, 12, 20, 18]],
    }.get(key)
    mock_paddleocr_cls.return_value.predict.return_value = [first, second]

    result = Ocr("some/path.png").run_ocr()

    assert [token.token_id for token in result.tokens] == ["ocr-0", "ocr-1"]
    assert [token.original_text for token in result.tokens] == ["10", "20"]
    assert result.tokens[0].box.left == 2
