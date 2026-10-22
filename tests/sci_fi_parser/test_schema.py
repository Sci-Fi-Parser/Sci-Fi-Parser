from pathlib import Path

import sci_fi_parser.schema as schema


def _mock_jsonl_file(mocker, lines: list[str], exists: bool = True):
    """Patch Path.exists/Path.open so reading `path` yields `lines`.

    `lines` should be the raw text lines (no trailing newline needed).
    """
    mocker.patch.object(Path, "exists", return_value=exists)

    file_content = "\n".join(lines) + ("\n" if lines else "")
    mock_open = mocker.mock_open(read_data=file_content)
    mocker.patch.object(Path, "open", mock_open)

    return mock_open


class TestImageSetFromJsonl:
    def test_missing_file_returns_empty_image_set(self, mocker):
        mocker.patch.object(Path, "exists", return_value=False)
        mock_open = mocker.patch.object(Path, "open")

        image_set = schema.ImageSet.from_jsonl("does/not/exist.jsonl")

        assert len(image_set) == 0
        mock_open.assert_not_called()

    def test_loads_single_record(self, mocker):
        line = (
            '{"image_id": "img-1", "metadata": {"extraction": {}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}, '
            '"output": {"path": "img-1.png"}, '
            '"classification": {"result": "", "raw": {}}, '
            '"ocrcv": {"result": "", "raw": {}}, '
            '"vlm": {"result": {}, "raw": {}}}'
        )
        _mock_jsonl_file(mocker, [line])

        image_set = schema.ImageSet.from_jsonl("raw/image_set.jsonl")

        assert len(image_set) == 1
        assert list(image_set) == ["img-1"]
        assert image_set.get("img-1")["output"]["path"] == "img-1.png"

    def test_loads_multiple_records(self, mocker):
        lines = [
            '{"image_id": "img-1", "metadata": {}, "output": {"path": "a.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}',
            '{"image_id": "img-2", "metadata": {}, "output": {"path": "b.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}',
        ]
        _mock_jsonl_file(mocker, lines)

        image_set = schema.ImageSet.from_jsonl("raw/image_set.jsonl")

        assert len(image_set) == 2
        assert set(image_set) == {"img-1", "img-2"}

    def test_skips_blank_lines(self, mocker):
        lines = [
            '{"image_id": "img-1", "metadata": {}, "output": {"path": "a.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}',
            "",
            "   ",
            '{"image_id": "img-2", "metadata": {}, "output": {"path": "b.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}',
        ]
        _mock_jsonl_file(mocker, lines)

        image_set = schema.ImageSet.from_jsonl("raw/image_set.jsonl")

        assert len(image_set) == 2

    def test_empty_file_returns_empty_image_set(self, mocker):
        _mock_jsonl_file(mocker, [])

        image_set = schema.ImageSet.from_jsonl("raw/image_set.jsonl")

        assert len(image_set) == 0

    def test_later_duplicate_image_id_overwrites_earlier_one(self, mocker):
        lines = [
            '{"image_id": "img-1", "metadata": {}, "output": {"path": "old.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}',
            '{"image_id": "img-1", "metadata": {}, "output": {"path": "new.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}',
        ]
        _mock_jsonl_file(mocker, lines)

        image_set = schema.ImageSet.from_jsonl("raw/image_set.jsonl")

        assert len(image_set) == 1
        assert image_set.get("img-1")["output"]["path"] == "new.png"

    def test_accepts_path_object_as_well_as_str(self, mocker):
        line = (
            '{"image_id": "img-1", "metadata": {}, "output": {"path": "a.png"}, '
            '"classification": {}, "ocrcv": {}, "vlm": {}}'
        )
        _mock_jsonl_file(mocker, [line])

        image_set = schema.ImageSet.from_jsonl(Path("raw/image_set.jsonl"))

        assert len(image_set) == 1


class TestPdfSetFromJsonl:
    def test_missing_file_returns_empty_pdf_set(self, mocker):
        mocker.patch.object(Path, "exists", return_value=False)
        mock_open = mocker.patch.object(Path, "open")

        pdf_set = schema.PdfSet.from_jsonl("does/not/exist.jsonl")

        assert len(pdf_set) == 0
        mock_open.assert_not_called()

    def test_loads_single_record(self, mocker):
        line = '{"pdf_id": "3f9a...c21", "metadata": {"file_name": "Letter.pdf", "page_count": "11"}}'
        _mock_jsonl_file(mocker, [line])

        pdf_set = schema.PdfSet.from_jsonl("raw/pdf_set.jsonl")

        assert len(pdf_set) == 1
        assert pdf_set.get("3f9a...c21")["metadata"]["file_name"] == "Letter.pdf"

    def test_loads_multiple_records(self, mocker):
        lines = [
            '{"pdf_id": "hash-1", "metadata": {"file_name": "a.pdf", "page_count": "1"}}',
            '{"pdf_id": "hash-2", "metadata": {"file_name": "b.pdf", "page_count": "2"}}',
        ]
        _mock_jsonl_file(mocker, lines)

        pdf_set = schema.PdfSet.from_jsonl("raw/pdf_set.jsonl")

        assert len(pdf_set) == 2
        assert {pdf_id for pdf_id, _ in pdf_set.items()} == {"hash-1", "hash-2"}

    def test_skips_blank_lines(self, mocker):
        lines = [
            '{"pdf_id": "hash-1", "metadata": {"file_name": "a.pdf", "page_count": "1"}}',
            "",
            '{"pdf_id": "hash-2", "metadata": {"file_name": "b.pdf", "page_count": "2"}}',
        ]
        _mock_jsonl_file(mocker, lines)

        pdf_set = schema.PdfSet.from_jsonl("raw/pdf_set.jsonl")

        assert len(pdf_set) == 2

    def test_empty_file_returns_empty_pdf_set(self, mocker):
        _mock_jsonl_file(mocker, [])

        pdf_set = schema.PdfSet.from_jsonl("raw/pdf_set.jsonl")

        assert len(pdf_set) == 0

    def test_accepts_path_object_as_well_as_str(self, mocker):
        line = '{"pdf_id": "hash-1", "metadata": {"file_name": "a.pdf", "page_count": "1"}}'
        _mock_jsonl_file(mocker, [line])

        pdf_set = schema.PdfSet.from_jsonl(Path("raw/pdf_set.jsonl"))

        assert len(pdf_set) == 1
