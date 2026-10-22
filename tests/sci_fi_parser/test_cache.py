import sci_fi_parser.cache as cache


def test_in_cache_returns_true_if_in_cache(mocker):
    mocker.patch("sci_fi_parser.cache._hash_pdf", new_callable=mocker.Mock, return_value="abc")

    cache_file = mocker.MagicMock()
    cache_file.__contains__.return_value = True

    assert cache.in_cache(pdf=mocker.Mock(), cache=cache_file)


def test_in_cache_returns_false_if_not_in_cache(mocker):
    mocker.patch("sci_fi_parser.cache._hash_pdf", new_callable=mocker.Mock, return_value="abc")

    mock_cache = mocker.MagicMock()
    mock_cache.__contains__.return_value = False

    assert not cache.in_cache(pdf=mocker.Mock(), cache=mock_cache)


def test_add_to_cache_adds(mocker):
    mocker.patch("sci_fi_parser.cache._hash_pdf", new_callable=mocker.Mock, return_value="abc")
    mock_pdf_path = mocker.Mock()
    mock_pdf_path.name = "pdf_name"

    mock_cache = mocker.MagicMock()

    cache.add_to_cache(pdf_path=mock_pdf_path, cache=mock_cache)

    mock_cache.add.assert_called_with("abc", "pdf_name")
