# Scientific Figure Parser

[![CI](https://github.com/Sci-Fi-Parser/Sci-Fi-Parser/actions/workflows/ci.yaml/badge.svg)](https://github.com/Sci-Fi-Parser/Sci-Fi-Parser/actions/workflows/ci.yaml)

Sci-Fi-Parser is a Python package that provides a pipeline to take charts from PDF documents and extract data from them.

## Installation

```bash
$ pip install sci-fi-parser
```

## Usage

For full api usage, consult the documentation.

```python
from sci_fi_parser import parse_folder

result = parse_folder(
    "path/to/pdfs",
    output_dir="output",
    extracted_image_dir="temp/extracted_images",
    vlm_config="config/vlm.toml",
)

print(result.summary())
```

## License

Sci-Fi-Parser is licensed under the terms of the [MIT license](LICENSE).
