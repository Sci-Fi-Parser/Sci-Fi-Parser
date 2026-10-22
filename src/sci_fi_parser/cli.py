import argparse
from pathlib import Path

from sci_fi_parser.api import parse_folder


def cli_parse_folder():
    parser = argparse.ArgumentParser(
        prog="scifi-parser",
        description="Extract chart data from PDFs.",
    )

    parser.add_argument(
        "input",
        help="PDF file or directory of PDFs",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory",
    )

    parser.add_argument(
        "--no-classify",
        action="store_true",
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
    )

    parser.add_argument(
        "--no-vlm",
        action="store_true",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    kwargs = {
        "output_dir": args.output,
        "classify": not args.no_classify,
        "ocr": not args.no_ocr,
        "vlm": not args.no_vlm,
    }

    result = parse_folder(
        input_path,
        **kwargs,
    )

    print("\nFinished.")
    print(result.summary())
