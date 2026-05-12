"""CLI entry point for chartparse."""

import argparse
import logging
import sys
from pathlib import Path


def cmd_parse(args: argparse.Namespace) -> None:
    from chartparse import extract_charts

    result = extract_charts(
        args.pdf,
        output_dir=args.output_dir,
        confidence_threshold=args.threshold,
    )

    print(f"\nFile : {Path(args.pdf).name}")
    print(f"Found: {result.total_figures} figures, {result.charts_found} charts\n")

    for f in result.figures:
        tag = "CHART" if f.is_chart else "     "
        print(f"[{tag}]  page {f.page:>2}  {f.confidence:>5.1%}  {f.label}")
        if args.verbose and f.is_chart:
            print(f"         → {f.image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chartparse",
        description="Extract and classify figures in a PDF",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show image paths for charts"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="Classify figures in a single PDF")
    p.add_argument("pdf", help="Path to the PDF file")
    p.add_argument("--output-dir", default=None, help="Directory to save extracted images")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Minimum confidence to include a figure (default 0.0)")
    p.set_defaults(func=cmd_parse)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
