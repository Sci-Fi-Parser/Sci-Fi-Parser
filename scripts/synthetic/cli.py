"""Command-line entry point for the synthetic generator.

Run via the installed entry point (created by ``uv sync``)::

    synthetic-bars --config config/synthetic_bars.toml --overlay
    synthetic-bars --preview --out train_data/synthetic
    synthetic-bars --random 200 --out /tmp/synth

or as a module::

    python -m sci_fi_parser.accuracy.synthetic.cli --preview
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CATALOG, GenConfig, load_config
from .generate import generate_preview, generate_random, generate_series
from .output import SQLITE_DDL, augment, png_bytes, write_overlay


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--random",
        type=int,
        default=None,
        metavar="N",
        help="random mode: N fully-random charts (default: density series)",
    )
    ap.add_argument(
        "--preview", action="store_true", help="one sample per catalog type into <out>/preview, then exit"
    )
    ap.add_argument("--out", type=Path, default=Path("train_data/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=None, help="TOML overriding GenConfig")
    ap.add_argument("--augment", action="store_true", help="add JPEG/noise/blur realism")
    ap.add_argument("--overlay", action="store_true", help="write _debug overlays")
    ap.add_argument("--sqlite", action="store_true", help="also write dataset.sqlite3")
    ap.add_argument("--refresh", action="store_true", help="delete prior outputs in --out before generating")
    return ap.parse_args()


def _run_preview(cfg: GenConfig, out: Path) -> None:
    pdir = out / "preview"
    pdir.mkdir(parents=True, exist_ok=True)
    for name, image, _, _ in generate_preview(cfg):
        Image.fromarray(image).save(pdir / name)
    print(f"preview: {len(CATALOG)} type samples -> {pdir}")
    print("  types: " + ", ".join(CATALOG))


def _refresh(out: Path) -> None:
    for d in (out / "images", out / "_debug"):
        if d.exists():
            shutil.rmtree(d)
    for f in (out / "truth.jsonl", out / "dataset.sqlite3"):
        f.unlink(missing_ok=True)
    print(f"refresh: cleared old output in {out}")


def _insert_row(
    con: sqlite3.Connection, seed: int, name: str, truth: dict, metadata: dict, image: np.ndarray
) -> None:
    truth_record = {**truth, "metadata": metadata}
    con.execute(
        "INSERT INTO dataset(source,source_ref,label1,label2,geometry,meta,"
        "img,mime_type,width,height) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "synthetic",
            f"seed{seed}/{name}",
            truth.get("chart_type"),
            json.dumps(truth_record),
            json.dumps(truth.get("geometry")) if truth.get("geometry") else None,
            json.dumps(metadata),
            png_bytes(image),
            "image/png",
            image.shape[1],
            image.shape[0],
        ),
    )


def _open_sqlite(out: Path, enable: bool) -> sqlite3.Connection | None:
    if not enable:
        return None
    con = sqlite3.connect(out / "dataset.sqlite3")
    con.executescript(SQLITE_DDL)
    return con


def _write_dataset(
    stream,
    args: argparse.Namespace,
    img_dir: Path,
    dbg_dir: Path,
    con: sqlite3.Connection | None,
    rng: np.random.Generator,
) -> tuple[dict, int]:
    """Stream samples to images + truth.jsonl (+overlays/SQLite); return counts."""
    counts: dict[str, int] = {}
    n_total = 0
    with (args.out / "truth.jsonl").open("w", encoding="utf-8") as jsonl:
        for name, image, truth, metadata in stream:
            if args.augment:
                image = augment(image, rng)
            Image.fromarray(image).save(img_dir / name)
            jsonl.write(json.dumps({"image": name, **truth, "metadata": metadata}) + "\n")
            preset = metadata["preset"]
            counts[preset] = counts.get(preset, 0) + 1
            n_total += 1
            if args.overlay:
                write_overlay(dbg_dir / f"overlay_{name}", image, truth)
            if con is not None:
                _insert_row(con, args.seed, name, truth, metadata, image)
    return counts, n_total


def _print_summary(
    args: argparse.Namespace, img_dir: Path, dbg_dir: Path, counts: dict, n_total: int
) -> None:
    print(f"generated {n_total} charts -> {img_dir}")
    for preset, c in sorted(counts.items()):
        print(f"  {preset:12s} {c}")
    print(f"truth  -> {args.out / 'truth.jsonl'}")
    if args.overlay:
        print(f"overlays -> {dbg_dir}")
    if args.sqlite:
        print(f"sqlite  -> {args.out / 'dataset.sqlite3'}")


def main() -> None:
    """Generate a synthetic dataset (images + truth.jsonl, optional overlays/SQLite)."""
    args = _parse_args()
    cfg = load_config(args.config) if args.config else GenConfig()
    if args.preview:
        _run_preview(cfg, args.out)
        return
    if args.refresh:
        _refresh(args.out)

    rng = np.random.default_rng(args.seed)
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = args.out / "_debug"
    if args.overlay:
        dbg_dir.mkdir(exist_ok=True)
    con = _open_sqlite(args.out, args.sqlite)

    stream = generate_random(rng, cfg, args.random) if args.random is not None else generate_series(rng, cfg)
    counts, n_total = _write_dataset(stream, args, img_dir, dbg_dir, con, rng)
    if con is not None:
        con.commit()
        con.close()
    _print_summary(args, img_dir, dbg_dir, counts, n_total)


if __name__ == "__main__":
    main()
