

from pathlib import Path

import numpy as np
from sci_fi_parser.vlm.vlm_config import VLMProfile

from  sci_fi_parser.accuracy.benchmark import _parse_args, _print_summary, _resolve_profile, _run_extractor, _write_results_json, aggregate, load_truth, build_extractor, write_html



def run_benchmark(*, data: Path, out: Path, extractor_name: str = "noisy-oracle",
                  profile: VLMProfile | None = None,
                  seed: int = 0, limit: int | None = None,
                  print_summary: bool = True ) -> dict:
    """End-to-end run: load truth, score, write report.html + results.json.

    Returns the aggregate dict. Public entry point so other tools (e.g. the
    cross-model comparison runner) can drive it without going through argparse.
    """
    truth = load_truth(data)
    images = sorted(truth)[:limit] if limit else sorted(truth)
    rng = np.random.default_rng(seed)
    extractor = build_extractor(extractor_name, truth, rng, profile=profile)
    img_dir = data / "images"

    results = _run_extractor(extractor, truth, images, img_dir)
    agg = aggregate(results)
    out.mkdir(parents=True, exist_ok=True)
    _write_results_json(out, extractor, agg, results)
    write_html(out / "report.html", extractor.name, agg, results, img_dir)
    if print_summary:
        _print_summary(extractor, agg, results, out)
    return agg




def main() -> None:
    args = _parse_args()
    profile = _resolve_profile(args.vlm_config)
    run_benchmark(
        data=args.data, out=args.out,
        extractor_name=args.extractor,
        profile=profile,
        seed=args.seed, limit=args.limit,
    )


if __name__ == "__main__":
    main()
