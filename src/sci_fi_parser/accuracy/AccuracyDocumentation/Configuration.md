# Configuration (gentle version)

> Senior version: [../Configuration.md](../Configuration.md).

We have **two config files**, both at the top of the repo in the `config/`
folder. They're TOML files — a config format that looks a bit like INI
but is more standard.

| File | Controls |
|---|---|
| `config/synthetic_bars.toml` | The chart **generator** — what kind of charts to draw, how many, what they look like. |
| `config/vlm.toml` | The VLM **extractor** — which model to use, what prompt to send it, ollama tuning. |

You can edit these freely. None of the Python code needs to change when
you just want different settings.

## Where the defaults live

Each TOML mirrors a Python dataclass:

- `config/synthetic_bars.toml` ↔ `GenConfig` in `synthetic/config.py`
- `config/vlm.toml` ↔ `VLMProfile` in `vlm_config.py`

The dataclass holds the **defaults**. The TOML holds **overrides**. If a
field isn't in the TOML, the dataclass default is used.

Why two layers? So you don't have to write every setting every time. If
you only care about changing the model name, your TOML can be three lines.

## `config/synthetic_bars.toml` — the chart generator

This file controls what `synthetic-bars` produces. Open the file alongside
this section.

### Picking what to generate

| Key | Default | What it does |
|---|---|---|
| `output_types` | `["simple"]` | Which kinds of charts to draw. Pick from this menu: `simple`, `multicolor`, `grouped`, `stacked`, `horizontal`, `line`, `2-line`, `3-line`, `multiline`. Each one is a different chart style. |
| `per_type` | `1` | How many independent "styles" to draw per chart type. With `per_type=1`, you get one set of charts per type. With `per_type=3`, you get three sets, each with different colours / fonts / fig sizes. |
| `density_steps` | `[4, 5, 7, 9, 10, 12, 15, 20, 25, 30]` | How many bars per chart. The generator walks this list — one chart with 4 bars, one with 5, ... one with 30. Tighter at the low end where extractors start to struggle. |
| `resolutions` | `[480]` | Output image heights, in pixels. Each chart is also rendered at every resolution in this list. Lower resolution = blurrier = harder for extractors. |

**Total image count:**
```
len(output_types) × per_type × len(density_steps) × 2 (off/on twin) × len(resolutions)
```

Why the `× 2`? Because every chart is drawn twice — once **with** value
labels printed on the bars, once **without**. That gives us a paired
comparison: "how much better does the extractor do when the values are
written on the chart?"

### Settings that randomise per series (but stay fixed *within* a series)

These get sampled once when a "series" starts and then stay constant. So
all 10 charts in a density series share the same colours, fonts, figure
size, etc.

| Key | Default | What it does |
|---|---|---|
| `n_series` | `[2, 4]` | For "multi" charts (grouped / stacked / multiline), pick a random number of series in this range. |
| `geometry_full_prob` | `0.5` | Probability (0..1) of including pixel-level ground truth in `labels.jsonl`. The other half get a smaller label record without pixel coordinates. |
| `fig_w_in` | `[5.0, 9.0]` | Figure width in inches. Sampled uniformly. |
| `fig_h_in` | `[3.5, 5.5]` | Figure height in inches. |
| `dpi_choices` | `[90, 100, 120, 150]` | Resolution to use when `resolutions` isn't set (e.g. preview mode). When `resolutions` is set, DPI is computed automatically to hit the requested pixel height. |
| `bar_width` | `[0.6, 0.85]` | Bar width as a fraction of the space between categories. Lower = thinner bars. |
| `allow_negative` | `0.15` | Probability of letting the y-axis go below zero. Skipped for stacked charts (they need positive totals). |
| `grid_prob` | `0.5` | Probability of drawing gridlines. |
| `title_prob` | `0.7` | Probability of adding a chart title. |
| `label_rotation` | `90` | Rotation angle of category labels. 90 = vertical. |
| `tick_fontsize` | `[6, 11]` | Font size range for tick labels. Smaller end is used when there are lots of bars (so they fit). |

### `--random` mode only

| Key | Default | What it does |
|---|---|---|
| `value_labels_prob` | `0.0` | In `--random N` mode, probability that each random chart has value labels on the bars. Ignored in the default density-series mode (which always emits off/on pairs). |

### What happens if you typo a key

If you write `output_typse = ["simple"]` (typo), the generator **crashes
immediately** with a clear error like:

```
unknown config key 'output_typse' in config/synthetic_bars.toml;
valid: ['allow_negative', 'bar_width', 'density_steps', ...]
```

That's intentional. Silent fallthrough — where the typo is just ignored
and the default is used — is the worst kind of bug. You'd think your
setting was active when it wasn't.

## `config/vlm.toml` — the VLM extractor

This file tells the benchmark how to talk to the VLM. Open it alongside.

| Key | Default | What it does |
|---|---|---|
| `model` | `"qwen2.5vl:7b"` | The ollama model tag. Must already be pulled (`ollama pull qwen2.5vl:7b`). |
| `prompt` | the canonical default (see file) | The instruction sent with every image. |
| `num_ctx` | `2048` | Context window size in tokens. Bigger = more memory + slower, but room for longer JSON replies on dense charts. |
| `num_gpu` | unset | If set, request partial GPU offload. Ollama may still fall back to CPU if VRAM isn't enough. |

### Other models you might try

The TOML has commented-out alternatives. Knowing roughly what each costs
helps:

| Tag | Speed (CPU) | Notes |
|---|---|---|
| `qwen2.5vl:7b` (default) | baseline | Q4 quantisation. Fits in mid-range VRAM cards. |
| `qwen2.5vl:7b-q8_0` | ~80–100 s/chart | Higher precision (8-bit). Try if you suspect quantisation hurts accuracy. |
| `qwen2.5vl:7b-fp16` | ~150–200 s/chart | Full precision. For reference runs only. |
| `qwen2.5vl:3b` | fastest | Smaller. Weak on abbreviated y-axes (`200K`, `1.5M`). |

### How the prompt is constructed

The default prompt is in `vlm_config.py:DEFAULT_PROMPT`. It tells the
model things like:

- "Use the x-axis labels exactly as printed"
- "If there's no legend, use the y-axis title as the series name"
- "`200K` means 200000, `1.5M` means 1500000"

If you find the VLM systematically getting something wrong, the prompt is
the first place to look. Edit `prompt = """..."""` in `config/vlm.toml`.

## Precedence — when settings disagree, who wins?

This is the part that often confuses people. Settings can come from
multiple places (the dataclass default, the TOML, a CLI flag, an env
variable). When two sources disagree, here's who wins.

### Synthetic generator

```
CLI flag  >  TOML file (--config)  >  GenConfig default
```

For example: if you run `synthetic-bars --seed 42 --config config/synthetic_bars.toml`,
the `--seed 42` wins over anything in the TOML.

### VLM model name

```
--extractor ollama:<tag>    (one-shot override; affects only the model name)
       ↓ otherwise…
--vlm-config <path>         (load a different TOML file)
       ↓ otherwise…
config/vlm.toml in the cwd  (auto-loaded if present)
       ↓ otherwise…
DEFAULT_PROFILE             (built-in default)
```

So `benchmark --data ... --extractor ollama:qwen2.5vl:3b` swaps the model
just for this one run, without editing any file.

### VLM ollama options

| Field | Order |
|---|---|
| `num_ctx` | env `BENCH_NUM_CTX` > profile's `num_ctx` |
| `num_gpu` | env `BENCH_NUM_GPU` > profile's `num_gpu` (omitted if neither set) |

Env vars exist for quick experiments without editing the TOML. Example:

```bash
BENCH_NUM_CTX=4096 benchmark --data ... --extractor ollama
```

## Cookbook — what to edit when you want to…

| What you want | What to change |
|---|---|
| Generate a different mix of chart types | `output_types` in `config/synthetic_bars.toml` |
| Try smaller / bigger images | `resolutions` |
| More or fewer bars per chart | `density_steps` |
| Generate a bigger dataset | bump `per_type` (e.g. to 3 or 5) |
| Permanently switch VLM model | edit `model` in `config/vlm.toml` |
| One-time try a different model | `benchmark --extractor ollama:<tag>` |
| A/B test two profiles | make `config/vlm_a.toml` and `config/vlm_b.toml`, run twice with `--vlm-config <path>` |
| Bump ollama context just once | `BENCH_NUM_CTX=4096 benchmark ...` |
| Edit what the VLM is told | `prompt = """..."""` in `config/vlm.toml` |
| Reproduce a specific dataset | `synthetic-bars --seed <same-number> --config <same-toml>` |

## Common confusions

- **"Why doesn't my change take effect?"** Either there's a flag/env that
  overrides it (check the precedence above), or you have a typo in the
  TOML key (which would crash — re-read the error). If neither, you
  might be editing the wrong file (`vlm.toml` vs `synthetic_bars.toml`).
- **"Why does the same seed give me a different dataset?"** The dataset
  is reproducible only if you use the **same config too**. The seed
  controls the random draws; the config decides which draws happen.
- **"Why are there twice as many images as I expected?"** Each chart is
  drawn twice (with and without value labels). That's by design.

## Sources

- `src/sci_fi_parser/accuracy/synthetic/config.py`, `style.py`, `cli.py`
- `src/sci_fi_parser/accuracy/vlm_config.py`, `vlm.py`
- `src/sci_fi_parser/accuracy/benchmark.py` (the `_resolve_profile` function)
- `config/synthetic_bars.toml`, `config/vlm.toml`
