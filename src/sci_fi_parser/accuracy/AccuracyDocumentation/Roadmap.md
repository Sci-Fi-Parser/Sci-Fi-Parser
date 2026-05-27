# Roadmap (gentle version)

> Senior version: [../Roadmap.md](../Roadmap.md).

This file is the counterpart to [Decisions.md](Decisions.md). Decisions
are things we've already settled. Roadmap items are things in flight,
open questions, and rough edges we've noticed but haven't fixed.

## What's currently half-built

There are three working-tree changes on this branch that *aren't*
committed yet. Together they prepare a multi-model comparison runner.

### What is the "comparison runner"?

Today, `benchmark` runs **one** extractor on a dataset. A "comparison
runner" would run **several** models on the same dataset and produce a
side-by-side report (e.g. "qwen 7b vs qwen 3b vs llava — which is more
accurate, which is fastest?").

It would be called `benchmark-compare`. The code isn't written yet, but
the tests are — they pin the API in detail.

### The uncommitted changes

1. **`pyproject.toml`** — registers a new CLI entry:
   ```
   benchmark-compare = "sci_fi_parser.accuracy.vlm_compare:main"
   ```
   But the `vlm_compare` module doesn't exist yet, so the entry doesn't
   work.

2. **`benchmark.py` refactor** — the body of `main()` is moved into a
   reusable function:
   ```python
   def run_benchmark(*, data, out, extractor_name, profile,
                     seed, limit, print_summary) -> dict:
       ...
   ```
   `main()` is now a thin wrapper. This is so the future comparison
   runner can call `run_benchmark(...)` N times (once per model) without
   shelling out.

3. **`tests/test_smoke.py`** — five new tests that import
   `sci_fi_parser.accuracy.vlm_compare` and exercise its API. Right
   now, running `pytest` would fail at import for these tests, because
   the module doesn't exist.

## What `vlm_compare` should look like (from the tests)

The tests pin a fairly detailed API. Reading them gives us the spec.

### Public functions

```python
def main() -> None: ...
def load_comparison_config(path) -> tuple[RunConfig, list[Entry]]: ...
def resolve_pull_mode(missing, cli_mode, yes) -> str: ...
def format_preflight(statuses, mode) -> str: ...
```

### Data classes

```python
@dataclass
class RunConfig:
    data: Path | None
    out:  Path | None
    limit: int | None
    seed:  int | None
    pull:  str | None        # one of "prefetch" | "circular" | "skip" | None

@dataclass
class Entry:
    name:    str             # display label in the comparison report
    profile: VLMProfile      # reused from vlm_config.py

@dataclass
class ModelStatus:
    tag: str
    local: bool
    size_bytes: int | None
```

### The comparison TOML

```toml
# config/vlm_compare.toml (example)

extends = "config/vlm.toml"    # optional — start from a base VLMProfile

[run]                          # optional — like CLI flags but in the file
data  = "train_data/synthetic"
out   = "reports/compare"
limit = 50
seed  = 0
pull  = "circular"             # "prefetch" | "circular" | "skip"

[defaults]                     # optional — applied on top of extends
num_ctx = 4096
prompt  = "..."

[[model]]                      # one or more — each becomes an Entry
name  = "7b-q4"
model = "qwen2.5vl:7b"

[[model]]
name    = "7b-q8"
model   = "qwen2.5vl:7b-q8_0"
num_ctx = 8192                 # per-model override wins over [defaults]
```

Merge order, from lowest to highest priority:
```
extends   <   [defaults]   <   per-model fields
```

### Pull modes (from the test name)

The "pull mode" decides what to do when some models aren't downloaded
locally yet.

| Mode | What it does (best guess from name + tests) |
|---|---|
| `prefetch` | Pull every missing model up front, then run all of them. |
| `circular` | Pull one, run it, delete it, pull the next. Bounded disk usage. |
| `skip` | Don't pull anything. Only run models that are already local. |

The `resolve_pull_mode` function decides which mode to use based on
CLI flags and the `[run].pull` table value. There's an explicit
"footgun" guard: if the user passed `--yes` (auto-confirm) but didn't
choose a `--pull` mode, and some models are missing, the function
raises `SystemExit` — forcing the user to pick consciously.

### The preflight output

Before running, the tool prints a preflight table:

```
LOCAL:
  qwen2.5vl:7b     (5.0 GB)
MISSING (pull required):
  qwen2.5vl:7b-q8_0   ( ?  )

local total: 5.0 GB
free disk:   42.1 GB
mode: circular
```

(The exact format is what `format_preflight` produces — tests pin some
substrings but not the full layout.)

## Open questions

The tests pin the API but not all of the design. We should answer
these before writing more code:

1. **What does "circular" pull mode actually do?** Best guess: pull,
   run, delete, repeat. We should confirm before implementing.

2. **What's the default pull mode?** Tests don't say. Probably
   `prefetch` (predictable failure modes — fail early if disk is
   tight), with `circular` as the disk-bound escape hatch.

3. **In-process or subprocess per model?** The `run_benchmark`
   refactor strongly implies in-process — call it N times. Risk: state
   leakage between models (matplotlib already-locked-to-Agg helps;
   ollama HTTP client is fine). Worth a test that two consecutive
   in-process runs produce identical reports to two single-shot runs.

4. **Report shape.** Two options:
   - (a) **N independent reports + an index page** — easy to do
     (reuse `write_html` N times, write an `index.html` linking them).
   - (b) **One side-by-side diff report** — much more useful, but
     means designing a new template.
   Researcher preference needed.

5. **How do we talk to ollama for pull/list/show?** Either shell out
   (no extra dep, simpler) or use the existing Python `ollama`
   package (better error surfaces, already a dep). Pull progress bars
   matter for UX.

6. **Disk-space check.** `format_preflight` shows free disk. We should
   pick a threshold (say, 2× expected pull size) and refuse to start
   if it's tight.

7. **CLI vs `[run]` precedence.** Tests use the `[run]` table values.
   Implication: CLI overrides `[run]`. Document this.

## Code smells noticed during the doc pass

Pure observations. Not prescriptions — triage as a group before
fixing.

- **`_align_series_names` only handles single-series ↔ single-series.**
  Multi-series charts where the VLM mislabels series get scored as
  all-miss. May be the right call (see Decisions #14), but worth
  flagging.

- **`parse_chartdata`'s code-fence strip is heuristic.** Works on
  cases we've seen; a model that emits two top-level JSON objects
  in one reply would silently lose the second.

- **`NoisyOracle` copies `chart_type` straight from truth.** That
  means it has 100% type accuracy by construction — fine for a
  sanity check, but worth documenting alongside its metric.

- **`benchmark.py:_run_extractor` catches `Exception` and continues.**
  Correct behaviour for batch runs, but the failed image becomes a
  silent zero contributor. Consider counting exceptions and showing
  the count in the summary.

- **`build_extractor` raises `SystemExit`.** Inside a library, that's
  a bit aggressive — `ValueError` would be more appropriate. The CLI
  layer can translate it to `SystemExit`. Mechanical change; touches
  `_resolve_profile` too.

- **Smoke tests duplicate `subprocess.run` boilerplate.** Could be
  factored into one helper. Cosmetic.

## Stale notes elsewhere

A couple of things in the rest of the docs that don't match reality
anymore:

- `Dev Documentantion/Tools/TrainingData.md` mentions
  `scripts/synthetic_bars.py` shims. They don't exist anymore — the
  package moved and the CLIs are installed via `[project.scripts]`
  only. Update next time someone touches the diary.

- The model-comparison comments in `config/vlm.toml` (model speeds,
  etc.) are tribal knowledge. They're now also in the dev docs
  ([Configuration.md](Configuration.md)) — eventually the TOML can be
  trimmed to just the active settings.

## What I'd build next, in order

For discussion, not a commitment:

1. **Decide on pull-mode semantics and report shape.** Both are
   user-facing and locked-in once code lands. Don't skip this.

2. **Implement `vlm_compare`** to make the existing tests pass:
   - `load_comparison_config`
   - `resolve_pull_mode`
   - `format_preflight`
   - the runner itself (loop calling `run_benchmark`)
   - the report assembler (per the decision in #1 above)

3. **Wire `accuracy` into `sci_fi_parser/main.py`.** Right now
   `main.py` is a `print("Hello scifi_parser!")` stub. The
   `Dev Documentantion/Tools/TrainingData.md` 26.5.2026 entry already
   flagged this as next work.

4. **First real CV+OCR extractor**, following the pattern in
   [Extending.md](Extending.md). Probably PaddleOCR + a bar detector.
   Notes already exist in `Dev Documentantion/Data extraction/OCR.md`.

5. **A type-only classifier extractor.** Returns just `chart_type`
   (and `series=[]`). Lets us benchmark "type accuracy" in isolation,
   independent of value extraction.

## Sources

- Uncommitted working-tree diff (`pyproject.toml`,
  `accuracy/benchmark.py`, `tests/test_smoke.py`)
- `tests/test_smoke.py::test_vlm_comparison_*` (the spec for the
  half-built runner)
- `src/sci_fi_parser/accuracy/benchmark.py:run_benchmark`
- `Dev Documentantion/Tools/TrainingData.md`,
  `Dev Documentantion/Data extraction/OCR.md`
