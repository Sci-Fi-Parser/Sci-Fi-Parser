# Team Diary Guidelines

Write here information that you might think is relevant for your colleagues to know about. Do so as you personally prefer; bullet points are fine, summarized versions are fine, no pressure. You may leave spots empty as you please. This is meant for us, the bar is low. You may edit the format if you feel like it needs improvement 

Format:
``` 
## DD-MM-YYYY

### Worked On

### Learned

### Problems

### Next

## DD-MM-YYYY

### Worked On
.....
```

## 28-05-2026

### Worked On
- **VLM extractor moved out of `accuracy/`.** It now lives in a sibling
  top-level package, `src/sci_fi_parser/vlm/`. New canonical imports:
  - `from sci_fi_parser.vlm.vlm import OllamaVLM`
  - `from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile`

  Reason: the model client is independent from the measurement machinery.
  Pipeline / one-off scripts shouldn't have to import from `accuracy`
  just to talk to ollama.
- Wired up `vlm/pipeline.py` as the VLM stage of the runtime pipeline.
  It now imports `OCRSet` / `VLMSet` from `data_pipeline.py` (where the
  data containers and the offloader live).
- Updated every import site (`accuracy/benchmark.py`,
  `accuracy/vlm_compare.py`, `data_pipeline.py`, `main.py`,
  `tests/test_smoke.py`) plus the in-package accuracy docs
  (`README.md` and the relevant `AccuracyDocumentation/*.md` files).
  Tests still green, pylint 10/10 outside `cv/`.

### Learned
  - Newer Qwen models  (3.0+) offer better information on the models decision process.
  by using chat fields like thinking and stream, we can see what the model extracts from our input before it answers.

### Problems
- The Ollama daemon must be running before any `--extractor ollama`
  run; otherwise the benchmark prints one `ConnectionError` per chart
  and finishes with 0% recall. Fail-fast preflight still TBD.

### Next
- Naming nit worth a five-minute group decision: `vlm/vlm.py` →
  `vlm/extractor.py` (and `vlm_config.py` → `config.py`) to drop the
  `vlm.vlm` stutter? Or just re-export through `vlm/__init__.py`?
- Re-introduce the `scoring.py` / `report.py` / `leaderboard.py` /
  `ollama_api.py` modular split that existed on the `benchmark` branch
  but didn't reach `dev`.
  
- Wire a fail-fast Ollama-daemon preflight into `accuracy/benchmark.py`,
  mirroring what `vlm_compare.py` already does via `inspect_preflight`.

## 31-05-2026

### Worked On

- added a VLM class for OpenAI chat completions (compatible) API endpoints
- moved misc stuff in the library code to `/accuracy` (maybe later implemented library side) 

### Problems

- specific request format was a bit hard to pin down, the ChatCompletionsVLM class is now based on llama.cpp's expected request format, which should work for similar servers

### Next

- ollama might work with ChatCompletionsVLM as is? So it might be unnecessary
- batched extract?

