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


## 25.5.2026

### Worked On
Started working on creating reliable and accurate training data using matplotlib library. The script can create various bar charts, everything is tunable and a variety of default options to choose from eases creation. The data from each image is stored in a json and SQLite. 

### Learned
We can most likely create all the graphs we need from matplotlib, but fine tuning might be the difficult part. Using Claude, chatgpt will save us time. tbh 

### Problems

At some point we may need to start fine tuning to understand underlying issues in some bar charts. 

### Next
Create a proper benchmarking system that outputs data in the right format. Then Start running various local VLMs through it to test accuracy.


## 26.5.2026

### Worked On
Moved the synthetic generator and the benchmark out of `scripts/` into an importable package, `src/sci_fi_parser/accuracy/`, so the pipeline can `import` them directly instead of shelling out. `scripts/synthetic_bars.py` and `scripts/benchmark.py` are now thin shims that just call into the package, so every documented `.venv/bin/python scripts/…` command keeps working. The generator was also split into smaller modules (`config`, `style`, `render`, `generate`, `output`, `cli`) which makes pylint happy and the next person's life easier when we tweak rendering. Runtime deps (matplotlib, numpy, pillow, pydantic) are now declared in `pyproject.toml`; `opencv-python` and `ollama` are optional extras.

### Learned
RNG draw order is load-bearing: the synthetic dataset is reproducible by checksum (seed 7 with the test config gives a specific `labels.jsonl` hash), so any reorder of `rng.*` calls inside the generator silently breaks every saved baseline. Keep that in mind before any refactor that touches `generate.py` / `style.py`.

### Problems
`uv` wasn't installed on this machine, so `uv.lock` had to be regenerated after adding deps — easy to forget, and CI silently breaks if the lockfile is stale.

### Next
Wire `sci_fi_parser.accuracy` into the real pipeline (currently still only invoked via the shims), add a `tests/` import-smoke test so CI catches package breakage, and start running real local VLMs through the benchmark.
