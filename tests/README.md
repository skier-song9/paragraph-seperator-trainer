# OpenAI preprocessing comparison test

Reusable preprocessing code lives under `src/sermon_pipeline`.

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Prepare payloads without calling OpenAI:

```bash
PYTHONPATH=src python3 tests/openai_preprocessing_comparison.py --dry-run --max-sentences 16
```

Run the OpenAI smoke comparison:

```bash
PYTHONPATH=src python3 tests/openai_preprocessing_comparison.py --max-sentences 16
```

The script loads `OPENAI_API_KEY` from `.env` when the variable is not already present.
It writes payloads, raw responses, parsed annotations, training rows, `run_summary.json`, and `comparison.md` under `tests/results/<timestamp>/`.
