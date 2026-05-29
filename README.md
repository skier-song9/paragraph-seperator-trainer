# paragraph-seperator-trainer

Source-aware sermon paragraph boundary annotation pipeline.

## Environment

Create or refresh the virtual environment:

```bash
uv sync --extra hwp
```

Run tests:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Run the preprocessing dry run:

```bash
uv run sermon-preprocessing-comparison --dry-run --max-sentences 8
```
