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

Build the extraction dataset from `datas/`:

```bash
uv run sermon-dataset-build
```

Build GPT-5.5 teacher Batch requests from extracted sentences:

```bash
uv run sermon-teacher-batch-build \
  --sentences tests/results/dataset_final_check_20260529_1431/sentences.jsonl \
  --out-dir tests/results/teacher_batch_$(date +%Y%m%d_%H%M%S) \
  --model gpt-5.5 \
  --target-size 160 \
  --left-context 20 \
  --right-context 20
```

After downloading OpenAI Batch results, ingest and validate them:

```bash
uv run sermon-teacher-batch-ingest \
  --windows tests/results/teacher_batch_20260529_000000/windows.jsonl \
  --batch-output tests/results/teacher_batch_20260529_000000/openai_batch_output.jsonl \
  --out-dir tests/results/teacher_ingest_20260529_000000
```

Build student SFT datasets:

```bash
uv run sermon-sft-build \
  --annotations tests/results/teacher_ingest_20260529_000000/teacher_annotations.jsonl \
  --windows tests/results/teacher_batch_20260529_000000/windows.jsonl \
  --out-dir tests/results/student_sft_20260529_000000
```
