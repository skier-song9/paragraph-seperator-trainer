# Sermon Boundary Training Design

Date: 2026-05-29

## 1. Goal

Build a low-cost sermon discourse boundary model for RAG chunking.

The final student model must:

- accept at least 30,000 input tokens;
- use a small open-weight model first, before escalating to 3-4B models;
- receive text-only sentence-anchor input at inference time;
- output sparse boundary references, not dense labels;
- support evaluation both as a boundary detector and as a RAG preprocessing component.

This spec covers three linked parts:

1. GPT-5.5 teacher labeling of the current `datas/` corpus;
2. transformation from teacher labels into student SFT examples;
3. the first two student experiments: sparse-only and first-boundary curriculum.

## 2. Non-Goals

Excluded from this design:

- dense student output labels;
- segment-span output;
- permanent metadata in student input;
- GPT-5.5 fine-tuning;
- 3-4B student models as the first experiment.

Dense labels are still allowed inside teacher annotation artifacts because they make validation, audit, and conversion easier. They must not become the final student output format.

## 3. Prior-Work Position

The project should not claim to invent semantic chunking for RAG. The correct claim is narrower:

> sermon-specific supervised sentence-boundary modeling, distilled from LLM teacher labels and evaluated both on boundary quality and RAG answer quality.

The local prior-work review identifies PIC, LumberChunker, Meta-Chunking, and Max-Min semantic chunking as close baselines. LumberChunker is especially relevant because it asks an LLM to locate shift points in long-form narrative passages. That supports using first-boundary detection as an intermediate task, while the final system remains sparse multi-boundary prediction.

Local reference:

- `docs/reviews/sermon_rag_segmentation_prior_work_260526.html`

External references:

- LumberChunker: https://aclanthology.org/2024.findings-emnlp.377/
- Meta-Chunking: https://arxiv.org/abs/2410.12788
- Curriculum Learning: https://icml.cc/Conferences/2009/papers/119.pdf
- STILTs intermediate-task fine-tuning: https://arxiv.org/abs/1811.01088

## 4. Current Corpus

The current extraction dataset was generated from `datas/` into:

- `tests/results/dataset_final_check_20260529_1431/documents.jsonl`
- `tests/results/dataset_final_check_20260529_1431/sentences.jsonl`
- `tests/results/dataset_final_check_20260529_1431/failures.jsonl`

Current counts:

- total documents: 208
- total sentences: 115,687
- failures: 0
- `datalab_parsed_json`: 80 documents, 64,050 sentences
- `docx`: 90 documents, 28,530 sentences
- `hwp`: 38 documents, 23,107 sentences

`sentences.jsonl` is an extraction artifact, not a training dataset. It contains permanent `sentence_id`, source path, source type, block metadata, heading hints, and text. Student training examples must be derived from it.

## 5. GPT-5.5 Teacher Labeling Plan

GPT-5.5 is used only as a teacher annotation model. Official OpenAI docs list GPT-5.5 with a 1,050,000-token context window, Structured Outputs support, Batch support, and no fine-tuning support. Batch API is appropriate because OpenAI documents it for asynchronous classification of large datasets and lower-cost processing.

Official references:

- GPT-5.5 model docs: https://developers.openai.com/api/docs/models/gpt-5.5/
- Responses API: https://developers.openai.com/api/reference/resources/responses/methods/create
- Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Batch API: https://developers.openai.com/api/docs/guides/batch

### 5.1 Teacher Windowing

Do not annotate whole documents in one dense-output call by default. Long context is available, but dense JSON output for hundreds or thousands of sentences becomes large, expensive, and harder to validate.

Use target windows:

- target span: about 120-200 sentences;
- left context: about 20 sentences;
- right context: about 20 sentences;
- only target sentences require labels;
- context sentences are visible but not labeled;
- every request has a deterministic `custom_id`.

The target-window approach gives GPT-5.5 enough discourse context while keeping output bounded.

### 5.2 Teacher Input

Teacher input may include extraction metadata as hints:

- source type;
- document kind;
- source path;
- heading context;
- original paragraph/page hints;
- block type;
- extraction notes.

This is allowed because teacher annotation is an offline labeling step. The student model must not receive this metadata in its final SFT input.

Teacher prompt includes target sentences with local anchors:

```text
<S1> sentence text
<S2> sentence text
<S3> sentence text
```

Teacher-side hidden mapping stores:

- `custom_id`
- `document_id`
- `source_sentence_id`
- `global_sentence_index`
- `local_sid`
- whether the sentence is target or context

### 5.3 Teacher Output

Teacher output uses strict JSON schema, not free-form text.

Required shape:

```json
{
  "custom_id": "teacher:datalab_parsed_json.d1a842ffda53:w0004",
  "boundary_annotations": [
    {
      "local_sid": "S12",
      "source_sentence_id": "datalab_parsed_json.d1a842ffda53.s0371",
      "split_after": true,
      "boundary_type": "topic_shift",
      "confidence": 0.84,
      "rationale": "New discourse unit begins after this sentence."
    }
  ],
  "quality_flags": []
}
```

The teacher must return one annotation for each target sentence whose immediate next source sentence is visible in the target span or right context. The final document sentence and any sentence without a visible next sentence are not valid split candidates. `boundary_type` is `none` when `split_after=false`.

Allowed boundary types:

- `none`
- `topic_shift`
- `scripture_reading_start`
- `scripture_explanation_start`
- `illustration_start`
- `application_start`
- `prayer_or_closing`
- `enumeration_start`

### 5.4 Teacher Validation

Each GPT-5.5 response must be validated before conversion:

- JSON parse success;
- schema conformance;
- every target sentence covered exactly once;
- no unknown `local_sid`;
- no unknown `source_sentence_id`;
- no invalid boundary type;
- `split_after=false` implies `boundary_type=none`;
- `split_after=true` implies non-`none`;
- confidence in `[0, 1]`;
- no label after a sentence whose immediate next source sentence is not visible.

Validation outputs:

- `teacher_annotations.jsonl`
- `teacher_validation_issues.jsonl`
- `teacher_failures.jsonl`
- `needs_human_review.jsonl`
- `labeling_run_summary.json`

Failures must be traceable by `custom_id`, `document_id`, `source_path`, window start/end, exception type, and raw response file path.

### 5.5 Human Review Queue

Send these cases to human adjudication:

- confidence below 0.75;
- any `quality_flags`;
- overlap conflicts between adjacent teacher windows;
- rare boundary labels;
- boundaries near scripture references;
- long documents where boundary density is unusually high or low;
- failed or repaired JSON responses.

Human review produces the final adjudicated label artifact. The student trains on adjudicated labels when available, otherwise validated teacher labels.

## 6. Student Input Format

Student inference input has no source metadata.

Canonical format:

```text
<TASK>
Find sermon discourse split boundaries.
Return only boundary lines. Use NO_BOUNDARY if no split is needed.
Each boundary line means split after that sentence.
Allowed labels: topic_shift, scripture_reading_start, scripture_explanation_start, illustration_start, application_start, prayer_or_closing, enumeration_start

<TEXT>
<S1> sentence text
<S2> sentence text
<S3> sentence text
...
```

Rules:

- `<S1>`, `<S2>`, ... are window-local anchors.
- `<S10000>` is allowed if a window ever contains that many sentences, but normal windows should be much smaller.
- The anchor is not permanent document metadata.
- The source `sentence_id` is stored only in sidecar mapping files.
- `Sx label` means split after sentence `<Sx>`.
- valid output labels exclude `none`.

## 7. Student Output Format

Sparse Multi-Boundary List is final.

No boundary:

```text
NO_BOUNDARY
```

One or more boundaries:

```text
S12 topic_shift
S39 scripture_explanation_start
S88 application_start
```

Rules:

- no `N=`;
- one boundary per line;
- line order must be ascending by local sentence number;
- duplicate `Sx` is invalid;
- unknown labels are invalid;
- boundary after the final local sentence is invalid unless a future explicit document-end task is added.

## 8. Label-to-SFT Conversion

Teacher dense annotations are converted into two student datasets.

### 8.1 Sparse-Only Dataset

Input:

```text
<TASK>
...
<TEXT>
<S1> ...
<S2> ...
```

Target:

```text
S12 topic_shift
S39 scripture_explanation_start
```

If no non-`none` labels exist inside the target region:

```text
NO_BOUNDARY
```

### 8.2 First-Boundary Dataset

Input uses the same text format.

Target is the earliest non-`none` boundary:

```text
S12 topic_shift
```

If no boundary exists:

```text
NO_BOUNDARY
```

This dataset is used only for curriculum Stage 1.

### 8.3 Mapping Files

Every SFT example must have a sidecar mapping:

```json
{
  "example_id": "sft:datalab_parsed_json.d1a842ffda53:w0004",
  "document_id": "datalab_parsed_json.d1a842ffda53",
  "source_path": "datas/...",
  "local_to_source": {
    "S1": {
      "source_sentence_id": "datalab_parsed_json.d1a842ffda53.s0340",
      "global_sentence_index": 340
    }
  }
}
```

The mapping is never included in student input. It is used for evaluation, error analysis, and RAG chunk reconstruction.

## 9. Student Experiments

### 9.1 E1: Small Model Sparse-Only Baseline

Train the small model directly on Sparse Multi-Boundary List targets.

Purpose:

- establish the simplest final-task baseline;
- measure whether the small model can learn sparse boundary output without curriculum;
- provide the required comparison for curriculum.

### 9.2 E2: Small Model Curriculum

Stage 1: train on First Boundary Only.

Stage 2: continue training the same checkpoint on Sparse Multi-Boundary List.

Final inference uses only the sparse multi-boundary prompt.

Purpose:

- test whether first-boundary detection improves boundary sensitivity;
- preserve the final output contract;
- avoid committing to curriculum without ablation.

### 9.3 Model Priority

Start with small long-context models.

Candidate class:

- 1-2B models with at least 32K context;
- Qwen2.5-1.5B-Instruct and Gemma 3 1B are candidate families, subject to final environment check before training.

Because 32K context is close to the 30K requirement, every candidate must pass a stress test with the full task prompt, sentence anchors, and at least 30,000 input tokens of sermon text. A model that only accepts 30,000 tokens when anchors/instructions are removed does not satisfy this project requirement.

Escalate to 3-4B only if both E1 and E2 fail absolute quality thresholds.

## 10. Training Objective

Use causal LM supervised fine-tuning.

For each example:

- input prompt and sentence text are context;
- target boundary text is the completion;
- loss is token-level cross-entropy over target tokens only;
- prompt/input tokens are masked from loss.

E1 objective:

```text
maximize p(sparse_boundary_list | anchored_sentence_window)
```

E2 objective:

```text
Stage 1: maximize p(first_boundary | anchored_sentence_window)
Stage 2: maximize p(sparse_boundary_list | anchored_sentence_window)
```

Curriculum is a hypothesis, not an assumed improvement. It is adopted only if E2 beats E1 on validation metrics.

## 11. Evaluation

Boundary metrics:

- precision, recall, F1;
- +/-1 sentence tolerance F1;
- Pk;
- WindowDiff;
- parse success rate;
- invalid line rate;
- duplicate/unknown anchor rate;
- invalid label rate.

RAG metrics:

- retrieval recall@k;
- MRR;
- nDCG;
- citation hit rate;
- groundedness;
- answer correctness;
- irrelevant-context sensitivity.

Cost metrics:

- teacher labeling cost;
- student inference latency;
- VRAM;
- tokens per document;
- storage overhead;
- batch failure/retry rate.

Decision rule:

- if E2 > E1: keep curriculum;
- if E2 <= E1: remove curriculum;
- if both fail absolute thresholds: escalate to 3-4B long-context student models;
- if small model meets quality and cost thresholds: stop escalation.

## 12. Implementation Boundary

This spec does not implement code. The next implementation plan should add code in separate modules for:

- GPT-5.5 teacher batch request generation;
- batch result ingestion and validation;
- teacher-to-SFT conversion;
- sparse-only and first-boundary dataset writers;
- parse/evaluation utilities;
- run summaries and failure logs.

The implementation must preserve existing extraction outputs and not rewrite current `datas/` files.
