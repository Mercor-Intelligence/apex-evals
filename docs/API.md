## apex-eval Python package – API overview

This document describes the small, single-task API surface of the apex-eval Python package. It focuses on running one generation or one grading task at a time, without any CSV or bulk job abstractions.

---

## Imports

You typically need these imports (after installing the package so `generation` and `grading` are available on your `PYTHONPATH`):

```python
from generation import (
    GenerationTask,
    GenerationResult,
    ModelConfig,
    Attachment,
    run_generation_task,
)

from grading import (
    GradingTask,
    GradingResult,
    GradingModelConfig,
    run_grading_task,
)
```

---

## Generation

### `ModelConfig`

Configuration for a single model call.

- **Fields (common)**
  - `model_id: str` – model identifier (e.g. `"gpt-4o-mini"`, `"claude-3-5-haiku-20241022"`, `"gemini-2.5-flash"`).
  - `output_fields: List[str]` – optional logical labels for this model’s outputs; defaults to `[model_id]` when empty.
  - `number_of_runs: int` – how many times to call this model for the same prompt (≥1).
  - `max_input_tokens: Optional[int]` – trim input context to this many tokens (if supported).
  - `max_tokens: Optional[int]` – maximum tokens to generate.
  - `temperature: float` – sampling temperature in \[0.0, 2.0].
  - `top_p: Optional[float]` – nucleus sampling in \[0.0, 1.0].
  - `use_tools: bool` – whether to enable tool calling (if the underlying model supports tools).
  - `is_custom_model: bool` / `custom_model_config: Optional[dict]` – hook for custom model backends.
  - `enable_thinking: Optional[bool]` / `thinking_tokens: Optional[int]` – extended “thinking” budget for models that support it (e.g. Claude).
  - `api_key: Optional[str]` – optional override API key just for this model.

### `Attachment`

Represents a single document attachment to be parsed and used as context.

- **Fields**
  - `filename: str` – display name (for logging / prompts).
  - `url: str` – publicly accessible URL to the file (e.g. a PDF).

### `GenerationTask`

Input payload for a single generation request.

- **Required**
  - `prompt: str` – prompt text sent to the model(s).
  - `models: List[ModelConfig]` – one or more model configurations.
- **Optional**
  - `system_prompt: Optional[str]` – system-level instructions.
  - `retries: int` – retry attempts for transient failures (0–10, default 3).
  - `attachments: Optional[List[Attachment]]` – list of documents to parse and include.
  - `parsing_method: Optional[str]` – document parsing backend; `"reducto"` by default.
  - `cache_parsed_documents: bool` – whether to cache parsed documents between runs.

### `run_generation_task(task: GenerationTask) -> GenerationResult`

Executes the generation task synchronously.

- **Returns** a `GenerationResult` Pydantic model.

### `run_generation_task_async(task: GenerationTask) -> GenerationResult`

Async variant of the same API for use inside async applications.

### `GenerationResult`

Summary of a single generation task.

- **Fields**
  - `results: List[dict]` – one entry per model run, including model identifier, response text, token usage, and any error flags.
  - `completed: int` – number of successful runs.
  - `failed: int` – number of failed runs.
  - `total_tokens: int` – total tokens consumed across all runs.
  - `total_cost: float` – estimated total cost in USD.

---

## Grading

### `GradingModelConfig`

Configuration for the grading model used to evaluate a response.

- **Fields**
  - `model_id: str` – grading model identifier (default `"gemini-2.5-flash"`).
  - `max_tokens: int` – maximum tokens for grading (default `1_000_000`).
  - `temperature: float` – grading temperature (default `0.01`; keep low for stability).
  - `api_key: Optional[str]` – optional override API key for the grading model.
  - `use_tools: bool` – whether to enable tools when grading.

### `GradingTask`

Input payload describing the solution and rubric.

- **Required**
  - `solution: str` – the model response (or human-written answer) to grade.
  - `rubric: Union[str, dict, list]` – rubric definition:
    - If a **string**, it must be valid JSON and will be parsed.
    - If a **list**, it should be a list of dicts; they will be merged into a single dict.
    - After normalization it must be a **non-empty dict** mapping criterion names to configs.
- **Optional**
  - `grading_model: GradingModelConfig` – configuration for the grading model (defaults provided).
  - `grading_prompt_template: Optional[str]` – custom grading prompt template or path-like string.
  - `response_images: Optional[List[str]]` – optional image URLs associated with the response.

The rubric format is designed to be flexible. A common pattern is:

```python
rubric = [
    {
        "criterion 1": {
            "description": "What to evaluate",
            "weight": "Primary objective(s)",
            "criterion_type": ["Reasoning"],
        }
    },
    {
        "criterion 2": {
            "description": "Another aspect",
            "weight": "Secondary objective(s)",
            "criterion_type": ["Factual"],
        }
    },
]
```

### `run_grading_task(task: GradingTask) -> GradingResult`

Executes grading synchronously and returns a `GradingResult`.

### `run_grading_task_async(task: GradingTask) -> GradingResult`

Async variant of the grading API.

### `GradingResult`

Summary of grading outcomes.

- **Fields**
  - `points_earned: float` – total points awarded.
  - `points_possible: int` – maximum possible points.
  - `percentage_score: float` – score as a percentage.
  - `criteria_results: List[dict]` – per-criterion results (keys, pass/fail, reasons, etc.).
  - `grading_error: Optional[str]` – error message, if grading failed.
  - `execution_time_seconds: float` – wall-clock time to grade.
  - `total_grading_tokens: int` – grading token usage.
  - `total_grading_cost: float` – estimated grading cost in USD.

---

## Errors

The package raises structured exceptions defined in `errors`:

- `ApexEvalError` – base class for structured errors.
- `UserInputError` – configuration or caller input issues.
- `SystemExecutionError` – system-level or provider issues.

Each error can include helpful context and suggested next steps in its message. You can catch these to present cleaner errors to your own users.


