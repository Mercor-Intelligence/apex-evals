# APEX-v1-extended

A benchmark for measuring whether frontier models can perform economically valuable work across four jobs: investment banking associate, management consultant, big law associate, and primary care physician (MD)

This Python package contains our harness to evaluate LLMs on APEX-v1. It generates responses from multiple LLM providers and grades them against the tasks.

## Quickstart: run the APEX-v1-extended evals

Use this if you just want to run the benchmark end-to-end with Hugging Face data.

1. **Clone and enter this repo**
   - `git clone https://github.com/Mercor-Intelligence/apex-evals`
   - `cd apex-evals-v1-extended`
2. **Create and activate a virtual environment**
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3. **Install dependencies**
   - `pip install -r requirements.txt`
   - `pip install -e .`
4. **Get the APEX-v1-extended dataset**
   - `git clone https://huggingface.co/datasets/mercor/APEX-v1-extended`
5. **Create your `.env`**
   - `cp example.env .env` and fill in your API keys.
6. **Run the benchmark**
   - `python examples/run_with_hf.py --input_dir /full/path/to/APEX-v1-extended --output apex_results.csv --start_index 0 --limit 5`

## Installation

All commands below assume you are in the root of this repo.

```bash
# Clone and navigate
cd apex-evals-v1-extended

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Setup

The library expects your LLM and (optionally) document parsing API keys to be available via environment variables.
The simplest way to do this in local development is to create a `.env` file in the project root:

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Document Parsing
REDUCTO_API_KEY=...
```

## Usage

### 1. Generate Responses

```python
from generation import GenerationTask, ModelConfig, run_generation_task, Attachment

# Single task
task = GenerationTask(
    prompt="Answer this question: What is AI?",
    models=[
        ModelConfig(
            model_id="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )
    ],
    system_prompt="You are a helpful AI assistant.",  # Optional
    # Optional: Add document attachments
    attachments=[
        Attachment(
            filename="doc.pdf",
            url="https://example.com/doc.pdf"
        )
    ]
)

result = run_generation_task(task)
print(f"Completed: {result.completed}")
print(f"Total Cost: ${result.total_cost}")
print(f"Results: {result.results}")
```

### 2. Grade Responses

```python
from grading import GradingTask, run_grading_task

task = GradingTask(
    solution="AI is artificial intelligence...",
    rubric=[
        {
            "criterion 1": {
                "description": "Answer is factually accurate",
                "weight": "Primary objective(s)",
                "criterion_type": ["Reasoning"]
            }
        },
        {
            "criterion 2": {
                "description": "Answer is complete and thorough",
                "weight": "Primary objective(s)",
                "criterion_type": ["Reasoning"]
            }
        }
    ]
)

result = run_grading_task(task)
print(f"Score: {result.points_earned}/{result.points_possible}")
print(f"Percentage: {result.percentage_score}%")
```

## Configuration

### GenerationTask

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | str | Yes | - | Fully formed prompt to send to model |
| `models` | List[ModelConfig] | Yes | - | List of models to execute |
| `system_prompt` | str | No | None | System prompt |
| `attachments` | List[Attachment] | No | None | Document attachments to parse |
| `parsing_method` | str | No | "reducto" | Document parsing method |
| `cache_parsed_documents` | bool | No | True | Cache parsed attachments |
| `retries` | int | No | 3 | Retries for transient errors (0-10) |

### ModelConfig

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_id` | str | Yes | - | Model identifier (see Supported Models) |
| `output_fields` | List[str] | No | [model_id] | Logical labels for outputs |
| `number_of_runs` | int | No | 1 | Number of times to run (≥1) |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | No | None | Max tokens to generate |
| `max_input_tokens` | int | No | None | Max input tokens (for trimming) |
| `top_p` | float | No | None | Nucleus sampling (0.0-1.0) |
| `use_tools` | bool | No | False | Enable tool usage |
| `enable_thinking` | bool | No | None | Enable extended thinking (Claude) |
| `thinking_tokens` | int | No | None | Thinking token budget |
| `api_key` | str | No | None | Override API key for this model |
| `is_custom_model` | bool | No | False | Whether this is a custom model |
| `custom_model_config` | dict | No | None | Custom model configuration |
| `model_configs` | dict | No | None | Model specific configs |

### Attachment

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | str | Yes | Display name for the attachment |
| `url` | str | Yes | Publicly accessible URL |

### GradingTask

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `solution` | str | Yes | - | LLM response to grade |
| `rubric` | list/dict/str | Yes | - | Rubric (list, dict, or JSON string) |
| `grading_model` | GradingModelConfig | No | (default config) | Grading model configuration |
| `grading_prompt_template` | str | No | None | Custom grading prompt template |
| `response_images` | List[str] | No | None | Optional response image URLs |

### GradingModelConfig

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_id` | str | No | "gemini-2.5-flash" | Model used for grading |
| `max_tokens` | int | No | 1000000 | Max tokens for grading |
| `temperature` | float | No | 0.01 | Temperature (0.0-2.0) |
| `api_key` | str | No | None | API key override |
| `use_tools` | bool | No | False | Enable tool usage |

### Rubric Format

The rubric should be a **list of dictionaries**, where each dictionary contains a criterion name as the key:

```python
[
    {
        "criterion 1": {
            "description": "What to evaluate",  # REQUIRED
            "weight": "Primary objective(s)",   # REQUIRED: "Primary objective(s)" or "Secondary objective(s)"
            "criterion_type": ["Reasoning"],    # REQUIRED: e.g., ["Reasoning"], ["Factual"], ["Style"]
            "sources": "",                      # Optional: relevant sources
            "justification": "",                # Optional
            "human_rating": "False",            # Optional
            "dependent_criteria": []            # Optional: list of dependent criterion names
        }
    },
    {
        "criterion 2": {
            "description": "Another evaluation aspect",
            "weight": "Secondary objective(s)",
            "criterion_type": ["Factual"]
        }
    }
]
```

**Required Fields:**
- `description`: What to evaluate
- `weight`: Either `"Primary objective(s)"` or `"Secondary objective(s)"`
- `criterion_type`: List of types (e.g., `["Reasoning"]`, `["Factual"]`, `["Style"]`)

**Optional Fields:**
- `sources`: Relevant sources or references
- `justification`: Justification for the criterion
- `human_rating`: Human rating (if available)
- `dependent_criteria`: List of criteria names this depends on

### Result Objects

**GenerationResult:**
```python
{
    "results": [...],           # List of individual results
    "completed": 3,             # Number of successful completions
    "failed": 0,                # Number of failures
    "total_tokens": 1500,       # Total tokens used
    "total_cost": 0.025         # Total cost in USD
}
```

**GradingResult:**
```python
{
    "points_earned": 8.5,                    # Points earned
    "points_possible": 10,                   # Total possible points
    "percentage_score": 85.0,                # Percentage score
    "criteria_results": [...],               # Detailed results per criterion
    "grading_error": None,                   # Error message if any
    "execution_time_seconds": 2.3,           # Time taken
    "total_grading_tokens": 500,             # Tokens used for grading
    "total_grading_cost": 0.005              # Cost of grading in USD
}
```

## Supported Models

### OpenAI
- **GPT-4o Series**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **o-Series**: `o1-preview`, `o1-mini`, `o3`, `o3-mini`, `o3-deep-research`, `o4-mini-deep-research-2025-06-26`
- **GPT-4/3.5**: `gpt-4`, `gpt-3.5-turbo`
- **GPT-5**: `gpt-5`
- **GPT-5.1**: `gpt-5.1`

### Anthropic (Claude)
- **Claude Opus 4.5**: `claude-opus-4-5-20251101`,
- **Claude 4**: `claude-4`, `claude-4-sonnet-20250722`, `claude-4-haiku-20250722`, `claude-4-opus-20250722`
- **Claude Opus 4**: `claude-opus-4-1-20250805`, `claude-opus-4-20250514`
- **Claude Sonnet 4**: `claude-sonnet-4-20250514`, `claude-sonnet-4-5-20250929`
- **Claude 3.5**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- **Claude 3**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

### Google (Gemini)
- **Gemini 3**: `gemini-3-pro-preview` (1M context)
- **Gemini 2.5**: `gemini-2.5-flash`, `gemini-2.5-pro` (950K context)
- **Gemini 2.0**: `gemini-2.0-flash-exp` (950K context)
- **Gemini 1.5**: `gemini-1.5-pro`, `gemini-1.5-flash` (1M context)
- **Gemini 1.0**: `gemini-1.0-pro`, `gemini-pro` (30K context)

### xAI (Grok)
- **Grok 4**: `grok-4-0709`, `grok-code-fast-1` (256K context)
- **Grok 3**: `grok-3`, `grok-3-mini`, `grok-3-mini-beta` (131K context)
- **Grok Beta**: `grok-beta` (131K context)

## Examples

### Multiple Models

```python
task = GenerationTask(
    prompt="Explain quantum computing in simple terms",
    models=[
        ModelConfig(model_id="gpt-4o-mini"),
        ModelConfig(model_id="claude-3-5-haiku-20241022"),
        ModelConfig(model_id="gemini-2.5-flash")
    ]
)

result = run_generation_task(task)
# Results include responses from all 3 models
print(f"Completed: {result.completed}/3")
for res in result.results:
    print(f"Model: {res['model']}, Response: {res['response'][:100]}...")
```

### With Document Parsing

```python
from generation import Attachment

task = GenerationTask(
    prompt="Summarize the key findings from this research paper",
    attachments=[
        Attachment(
            filename="research.pdf",
            url="https://example.com/research.pdf"
        )
    ],
    models=[ModelConfig(model_id="gpt-4o")],
    parsing_method="reducto",
    cache_parsed_documents=True
)

result = run_generation_task(task)
```

### Complete Workflow

```python
from generation import GenerationTask, ModelConfig, run_generation_task
from grading import GradingTask, run_grading_task

# Step 1: Generate response
gen_task = GenerationTask(
    prompt="What is machine learning and how does it work?",
    models=[ModelConfig(model_id="gpt-4o-mini")]
)

gen_result = run_generation_task(gen_task)

# Get the response from first result
response_text = gen_result.results[0]["response"]

# Step 2: Grade response
grade_task = GradingTask(
    solution=response_text,
    rubric=[
        {
            "criterion 1": {
                "description": "Answer is accurate and factually correct",
                "weight": "Primary objective(s)",
                "criterion_type": ["Reasoning"]
            }
        },
        {
            "criterion 2": {
                "description": "Explanation is clear and easy to understand",
                "weight": "Secondary objective(s)",
                "criterion_type": ["Style"]
            }
        }
    ]
)

grade_result = run_grading_task(grade_task)
print(f"Score: {grade_result.points_earned}/{grade_result.points_possible}")
print(f"Percentage: {grade_result.percentage_score}%")
```

## Troubleshooting

**"Module not found"**
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**"API key not found"**
- Check `.env` file exists and has correct keys
- Or export directly: `export OPENAI_API_KEY="sk-..."`

**"Parser validation failed"**
- Verify REDUCTO_API_KEY is set
- Verify reducto is installed

**Rate limits**
- SDK has built-in retry logic
- Check your api keys limit

## License

CC-by-4.0
