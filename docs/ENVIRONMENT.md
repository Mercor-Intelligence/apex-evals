## apex-eval Python package – Environment & configuration

This document lists the main environment variables and configuration expectations for using the `apex-eval` Python package for single-task generation and grading.

Always create and activate a virtual environment before installing and running Python scripts.

---

## Virtual environment

Example using `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you install the package from PyPI instead of a local checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install apex-eval
```

---

## Environment variables

Set your API keys as environment variables or in a `.env` file (loaded with `python-dotenv`).

### LLM provider keys

- **`OPENAI_API_KEY`** – API key for OpenAI models (e.g. `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`, `gpt-5`, etc.).
- **`ANTHROPIC_API_KEY`** – API key for Anthropic/Claude models (e.g. `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`).
- **`GOOGLE_API_KEY`** – API key for Google Gemini models (e.g. `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-pro`).
- **`XAI_API_KEY`** – API key for xAI Grok models (e.g. `grok-4-0709`, `grok-3-mini`).
- **`FIREWORKS_API_KEY`** (if applicable) – API key for Fireworks-hosted models (Qwen, LLaMA, DeepSeek, etc.).

These keys are used by the underlying `litellm` client when calling different providers, depending on the `model_id` you choose in `ModelConfig`.

### Document parsing

- **`REDUCTO_API_KEY`** – API key for the Reducto document parsing service.

  Required when you:
  - Provide `attachments` on a `GenerationTask`, and
  - Use the default `parsing_method="reducto"` (or explicitly set it to `"reducto"`).

Without a valid `REDUCTO_API_KEY`, attachment parsing requests will fail.

---

## Using a `.env` file

The package can load configuration from a `.env` file using `python-dotenv`. A minimal example:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
FIREWORKS_API_KEY=...
REDUCTO_API_KEY=...
```

In your Python script:

```python
from dotenv import load_dotenv

load_dotenv()  # reads .env from current working directory
```

You can also set environment variables directly in your shell instead of using `.env`.

---

## Common issues & troubleshooting

- **Missing module or import errors**
  - Ensure your virtual environment is activated.
  - Confirm the package is installed: `pip show apex-eval`.

- **“API key not found” or authentication failures**
  - Verify the corresponding environment variable is set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
  - If using `.env`, ensure you called `load_dotenv()` before creating tasks.

- **Document parser / attachment errors**
  - Check that `REDUCTO_API_KEY` is set and valid.
  - Confirm that attachment URLs are accessible and point to supported file types.

- **High cost or rate limit problems**
  - Prefer cheaper models (e.g. `gpt-4o-mini`, `gemini-2.5-flash`).
  - Lower `max_tokens` and `max_input_tokens` in `ModelConfig`.
  - Keep grading temperature low to avoid repeated retries.


