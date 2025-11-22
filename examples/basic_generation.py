"""
Minimal single-task generation example using the apex-eval Python package.

Always create and activate a virtual environment before running this script.
"""

import logging

from dotenv import load_dotenv

from generation import (
    GenerationResult,
    GenerationTask,
    ModelConfig,
    run_generation_task,
)


def main() -> None:
    # Load environment variables from .env (OPENAI_API_KEY, etc.)
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    task = GenerationTask(
        prompt="Explain what large language models are, in two short paragraphs.",
        models=[
            ModelConfig(
                model_id="gpt-4o-mini",
                max_tokens=300,
                temperature=0.4,
            )
        ],
    )

    result: GenerationResult = run_generation_task(task)

    print("Generation result:")
    print(f"  completed={result.completed} failed={result.failed}")
    print(f"  tokens={result.total_tokens} cost=${result.total_cost:.4f}")

    if not result.results:
        print("No results returned.")
        return

    first = result.results[0]
    print("\nFirst response:")
    print(first.get("response", "")[:1000])


if __name__ == "__main__":
    main()


