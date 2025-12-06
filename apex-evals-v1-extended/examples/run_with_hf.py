"""
APEX Evaluation Script for APEX-v1-extended dataset.

Usage:
    python run_with_hf.py --input_dir /path/to/APEX-v1-extended --output results.csv
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from generation import Attachment, GenerationTask, ModelConfig, run_generation_task_async
from grading import GradingModelConfig, GradingTask, run_grading_task_async

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = Path("prompt/response_generation_prompt.txt").read_text(encoding="utf-8")


def format_prompt(domain: str, task_prompt: str) -> str:
    """Format the prompt template with domain and task prompt values."""
    return PROMPT_TEMPLATE.replace("{{Domain}}", domain).replace("{{Prompt}}", task_prompt)



MODELS = [
    # {
    #     "model_id": "gpt-5",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 128000,
    #     "max_input_tokens": 272000,
    # },
    # {
    #     "model_id": "gpt-5.1",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 127997,
    #     "max_input_tokens": 272000,
    # },
    # {
    #     "model_id": "o3",
    #     "model_configs": None,
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 100000,
    #     "max_input_tokens": 200000,
    # },
    # {
    #     "model_id": "claude-opus-4-1-20250805",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 1,
    #     "top_p": 0.9,
    #     "max_tokens": 32000,
    #     "max_input_tokens": 200000,
    # },
    {
        "model_id": "claude-opus-4-5-20251101",
        "model_configs": {"reasoning_effort": "high"},
        "temperature": 1,
        "top_p": 0.9,
        "max_tokens": 64000,
        "max_input_tokens": 200000,
    },
    # {
    #     "model_id": "claude-sonnet-4-5-20250929",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 64000,
    #     "max_input_tokens": 200000,
    # },
    # {
    #     "model_id": "gemini-2.5-pro",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 65535,
    #     "max_input_tokens": 1048576,
    # },
    # {
    #     "model_id": "gemini-2.5-flash",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 65535,
    #     "max_input_tokens": 1048576,
    # },
    # {
    #     "model_id": "grok-4-0709",
    #     "model_configs": {"reasoning_effort": "high"},
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "max_tokens": 256000,
    #     "max_input_tokens": 256000,
    # },
    {
        "model_id": "gemini-3-pro-preview",
        "model_configs": {"reasoning_effort": "high"},
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 65535,
        "max_input_tokens": 1048576,
    },
]

GRADING_MODEL = "gemini-2.5-flash"
GRADING_MAX_TOKENS = 65535
NUMBER_OF_RUNS = 1


def sanitize(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace("/", "_")


def get_csv_headers() -> list:
    headers = ["task_id", "domain", "status"]
    for model in MODELS:
        model_id = model["model_id"]
        for run in range(1, NUMBER_OF_RUNS + 1):
            headers.extend([f"{sanitize(model_id)}_{run}_response", f"{sanitize(model_id)}_{run}_score", f"{sanitize(model_id)}_{run}_score_summary"])
    return headers


def create_attachments(file_attachments_str: str, base_dir: str) -> list:
    attachments = []
    for rel_path in file_attachments_str.strip().split("\n"):
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            attachments.append(Attachment(filename=os.path.basename(full_path), url=f"file://{os.path.abspath(full_path)}"))
        else:
            logger.warning(f"File not found: {full_path}")
    return attachments


def parse_rubric(rubric_json: str) -> dict:
    """Parse rubric JSON into a dict keyed by criterion.

    Raises if the JSON is invalid or does not resolve to an object.
    """
    if not rubric_json:
        raise ValueError("Empty rubric JSON for grading")

    try:
        data = json.loads(rubric_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse rubric JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Rubric JSON is not an object; got {type(data).__name__}")

    return data


async def run_generation(prompt: str, model_cfg: dict, attachments: list) -> dict:
    try:
        model_id = model_cfg["model_id"]
        task = GenerationTask(
            prompt=prompt,
            models=[ModelConfig(
                model_id=model_id,
                max_tokens=model_cfg["max_tokens"],
                temperature=model_cfg["temperature"],
                model_configs=model_cfg.get("model_configs"),
                number_of_runs=1,
            )],
            attachments=attachments or None,
        )
        result = await run_generation_task_async(task)
        if result.results and result.results[0].get("success"):
            return {"success": True, "response": result.results[0].get("response", "")}
        return {"success": False, "response": "", "error": result.results[0].get("error_message", "Unknown") if result.results else "No results"}
    except Exception as e:
        return {"success": False, "response": "", "error": str(e)}


async def run_grading(response: str, rubric_json: str) -> dict:
    """Run grading and return score + updated rubric JSON.

    Any failure raises an exception; no fallback summaries are produced.
    """
    if not response:
        raise ValueError("Empty model response for grading")

    # Parse rubric JSON first so errors surface clearly.
    rubric_dict = parse_rubric(rubric_json)

    config = GradingModelConfig(
        model_id=GRADING_MODEL,
        max_tokens=GRADING_MAX_TOKENS,
        temperature=0.1,
    )
    grading_task = GradingTask(solution=response, rubric=rubric_json, grading_model=config)
    result = await run_grading_task_async(grading_task)

    if not result.criteria_results:
        raise ValueError("Grading model returned no criteria results")

    # Inject autorating + reason into the original rubric structure.
    for criterion_result in result.criteria_results:
        key = criterion_result.get("criterion_key")
        if key in rubric_dict and isinstance(rubric_dict[key], dict):
            rubric_dict[key]["autorating"] = bool(criterion_result.get("autorating"))
            rubric_dict[key]["reason"] = criterion_result.get("reason", "")

    score_summary = json.dumps(rubric_dict, ensure_ascii=False)
    return {"score": result.percentage_score, "score_summary": score_summary}


def load_existing_results(output_file: str) -> dict:
    if not os.path.exists(output_file):
        return {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return {row["task_id"]: row for row in csv.DictReader(f) if row.get("task_id")}
    except Exception:
        return {}


async def process_task(task_data: dict, base_dir: str) -> dict:
    task_id = task_data.get("Task ID", "unknown")
    domain = task_data.get("Domain", "")
    task_prompt = task_data.get("Prompt", "")
    logger.info(f"Processing: {task_id}")
    result_row = {"task_id": task_id, "domain": domain, "status": "pending"}

    try:
        attachments = create_attachments(task_data.get("File Attachments", ""), base_dir)
        rubric_json = task_data.get("Rubric JSON", "").strip()
        formatted_prompt = format_prompt(domain, task_prompt)

        for model_cfg in MODELS:
            model_id = model_cfg["model_id"]
            num_runs = NUMBER_OF_RUNS
            for run in range(1, num_runs + 1):
                prefix = f"{sanitize(model_id)}_{run}"
                logger.info(f"  {model_id} run {run}/{num_runs}")

                gen = await run_generation(formatted_prompt, model_cfg, attachments)
                response = gen.get("response", "")
                result_row[f"{prefix}_response"] = response

                if not gen.get("success"):
                    result_row[f"{prefix}_score"] = 0
                    result_row[f"{prefix}_score_summary"] = f"Generation failed: {gen.get('error', '')[:100]}"
                    continue

                if rubric_json and response:
                    try:
                        grade = await run_grading(response, rubric_json)
                    except Exception as e:
                        # Do not write any grading data to CSV for this run; just log the error clearly.
                        logger.error(
                            f"    Grading failed for {model_id} run {run}: {e}",
                            exc_info=True,
                        )
                        continue

                    result_row[f"{prefix}_score"] = grade["score"]
                    result_row[f"{prefix}_score_summary"] = grade["score_summary"]
                    logger.info(f"    Score: {grade['score']:.1f}%")
                else:
                    result_row[f"{prefix}_score"] = 0
                    result_row[f"{prefix}_score_summary"] = "No rubric or empty response"

        result_row["status"] = "completed"
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        result_row["status"] = f"error: {str(e)[:100]}"

    return result_row


async def main():
    parser = argparse.ArgumentParser(description="Run APEX evaluations")
    parser.add_argument("--input_dir", default="/Users/kanishkasahu/Documents/benchmark-framework/apex/apex-evals/APEX-v1-extended")
    parser.add_argument("--output", default="apex_results.csv")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    csv_path = os.path.join(args.input_dir, "data", "train.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Input CSV not found: {csv_path}")
        sys.exit(1)

    headers = get_csv_headers()
    existing = load_existing_results(args.output) if args.resume else {}

    with open(csv_path, "r", encoding="utf-8") as f:
        tasks = list(csv.DictReader(f))

    end_idx = len(tasks) if args.limit is None else min(args.start_index + args.limit, len(tasks))
    tasks = tasks[args.start_index:end_idx]
    logger.info(f"Processing {len(tasks)} tasks")

    if not args.resume or not os.path.exists(args.output):
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    processed = 0
    for task_data in tasks:
        task_id = task_data.get("Task ID", "unknown")
        if args.resume and task_id in existing:
            logger.info(f"Skipping {task_id} (already done)")
            continue

        result = await process_task(task_data, args.input_dir)
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers, extrasaction="ignore").writerow(result)
        processed += 1

    logger.info(f"Done. Processed {processed} tasks. Results: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
