import argparse
import asyncio
import csv
import json
import logging
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from generation import Attachment, GenerationTask, ModelConfig, run_generation_task_async
from grading import GradingModelConfig, GradingTask, run_grading_task_async

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === CONFIGURATION ===

PROMPT_TEMPLATE = Path("prompt/response_generation_prompt.txt").read_text(encoding="utf-8")

MODELS = [
    {"model_id": "gpt-5", "model_configs": {"reasoning_effort": "high" , "verbosity" : "medium"}, "max_tokens": 128000, "max_input_tokens": 272000},
    {"model_id": "gpt-5.1", "model_configs": {"reasoning_effort": "high" , "verbosity" : "medium"}, "max_tokens": 127997, "max_input_tokens": 272000},
    {"model_id": "gpt-5.2", "model_configs": {"reasoning_effort": "high" , "verbosity" : "medium"}, "max_tokens": 127997, "max_input_tokens": 272000},
    {"model_id": "gpt-5.2-pro", "model_configs": {"reasoning_effort": "high" , "verbosity" : "medium"}, "max_tokens": 127997, "max_input_tokens": 272000},
    {"model_id": "o3", "model_configs": {"reasoning_effort": "high" }, "max_tokens": 100000, "max_input_tokens": 200000},
    {"model_id": "claude-opus-4-1-20250805", "model_configs": {"reasoning_effort": "high"}, "temperature": 1, "max_tokens": 32000, "max_input_tokens": 200000},
    {"model_id": "claude-opus-4-5-20251101", "model_configs": {"reasoning_effort": "high"}, "temperature": 1, "max_tokens": 64000, "max_input_tokens": 200000},
    {"model_id": "claude-sonnet-4-5-20250929", "model_configs": {"reasoning_effort": "high"}, "temperature": 1, "max_tokens": 64000, "max_input_tokens": 200000},
    {"model_id": "gemini-2.5-pro", "model_configs": {"reasoning_effort": "high"}, "temperature": 0.7, "max_tokens": 65535, "max_input_tokens": 1048576},
    {"model_id": "gemini-2.5-flash", "model_configs": {"reasoning_effort": "high"}, "temperature": 0.7, "max_tokens": 65535, "max_input_tokens": 1048576},
    {"model_id": "grok-4-0709", "model_configs": {"reasoning_effort": "high"}, "temperature": 0.8, "max_tokens": 256000, "max_input_tokens": 256000},
    {"model_id": "gemini-3-pro-preview", "model_configs": {"reasoning_effort": "high"}, "max_tokens": 65535, "max_input_tokens": 1048576},
]

GRADING_MODEL = "gemini-2.5-flash"
GRADING_MAX_TOKENS = 65535
NUMBER_OF_RUNS = 1
VALID_DOMAINS = ["Consulting", "Finance", "Legal", "Medicine"]


# === HELPERS ===

def sanitize(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace("/", "_")


def get_model_keys() -> list[str]:
    return [sanitize(m["model_id"]) for m in MODELS]


def get_csv_headers() -> list[str]:
    headers = ["task_id", "domain", "status"]
    for model_key in get_model_keys():
        for run in range(1, NUMBER_OF_RUNS + 1):
            headers.extend([f"{model_key}_{run}_response", f"{model_key}_{run}_score", f"{model_key}_{run}_score_summary"])
    return headers


def create_attachments(file_attachments_str: str, base_dir: str) -> list[Attachment]:
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


def load_completed_tasks(output_file: str) -> set[str]:
    """Load task IDs that have status='completed'."""
    if not os.path.exists(output_file):
        return set()
    completed = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "completed":
                completed.add(row.get("task_id", ""))
    return completed


def save_result(output_file: str, headers: list[str], result: dict):
    """Append a single result row to CSV."""
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writerow(result)


# === GENERATION & GRADING ===

async def generate(prompt: str, model_cfg: dict, attachments: list) -> dict:
    try:
        model_config_kwargs = {
            "model_id": model_cfg["model_id"],
            "max_tokens": model_cfg["max_tokens"],
            "max_input_tokens": model_cfg.get("max_input_tokens"),
            "model_configs": model_cfg.get("model_configs"),
            "number_of_runs": 1,
        }
        if "temperature" in model_cfg:
            model_config_kwargs["temperature"] = model_cfg["temperature"]

        task = GenerationTask(
            prompt=prompt,
            models=[ModelConfig(**model_config_kwargs)],
            attachments=attachments or None,
        )
        result = await run_generation_task_async(task)
        if result.results and result.results[0].get("success"):
            return {"success": True, "response": result.results[0].get("response", "")}
        error = result.results[0].get("error_message", "Unknown") if result.results else "No results"
        return {"success": False, "response": "", "error": error}
    except Exception as e:
        return {"success": False, "response": "", "error": str(e)}


async def grade(response: str, rubric_json: str) -> dict:
    rubric_dict = json.loads(rubric_json)
    config = GradingModelConfig(model_id=GRADING_MODEL, max_tokens=GRADING_MAX_TOKENS, temperature=0.01)
    grading_task = GradingTask(solution=response, rubric=rubric_json, grading_model=config)
    result = await run_grading_task_async(grading_task)

    if not result.criteria_results:
        raise ValueError("Grading returned no results")

    for cr in result.criteria_results:
        key = cr.get("criterion_key")
        if key in rubric_dict and isinstance(rubric_dict[key], dict):
            rubric_dict[key]["autorating"] = bool(cr.get("autorating"))
            rubric_dict[key]["reason"] = cr.get("reason", "")

    return {"score": result.percentage_score, "score_summary": json.dumps(rubric_dict, ensure_ascii=False)}


# === TASK PROCESSING ===

async def process_task(task_data: dict, base_dir: str) -> dict | None:
    """Process a task. Returns None if any grading fails (task will be skipped)."""
    task_id = task_data.get("Task ID", "unknown")
    domain = task_data.get("Domain", "")
    logger.info(f"Processing: {task_id} ({domain})")

    result = {"task_id": task_id, "domain": domain, "status": "pending"}

    try:
        attachments = create_attachments(task_data.get("File Attachments", ""), base_dir)
        rubric_json = task_data.get("Rubric JSON", "").strip()
        prompt = PROMPT_TEMPLATE.replace("{{Domain}}", domain).replace("{{Prompt}}", task_data.get("Prompt", ""))

        for model_cfg in MODELS:
            model_key = sanitize(model_cfg["model_id"])

            for run in range(1, NUMBER_OF_RUNS + 1):
                prefix = f"{model_key}_{run}"

                # Generate
                gen = await generate(prompt, model_cfg, attachments)
                if not gen.get("success"):
                    error_msg = gen.get("error", "Unknown error")
                    logger.error(f"  {prefix}: Generation failed - {error_msg}")
                    return None

                result[f"{prefix}_response"] = gen["response"]

                # Grade
                if not rubric_json or not gen["response"]:
                    logger.error(f"  {prefix}: No rubric or empty response - skipping task")
                    return None

                try:
                    g = await grade(gen["response"], rubric_json)
                    result[f"{prefix}_score"] = g["score"]
                    result[f"{prefix}_score_summary"] = g["score_summary"]
                    logger.info(f"  {prefix}: {g['score']:.1f}%")
                except Exception as e:
                    logger.error(f"  {prefix}: Grading failed ({e}) - skipping task")
                    return None

        result["status"] = "completed"
        return result

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e} - skipping task")
        return None


# === STATISTICS ===

def calculate_stats(output_file: str):
    """Calculate and display final statistics: mean of median scores per task."""
    if not os.path.exists(output_file):
        return

    with open(output_file, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "completed"]

    if not rows:
        return

    model_keys = get_model_keys()

    # Collect scores: {model: {domain: [median_per_task, ...]}}
    domain_scores = {mk: {} for mk in model_keys}
    overall_scores = {mk: [] for mk in model_keys}

    for row in rows:
        domain = row.get("domain", "Unknown")

        for model_key in model_keys:
            run_scores = []
            for run in range(1, NUMBER_OF_RUNS + 1):
                try:
                    score = float(row.get(f"{model_key}_{run}_score", 0) or 0)
                    run_scores.append(score)
                except (ValueError, TypeError):
                    run_scores.append(0)

            task_median = statistics.median(run_scores) if run_scores else 0

            if domain not in domain_scores[model_key]:
                domain_scores[model_key][domain] = []
            domain_scores[model_key][domain].append(task_median)
            overall_scores[model_key].append(task_median)

    # Print results
    print("\n" + "=" * 60)
    print("APEX EVALUATION RESULTS")
    print(f"Total Tasks: {len(rows)} | Runs per Task: {NUMBER_OF_RUNS}")
    print("=" * 60)

    for model_key in model_keys:
        if not overall_scores[model_key]:
            continue

        overall_mean = statistics.mean(overall_scores[model_key])
        print(f"\n{model_key}")
        print("-" * 40)
        print(f"  Overall: {overall_mean:.2f}% ({len(overall_scores[model_key])} tasks)")

        for domain in sorted(domain_scores[model_key].keys()):
            scores = domain_scores[model_key][domain]
            if scores:
                print(f"  {domain}: {statistics.mean(scores):.2f}% ({len(scores)} tasks)")

    print("\n" + "=" * 60 + "\n")


# === MAIN ===

async def main():
    parser = argparse.ArgumentParser(description="Run APEX evaluations")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="apex_results.csv")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--domain", type=str, nargs="+", choices=VALID_DOMAINS, default=None)
    args = parser.parse_args()

    # Load tasks
    csv_path = os.path.join(args.input_dir, "data", "train.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Input CSV not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        tasks = list(csv.DictReader(f))

    # Filter by domain
    if args.domain:
        tasks = [t for t in tasks if t.get("Domain", "") in args.domain]
        logger.info(f"Filtered to domains: {', '.join(args.domain)}")

    # Apply start/limit
    end_idx = len(tasks) if args.limit is None else args.start_index + args.limit
    tasks = tasks[args.start_index:end_idx]

    # Load completed tasks to skip
    completed = load_completed_tasks(args.output)
    tasks = [t for t in tasks if t.get("Task ID", "unknown") not in completed]

    logger.info(f"Tasks to process: {len(tasks)} (skipped {len(completed)} completed)")

    # Initialize CSV if needed
    headers = get_csv_headers()
    if not os.path.exists(args.output):
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    # Process tasks
    saved = 0
    skipped = 0
    for idx, task_data in enumerate(tasks):
        logger.info(f"\n[{idx + 1}/{len(tasks)}]")
        result = await process_task(task_data, args.input_dir)
        if result:
            save_result(args.output, headers, result)
            saved += 1
        else:
            skipped += 1
            logger.warning(f"Task skipped (not saved)")

    logger.info(f"\nSaved: {saved} | Skipped: {skipped}")

    # Show statistics
    calculate_stats(args.output)
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
