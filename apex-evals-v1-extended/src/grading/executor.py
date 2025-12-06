import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from errors import SystemExecutionError
from handler.validator import ConfigValidator

from .config import GradingResult, GradingTask

logger = logging.getLogger(__name__)


DEFAULT_GRADING_PROMPT_PATH = Path("prompt/grading_prompt.txt")

if DEFAULT_GRADING_PROMPT_PATH.exists():
    DEFAULT_GRADING_PROMPT = DEFAULT_GRADING_PROMPT_PATH.read_text()
else:
    logger.error(f"Default grading prompt not found at {DEFAULT_GRADING_PROMPT_PATH}")


def parse_llm_json_response(content: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response.
    """
    content_cleaned = content.strip()
    if content_cleaned.startswith('```'):
        content_cleaned = re.sub(r'^```(?:json)?\s*\n', '', content_cleaned)
        content_cleaned = re.sub(r'\n```\s*$', '', content_cleaned)

    json_blocks: List[str] = []
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(content_cleaned):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx >= 0:
                json_blocks.append(content_cleaned[start_idx : i + 1])
                start_idx = -1

    if not json_blocks:
        try:
            parsed = json.loads(content_cleaned)
            if "result" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
            
        raise SystemExecutionError(
            title="LLM response missing JSON",
            summary="The grading model did not return a JSON object as instructed.",
            context={"response_preview": content[:200].strip() or "<empty>"},
            probable_causes=[
                "The model ignored the JSON-only instruction in the grading prompt.",
                "Temperature is too high, causing the model to add commentary.",
            ],
            next_steps=[
                "Tighten the grading prompt to explicitly require JSON output.",
                "Lower the temperature for the grading model.",
            ],
            tips=["Wrapping the prompt in ```json fences often increases compliance."],
        )

    for json_str in json_blocks:
        if '"result"' not in json_str:
            continue

        try:
            parsed = json.loads(json_str)
            if "result" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    for json_str in json_blocks:
        result_match = re.search(r'"result"\s*:\s*([01])', json_str)
        if result_match:
            return {
                "result": int(result_match.group(1)),
                "reason": "JSON parse failed, extracted result only"
            }
    
    result_match = re.search(r'"result"\s*:\s*([01])', content_cleaned)
    if result_match:
        return {
            "result": int(result_match.group(1)),
            "reason": "JSON parse failed, extracted result only from raw content"
        }

    raise SystemExecutionError(
        title="Missing 'result' field in LLM response",
        summary="The grading model returned JSON, but none of the objects contained a 'result' key.",
        context={"response_preview": content[:200].strip() or "<empty>"},
        probable_causes=[
            "The rubric instructions do not emphasize the 'result' field strongly enough.",
            "The model returned diagnostic information instead of the expected schema.",
        ],
        next_steps=[
            "Ensure the prompt includes an explicit JSON schema with the 'result' field.",
            "Inspect the raw response stored in results to refine the prompt.",
        ],
        tips=["Use ConfigValidator to preflight prompts before large runs."],
    )


async def grade_single_criterion(
    solution: str,
    criterion: Dict[str, Any],
    criterion_key: str,
    criterion_index: int,
    total_criteria: int,
    grading_model_config: Dict[str, Any],
    grading_prompt_template: str,
    response_images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Grades solution against single criterion."""
    start_time = time.time()
    criterion_description = criterion.get("description", "")
    logger.info(f"[criterion {criterion_index}/{total_criteria}] Grading {criterion_key}")

    try:
        from call_llm import (
            LLMMessage,
            LLMRequest,
            LLMRole,
            ProcessedImage,
            create_litellm_client,
        )

        # Prepare format arguments: default ones + all keys from the criterion dict
        format_args = criterion.copy()
        format_args.update({
            "criterion_description": criterion_description,
            "solution": solution,
        })
        
        full_prompt = grading_prompt_template.format(**format_args)
        messages = [LLMMessage(role=LLMRole.USER, content=full_prompt)]

        processed_images: List[ProcessedImage] = []
        if response_images and len(response_images) > 0:
            logger.info(
                f"[Vision Grading] Processing {len(response_images)} images for {criterion_key}"
            )

        request = LLMRequest(
            model=grading_model_config["model_id"],
            messages=messages,
            temperature=grading_model_config["temperature"],
            max_tokens=grading_model_config["max_tokens"],
            api_key=grading_model_config.get("api_key"),
        )

        client = create_litellm_client()
        max_retries = 10

        for attempt in range(max_retries):
            try:
                response = await client.call_llm(request)
            except Exception as exc:
                # Handle network/client errors that don't return a response object
                logger.warning(f"LLM call raised exception (attempt {attempt+1}/{max_retries}): {exc}")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter: 1s, 2s, 4s... capped at 30s
                    delay = min(30, (2 ** attempt) + random.uniform(0, 1))
                    logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                raise  # Re-raise if it's the last attempt

            if response.success:
                parsed = parse_llm_json_response(response.content)
                result_value = parsed.get("result")
                reason = parsed.get("reason", "")

                if result_value is None:
                    raise SystemExecutionError(
                        title="LLM response missing grading decision",
                        summary="The parsed JSON did not include a 'result' value to determine pass/fail.",
                        context={"response_json": parsed},
                        probable_causes=[
                            "The model deviated from the required JSON schema.",
                            "Prompt template was modified and no longer references 'result'.",
                        ],
                        next_steps=[
                            "Re-run a single record with logging enabled to inspect the raw response.",
                            "Reinforce the JSON schema in your grading prompt template.",
                        ],
                    )

                is_met = result_value in [1, "1", True]
                execution_time = time.time() - start_time

                return {
                    "criterion_key": criterion_key,
                    "description": criterion_description,
                    "weight": criterion.get("weight", ""),
                    "sources": criterion.get("sources", ""),
                    "criterion_type": criterion.get("criterion_type", []),
                    "dependent_criteria": criterion.get("dependent_criteria", []),
                    "autorating": is_met,
                    "reason": reason,
                    "criterion_index": criterion_index,
                    "grading_success": True,
                    "grading_error": None,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_cost": response.usage.total_cost if response.usage else 0.0,
                    "execution_time_seconds": execution_time,
                    "raw_response": response.content,
                }

            # Exponential backoff for failed response (e.g. API error, rate limit)
            if attempt < max_retries - 1:
                delay = min(30, (2 ** attempt) + random.uniform(0, 1))
                logger.warning(f"Attempt {attempt+1} failed. Retrying in {delay:.2f}s... Error: {response.error}")
                await asyncio.sleep(delay)

        raise SystemExecutionError(
            title="Grading LLM call failed repeatedly",
            summary=f"Grading failed after {max_retries} attempts. Last error: {response.error}",
            context={
                "model_id": grading_model_config.get("model_id"),
                "criterion_key": criterion_key,
            },
            probable_causes=[
                "The grading provider rejected the request (rate limit, auth failure, etc.).",
                "Network connectivity issues prevented the request from completing.",
                "The model name is incorrect or unavailable in this region.",
            ],
            next_steps=[
                "Verify the API key has access to the requested model.",
                "Check provider status dashboards for outages or rate limits.",
                "Reduce concurrency or increase retries for long-running batches.",
            ],
            tips=["Set `retries` higher or implement exponential backoff when grading at scale."],
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[criterion {criterion_index}] Grading failed for {criterion_key}: {e}")
        logger.error(f"[criterion {criterion_index}] Exception type: {type(e).__name__}")
        logger.error(f"[criterion {criterion_index}] Full traceback:", exc_info=True)

        return {
            "criterion_key": criterion_key,
            "description": criterion_description,
            "weight": criterion.get("weight", ""),
            "sources": criterion.get("sources", ""),
            "criterion_type": criterion.get("criterion_type", []),
            "dependent_criteria": criterion.get("dependent_criteria", []),
            "autorating": False,
            "reason": f"Grading failed: {str(e)}",
            "criterion_index": criterion_index,
            "grading_success": False,
            "grading_error": str(e),
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0,
            "execution_time_seconds": execution_time,
            "raw_response": "",
        }


async def grade_solution_against_rubric(
    solution: str,
    rubric: Dict[str, Any],
    grading_model_config: Dict[str, Any],
    grading_prompt_template: str,
    response_images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Grades solution against rubric."""
    start_time = time.time()

    if not solution or solution.strip() == "":
        return {
            "points_earned": 0.0,
            "points_possible": len(rubric),
            "percentage_score": 0.0,
            "criteria_results": [],
            "grading_error": "Empty solution",
            "execution_time_seconds": 0.0,
            "total_grading_tokens": 0,
            "total_grading_cost": 0.0,
        }

    if isinstance(solution, str) and solution.strip().lower().startswith("error:"):
        return {
            "points_earned": 0.0,
            "points_possible": len(rubric),
            "percentage_score": 0.0,
            "criteria_results": [],
            "grading_error": solution.strip(),
            "execution_time_seconds": 0.0,
            "total_grading_tokens": 0,
            "total_grading_cost": 0.0,
        }

    try:
        total_criteria = len(rubric)
        logger.info("Grading solution against %d criteria", total_criteria)

        max_retries = 10
        criterion_results: List[Optional[Dict[str, Any]]] = []
        failed_criteria: List[Any] = []

        for retry_attempt in range(max_retries):
            if retry_attempt == 0:
                criteria_to_grade = [
                    (idx, criterion_key, criterion_data)
                    for idx, (criterion_key, criterion_data) in enumerate(rubric.items(), 1)
                ]
            else:
                if not failed_criteria:
                    break

                logger.info(
                    "Retry %d: Retrying %d failed criteria",
                    retry_attempt,
                    len(failed_criteria),
                )
                criteria_to_grade = failed_criteria
                failed_criteria = []

            tasks = []
            task_indices = []

            for idx, criterion_key, criterion_data in criteria_to_grade:
                task = grade_single_criterion(
                    solution=solution,
                    criterion=criterion_data,
                    criterion_key=criterion_key,
                    criterion_index=idx,
                    total_criteria=total_criteria,
                    grading_model_config=grading_model_config,
                    grading_prompt_template=grading_prompt_template,
                    response_images=response_images,
                )
                tasks.append(task)
                task_indices.append((idx, criterion_key, criterion_data))

            logger.info(
                "Executing %d grading tasks in parallel (attempt %d/%d)...",
                len(tasks),
                retry_attempt + 1,
                max_retries,
            )
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                idx, criterion_key, criterion_data = task_indices[i]

                if isinstance(result, Exception):
                    logger.warning("Criterion %s raised exception: %s", criterion_key, result)
                    failed_criteria.append((idx, criterion_key, criterion_data))

                    if retry_attempt == 0:
                        criterion_results.append(None)
                elif result.get("grading_success") is False:
                    logger.warning(
                        "Criterion %s grading failed: %s",
                        criterion_key,
                        result.get("grading_error"),
                    )
                    failed_criteria.append((idx, criterion_key, criterion_data))

                    if retry_attempt == 0:
                        criterion_results.append(result)
                    else:
                        criterion_results[idx - 1] = result
                else:
                    if retry_attempt == 0:
                        criterion_results.append(result)
                    else:
                        criterion_results[idx - 1] = result

            if not failed_criteria:
                logger.info("All criteria graded successfully")
                break

            if retry_attempt < max_retries - 1:
                await asyncio.sleep(1.0)

        for idx, criterion_key, criterion_data in failed_criteria:
            logger.error("Criterion %s failed after %d attempts", criterion_key, max_retries)
            criterion_results[idx - 1] = {
                "criterion_key": criterion_key,
                "description": criterion_data.get("description", ""),
                "weight": criterion_data.get("weight", ""),
                "sources": criterion_data.get("sources", ""),
                "criterion_type": criterion_data.get("criterion_type", []),
                "dependent_criteria": criterion_data.get("dependent_criteria", []),
                "autorating": False,
                "reason": f"Grading failed after {max_retries} retry attempts",
                "grading_error": f"Failed after {max_retries} retries",
                "grading_success": False,
            }

        valid_results = [r for r in criterion_results if r is not None]

        total_possible = len(valid_results)
        total_earned = sum(1 for r in valid_results if r.get("autorating", False))
        percentage_score = (total_earned / total_possible * 100) if total_possible > 0 else 0

        execution_time = time.time() - start_time
        logger.info(
            "Grading complete: %d/%d (%.1f%%) in %.2fs",
            total_earned,
            total_possible,
            percentage_score,
            execution_time,
        )

        return {
            "points_earned": float(total_earned),
            "points_possible": total_possible,
            "percentage_score": round(percentage_score, 2),
            "criteria_results": valid_results,
            "grading_error": None,
            "execution_time_seconds": execution_time,
            "total_grading_tokens": sum(r.get("tokens_used", 0) for r in valid_results),
            "total_grading_cost": sum(r.get("total_cost", 0.0) for r in valid_results),
        }

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("Grading exception: %s", e)

        return {
            "points_earned": 0.0,
            "points_possible": len(rubric),
            "percentage_score": 0.0,
            "criteria_results": [],
            "grading_error": str(e),
            "execution_time_seconds": execution_time,
            "total_grading_tokens": 0,
            "total_grading_cost": 0.0,
        }


async def run_grading_task_async(task: Union[GradingTask, Dict[str, Any]]) -> GradingResult:
    """Runs grading task asynchronously."""
    grading_task = task if isinstance(task, GradingTask) else GradingTask(**task)

    ConfigValidator.validate_api_keys_for_models([grading_task.grading_model])
    prompt_template = _resolve_prompt_template(grading_task.grading_prompt_template)

    logger.info("Starting grading task (criteria=%d)", len(grading_task.rubric))

    result_dict = await grade_solution_against_rubric(
        solution=grading_task.solution,
        rubric=grading_task.rubric,
        grading_model_config=grading_task.grading_model.model_dump(),
        grading_prompt_template=prompt_template,
        response_images=grading_task.response_images,
    )

    return GradingResult(**result_dict)


def run_grading_task(task: Union[GradingTask, Dict[str, Any]]) -> GradingResult:
    """Synchronous wrapper for grading task."""
    return asyncio.run(run_grading_task_async(task))


def _resolve_prompt_template(template: Optional[str]) -> str:
    if not template:
        return DEFAULT_GRADING_PROMPT

    candidate = Path(template)
    if candidate.exists():
        return candidate.read_text()

    return template
