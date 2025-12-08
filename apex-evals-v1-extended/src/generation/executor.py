import asyncio
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from parser import parse_documents
from errors import UserInputError
from handler.validator import ConfigValidator

from .config import Attachment, GenerationResult, GenerationTask, ModelConfig
from call_llm import LLMMessage, LLMRequest, LLMRole, create_litellm_client

if TYPE_CHECKING:
    from call_llm.base import LLMResponse

logger = logging.getLogger(__name__)

RETRYABLE_ERROR_TYPES = {
    "rate_limit_error",
    "timeout_error",
    "api_error",
    "server_error",
    "temporary_error",
    "network_error",
}

TRANSIENT_ERROR_SUBSTRINGS = (
    "temporarily unavailable",
    "try again",
    "connection reset",
    "gateway timeout",
    "502",
    "503",
    "504",
)


async def run_generation_task_async(task: Union[GenerationTask, Dict[str, Any]]) -> GenerationResult:
    """Runs generation task asynchronously."""
    generation_task = task if isinstance(task, GenerationTask) else GenerationTask(**task)

    ConfigValidator.validate_api_keys_for_models(generation_task.models)
    attachments_present = bool(generation_task.attachments)
    ConfigValidator.validate_parser_api_key(generation_task.parsing_method, attachments_present)

    parsed_attachments = await _parse_attachments(generation_task)
    final_prompt = _build_final_prompt(generation_task, parsed_attachments)

    results: List[Dict[str, Any]] = []
    for model_config in generation_task.models:
        model_results = await _execute_model_runs(
            task=generation_task,
            model_config=model_config,
            final_prompt=final_prompt,
        )
        results.extend(model_results)

    cleaned_results = [_clean_result(result) for result in results]
    completed = sum(1 for r in cleaned_results if r.get("success"))
    failed = len(cleaned_results) - completed
    total_tokens = sum(r.get("tokens_used", 0) for r in cleaned_results if r.get("success"))
    total_cost = sum(r.get("total_cost", 0.0) for r in cleaned_results if r.get("success"))

    logger.info(
        "Generation complete (completed=%s failed=%s tokens=%s cost=$%.4f)",
        completed,
        failed,
        total_tokens,
        total_cost,
    )

    return GenerationResult(
        results=cleaned_results,
        completed=completed,
        failed=failed,
        total_tokens=total_tokens,
        total_cost=total_cost,
    )


def run_generation_task(task: Union[GenerationTask, Dict[str, Any]]) -> GenerationResult:
    """Synchronous wrapper for generation task."""
    return asyncio.run(run_generation_task_async(task))


def _should_retry_error(error_type: Optional[str], error_message: Optional[str]) -> bool:
    """Checks if error is retryable."""
    if error_type and error_type.lower() in RETRYABLE_ERROR_TYPES:
        return True
    if error_message:
        lowered = error_message.lower()
        return any(fragment in lowered for fragment in TRANSIENT_ERROR_SUBSTRINGS)
    return False


def _build_success_result(
    *,
    model_id: str,
    run_number: int,
    final_response: str,
    response: "LLMResponse",
    execution_time: float,
    temperature: float,
    max_tokens: Optional[int],
    final_prompt: str,
    started_at: str,
) -> Dict[str, Any]:
    """Creates success result."""
    usage = getattr(response, "usage", None)
    return {
        "success": True,
        "model_id": model_id,
        "run_number": run_number,
        "response": final_response,
        "raw_response": response.content,
        "execution_time_seconds": execution_time,
        "tokens_used": usage.total_tokens if usage else 0,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "total_cost": usage.total_cost if usage else 0.0,
        "per_token_cost": usage.per_token_cost if usage else 0.0,
        "api_provider": response.provider.value if response.provider else "unknown",
        "error_message": "",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "_final_prompt": final_prompt,
        "_parsed_attachments": "",
    }


def _build_failure_result(
    *,
    model_id: str,
    run_number: int,
    error_message: str,
    execution_time: float,
    temperature: float,
    max_tokens: Optional[int],
    started_at: str,
) -> Dict[str, Any]:
    """Creates failure result."""
    return {
        "success": False,
        "model_id": model_id,
        "run_number": run_number,
        "response": "",
        "raw_response": "",
        "execution_time_seconds": execution_time,
        "tokens_used": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "per_token_cost": 0.0,
        "api_provider": "unknown",
        "error_message": error_message,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "_final_prompt": "",
    }


async def execute_single_llm_call(
    model_id: str,
    run_number: int,
    final_prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    max_input_tokens: Optional[int],
    retries: int,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    is_custom_model: bool = False,
    custom_model_config: Optional[Dict[str, Any]] = None,
    model_configs: Optional[Dict[str, Any]] = None,
    enable_thinking: Optional[bool] = None,
    thinking_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Executes single LLM call with retries."""
    start_time = time.time()
    started_at_iso = datetime.fromtimestamp(start_time).isoformat()

    try:
        logger.debug(f"Final prompt ready for {model_id}/run{run_number}: {len(final_prompt)} chars")

        messages = []
        if system_prompt:
            messages.append(LLMMessage(role=LLMRole.SYSTEM, content=system_prompt))
        messages.append(LLMMessage(role=LLMRole.USER, content=final_prompt))

        request = LLMRequest(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            api_key=api_key,
            is_custom_model=is_custom_model,
            custom_model_config=custom_model_config,
            model_configs=model_configs,
            enable_thinking=enable_thinking,
            thinking_tokens=thinking_tokens,
        )

        client = create_litellm_client()
        last_response = None

        for attempt in range(retries + 1):
            logger.debug(f"Attempt {attempt + 1}/{retries + 1} for {model_id}/run{run_number}")

            response = await client.call_llm(request)
            last_response = response

            if response.success:
                final_response = response.content
                execution_time = time.time() - start_time

                result = _build_success_result(
                    model_id=model_id,
                    run_number=run_number,
                    final_response=final_response,
                    response=response,
                    execution_time=execution_time,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    final_prompt=final_prompt,
                    started_at=started_at_iso,
                )

                logger.info(
                    "Success: %s/run%s (%.2fs, %s tokens)",
                    model_id,
                    run_number,
                    execution_time,
                    result["tokens_used"],
                )
                return result

            error_type = (response.error_type or "unknown").lower()
            error_message = response.error or ""
            is_retryable = _should_retry_error(error_type, error_message)
            is_last_attempt = attempt == retries

            if is_retryable and not is_last_attempt:
                base_delay = min(2**attempt, 20)
                jitter = random.uniform(0, 1)
                delay = base_delay + jitter
                logger.warning(
                    "Retryable error for %s/run%s (type=%s): %s. Retrying in %.2fs",
                    model_id,
                    run_number,
                    error_type,
                    error_message,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            break

        execution_time = time.time() - start_time
        error_msg = last_response.error if last_response else "Unknown error"

        result = _build_failure_result(
            model_id=model_id,
            run_number=run_number,
            error_message=error_msg,
            execution_time=execution_time,
            temperature=temperature,
            max_tokens=max_tokens,
            started_at=started_at_iso,
        )

        logger.error("Failed: %s/run%s - %s", model_id, run_number, error_msg)
        return result

    except Exception as exc:
        execution_time = time.time() - start_time
        logger.error("Exception: %s/run%s - %s", model_id, run_number, exc)

        return _build_failure_result(
            model_id=model_id,
            run_number=run_number,
            error_message=str(exc),
            execution_time=execution_time,
            temperature=temperature,
            max_tokens=max_tokens,
            started_at=started_at_iso,
        )


async def _execute_model_runs(
    task: GenerationTask,
    model_config: ModelConfig,
    final_prompt: str,
) -> List[Dict[str, Any]]:
    run_tasks = []
    for run_number in range(1, model_config.number_of_runs + 1):
        run_tasks.append(
            execute_single_llm_call(
                model_id=model_config.model_id,
                run_number=run_number,
                final_prompt=final_prompt,
                system_prompt=task.system_prompt,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                max_input_tokens=model_config.max_input_tokens,
                retries=task.retries,
                api_key=model_config.api_key,
                is_custom_model=model_config.is_custom_model,
                custom_model_config=model_config.custom_model_config,
                model_configs=model_config.model_configs,
                enable_thinking=model_config.enable_thinking,
                thinking_tokens=model_config.thinking_tokens,
            )
        )

    results: List[Dict[str, Any]] = []
    gathered = await asyncio.gather(*run_tasks, return_exceptions=True)
    for run_number, result in enumerate(gathered, start=1):
        if isinstance(result, Exception):
            logger.error(
                "Model %s run %s failed with exception: %s",
                model_config.model_id,
                run_number,
                result,
            )
            failure = {
                "success": False,
                "model_id": model_config.model_id,
                "run_number": run_number,
                "response": "",
                "raw_response": "",
                "error_message": str(result),
                "tokens_used": 0,
                "total_cost": 0.0,
            }
            results.append(failure)
        else:
            results.append(result)

    return results


def _build_final_prompt(task: GenerationTask, parsed_attachments: str) -> str:
    prompt_with_fields = task.prompt

    if parsed_attachments.strip():
        return (
            f"{prompt_with_fields}\n\n"
            "==== Attached files content: ====\n\n"
            f"{parsed_attachments}"
        )
    return prompt_with_fields


async def _parse_attachments(task: GenerationTask) -> str:
    attachments = _collect_attachments(task)
    if not attachments:
        return ""

    if not task.parsing_method:
        logger.info("Attachments provided but no parsing_method configured; skipping parsing.")
        return ""

    try:
        parsed_text = await parse_documents(
            attachments=attachments,
            parser_name=task.parsing_method,
            use_cache=task.cache_parsed_documents,
        )
        logger.info("Parsed %s characters from attachments", len(parsed_text))
        return parsed_text
    except Exception as exc:
        logger.error("Failed to parse attachments: %s", exc)
        return ""


def _collect_attachments(task: GenerationTask) -> List[Dict[str, str]]:
    attachments: List[Dict[str, str]] = []

    if task.attachments:
        for attachment in task.attachments:
            if isinstance(attachment, Attachment):
                attachments.append(attachment.model_dump())
            elif isinstance(attachment, dict):
                filename = attachment.get("filename")
                url = attachment.get("url")
                if filename and url:
                    attachments.append({"filename": filename, "url": url})

    return attachments


def _clean_result(result: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(result)
    final_prompt = cleaned.pop("_final_prompt", "")
    parsed_attachments = cleaned.pop("_parsed_attachments", "")

    if final_prompt:
        cleaned["final_prompt"] = final_prompt
    if parsed_attachments:
        cleaned["parsed_attachments"] = parsed_attachments

    return cleaned

