"""
LiteLLM Client for OpenAI, Anthropic, Google, xAI, and custom providers.
"""

import logging
import os
import re
import time
import traceback
from typing import Dict, List, Any, Optional

import litellm
from litellm import acompletion, token_counter, cost_per_token, completion_cost
from litellm.utils import trim_messages

from .base import (
    BaseLLMProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMMessage,
    LLMRole,
)
from .model_limits import get_context_limit_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s %(filename)s:%(lineno)d - %(message)s',
)
logger = logging.getLogger(__name__)


def clean_think_tags(content: str) -> str:
    """Removes <think> tags from content."""
    if not content:
        return content

    # Remove <think>...</think> content (multiline, case-insensitive)
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class LiteLLMClient(BaseLLMProvider):
    """Universal LLM client using LiteLLM."""

    # Model mapping: provider -> list of models (without provider prefix)
    MODEL_MAPPINGS = {
        # OpenAI models
        "openai": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
            "gpt-5",
            "o4-mini-deep-research-2025-06-26",
            "o3-mini",
            "o3",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.2-pro"
        ],

        # Anthropic models (Claude)
        "anthropic": [
            "claude-4",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-4-sonnet-20250722",
            "claude-4-haiku-20250722",
            "claude-4-opus-20250722",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
        ],

        # Google models (Gemini)
        "google": [
            "gemini-3-pro-preview",
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-2.0-flash-exp",
            "gemini-2.5-pro",
            "gemini-2.5-flash"
        ],

        # xAI models (Grok)
        "xai": [
            "grok-beta",
            "grok-code-fast-1",
            "grok-4-0709",
            "grok-3",
            "grok-3-mini",
            "grok-3-mini-beta"
        ],

        # Fireworks models
        "fireworks": [
            # Qwen models
            "accounts/fireworks/models/qwen3-30b-a3b-thinking-2507",
            "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",

            # Llama4 models
            "accounts/fireworks/models/llama4-scout-instruct-basic",
            "accounts/fireworks/models/llama4-maverick-instruct-basic",

            # Deepseek models
            "accounts/fireworks/models/deepseek-v3p1",
            "accounts/fireworks/models/deepseek-v3p1-terminus",
            "accounts/fireworks/models/deepseek-r1-0528"
        ],
    }

    # Provider prefix mapping for LiteLLM routing
    PROVIDER_PREFIXES = {
        "openai": "",  # No prefix for OpenAI
        "anthropic": "anthropic",
        "google": "gemini",
        "xai": "xai",
        "fireworks": "fireworks_ai",
        "custom": ""
    }

    # Provider mapping: model prefix -> provider enum
    PROVIDER_MAPPINGS = {
        "gpt": LLMProvider.OPENAI,
        "o1": LLMProvider.OPENAI,
        "o3": LLMProvider.OPENAI,
        "o4": LLMProvider.OPENAI,
        "claude": LLMProvider.ANTHROPIC,
        "gemini": LLMProvider.GOOGLE,
        "grok": LLMProvider.XAI,
        "xai": LLMProvider.XAI,
    }

    def __init__(self, api_key: str = "", **kwargs):
        """Initializes LiteLLM client."""
        super().__init__(api_key or "", **kwargs)

        # Configure LiteLLM settings
        litellm.drop_params = True  # Automatically drop unsupported parameters

    @property
    def provider_name(self) -> LLMProvider:
        """Default provider."""
        return LLMProvider.OPENAI

    @property
    def supported_models(self) -> List[str]:
        """Gets supported models."""
        all_models = []
        for provider, models in self.MODEL_MAPPINGS.items():
            all_models.extend(models)
        return all_models

    def validate_model(self, model: str) -> bool:
        """Checks if model is supported."""
        for provider, models in self.MODEL_MAPPINGS.items():
            if model in models:
                return True
        return False

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Gets supported models."""
        all_models = []
        for provider, models in cls.MODEL_MAPPINGS.items():
            all_models.extend(models)
        return all_models

    @classmethod
    def is_model_supported(cls, model: str) -> bool:
        """Checks if model is supported."""
        for provider, models in cls.MODEL_MAPPINGS.items():
            if model in models:
                return True
        return False

    def get_provider_for_model(self, model: str) -> LLMProvider:
        """Determines provider for model."""
        for prefix, provider in self.PROVIDER_MAPPINGS.items():
            if model.startswith(prefix):
                return provider

        # Check if model is in fireworks
        if model.startswith("accounts/fireworks"):
            return LLMProvider.FIREWORKS

        return LLMProvider.CUSTOM

    def get_litellm_model_name(self, request: LLMRequest) -> str:
        """Converts internal model name to LiteLLM format."""
        model = request.model
        prefix = ""

        # Handle custom models
        if request.is_custom_model and request.custom_model_config:
            prefix = request.custom_model_config.get('prefix', '')
        else:
            # For standard models, find which provider this model belongs to
            for provider, models in self.MODEL_MAPPINGS.items():
                if model in models:
                    prefix = self.PROVIDER_PREFIXES.get(provider, "")
                    break

        # Only add "/" if prefix exists and is not empty
        if prefix and prefix.strip():
            return f"{prefix}/{model}"
        else:
            return model

    def _estimate_token_count(self, messages: List[LLMMessage], model_name_with_prefix: str) -> int:
        """Estimates token count."""
        messages_dict = self.prepare_messages_for_litellm(messages)
        try:
            total_tokens = token_counter(model=model_name_with_prefix, messages=messages_dict)
            logger.debug(f"Token count for {model_name_with_prefix}: {total_tokens}")
            return total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback: rough estimation (4 chars = 1 token)
            return sum(len(msg.content) // 4 for msg in messages)

    def prepare_messages_for_litellm(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Converts messages to LiteLLM format."""
        litellm_messages = []

        for msg in messages:
            message = {
                "role": msg.role.value,
                "content": msg.content
            }

            # Check if there are inline images (passed via __dict__)
            inline_images = msg.__dict__.get('inline_images', None)

            # Handle inline images or processed_images if present
            if inline_images or msg.processed_images:
                # For vision models, convert to multi-modal format
                content_parts = [{"type": "text", "text": msg.content}]

                # Handle inline images (direct URLs)
                if inline_images:
                    for img in inline_images:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": img.url}
                        })

                # Handle processed images (base64 encoded)
                if msg.processed_images:
                    for image in msg.processed_images:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.mime_type};base64,{image.base64_data}"
                            }
                        })

                message["content"] = content_parts

            litellm_messages.append(message)

        return litellm_messages

    def map_parameters_for_litellm(self, request: LLMRequest) -> Dict[str, Any]:
        """Maps parameters to LiteLLM format."""
        params = {}

        # If model_configs is provided, use it directly as the base
        if request.model_configs:
            params.update(request.model_configs)

        # Backward compatible: individual fields override model_configs if explicitly set
        if request.temperature is not None:
            params["temperature"] = request.temperature

        # Handle token limits
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            params["stop"] = request.stop

        # Claude thinking parameters
        if request.enable_thinking is not None and request.enable_thinking:
            thinking_config = {"type": "enabled"}
            if request.thinking_tokens is not None:
                thinking_config["budget_tokens"] = request.thinking_tokens
            params["thinking"] = thinking_config

        if request.provider_params:
            params.update(request.provider_params)

        return params

    def classify_litellm_error(self, error: Exception) -> str:
        """Classifies LiteLLM errors."""
        error_str = str(error).lower()
        error_class = error.__class__.__name__

        # Rate limit errors
        if any(pattern in error_str for pattern in [
            "rate limit", "quota", "too many requests", "429"
        ]) or "RateLimitError" in error_class:
            return "rate_limit_error"

        # Authentication errors
        if any(pattern in error_str for pattern in [
            "unauthorized", "invalid api key", "authentication", "401"
        ]) or "AuthenticationError" in error_class:
            return "authentication_error"

        # Permission errors
        if any(pattern in error_str for pattern in [
            "permission", "forbidden", "403"
        ]) or "PermissionError" in error_class:
            return "permission_error"

        # Invalid model/parameter errors
        if any(pattern in error_str for pattern in [
            "invalid model", "model not found", "unsupported", "400"
        ]) or "BadRequestError" in error_class:
            return "invalid_request_error"

        # Timeout errors
        if any(pattern in error_str for pattern in [
            "timeout", "timed out", "connection"
        ]) or "TimeoutError" in error_class:
            return "timeout_error"

        return "api_error"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Checks if error is rate limit."""
        error_str = str(error).lower()

        # Check for specific exception types first
        if hasattr(error, '__class__'):
            error_class_name = error.__class__.__name__
            if any(rate_limit_class in error_class_name for rate_limit_class in [
                "RateLimitError", "ResourceExhausted", "TooManyRequestsError"
            ]):
                return True

        # Check for common rate limit patterns
        rate_limit_patterns = [
            "rate limit", "rate-limit", "rate_limit",
            "too many requests", "too_many_requests",
            "429", "quota exceeded", "exceeded your current quota",
            "resource exhausted", "throttled", "throttling",
        ]

        return any(pattern in error_str for pattern in rate_limit_patterns)

    def _create_error_response(self, request: LLMRequest, error: str, error_type: str) -> LLMResponse:
        """Creates error response."""
        logger.error(f"LLM call failed - Model: {request.model}, Error: {error}, Type: {error_type}")

        provider = self.get_provider_for_model(request.model)

        return LLMResponse(
            content="",
            model=request.model,
            provider=provider,
            request_id=request.request_id,
            response_time_ms=0,
            success=False,
            error=error,
            error_type=error_type
        )

    async def call_llm(self, request: LLMRequest) -> LLMResponse:
        """Calls LiteLLM API."""
        start_time = time.time()
        model_name_with_prefix = self.get_litellm_model_name(request)

        params = self.map_parameters_for_litellm(request)
        messages_dict = self.prepare_messages_for_litellm(request.messages)
        
        # Use API key from request if provided (overrides environment variables)
        if request.api_key:
            params['api_key'] = request.api_key

        # Use max_input_tokens for message trimming if available
        trim_kwargs = {}
        if request.max_input_tokens is not None:
            trim_kwargs['max_tokens'] = request.max_input_tokens
        elif params.get('max_tokens'):
            trim_kwargs['max_tokens'] = params['max_tokens']

        messages = trim_messages(messages_dict, model_name_with_prefix, **trim_kwargs)

        params['model'] = model_name_with_prefix
        params['messages'] = messages

        before_size = sum(len(msg.get('content', '')) for msg in messages_dict if isinstance(msg.get('content'), str))
        after_size = sum(len(msg.get('content', '')) for msg in messages if isinstance(msg.get('content'), str))
        logger.info(f"Trim messages - Before: {before_size} chars, After: {after_size} chars")

        provider = self.get_provider_for_model(request.model)

        try:
            # Route 1: Custom model
            if request.is_custom_model:
                custom_model_config = request.custom_model_config or {}
                filtered_config = {k: v for k, v in custom_model_config.items() if k not in ['prefix']}
                params.update(**filtered_config)
                response = await acompletion(**params)

            # Route 2: Standard provider models
            else:
                if not self.validate_model(request.model):
                    return self._create_error_response(
                        request,
                        f"Model {request.model} is not supported",
                        "unsupported_model"
                    )

                # Use direct API call (requires env vars or params['api_key'])
                response = await acompletion(**params)

            response_time_ms = int((time.time() - start_time) * 1000)

            # Extract usage information
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            thinking_tokens = 0
            total_cost = 0.0
            completion_cost_per_token = 0.0

            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                total_tokens = getattr(response.usage, 'total_tokens', 0) or 0
                thinking_tokens = getattr(response.usage, 'thinking_tokens', 0) or 0

                # Calculate cost
                if not request.is_custom_model:
                    try:
                        total_cost = completion_cost(completion_response=response)
                        _, completion_cost_per_token = cost_per_token(
                            model=params['model'],
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens
                        )
                    except Exception as e:
                        logger.debug(f"Cost calculation failed: {e}")

            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                thinking_tokens=thinking_tokens,
                total_cost=total_cost,
                per_token_cost=completion_cost_per_token
            )

            # Clean <think> tags from response content (only for custom models)
            content = response.choices[0].message.content or ""
            if request.is_custom_model:
                content = clean_think_tags(content)

            return LLMResponse(
                content=content,
                model=request.model,
                provider=provider,
                request_id=request.request_id,
                usage=usage,
                response_time_ms=response_time_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                success=True
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"Unexpected error in LLM call: {error_str}")
            traceback.print_exc()

            # Classify the error type
            error_type = self.classify_litellm_error(e)

            # If it's a rate limit error, provide more specific messaging
            if error_type == "rate_limit_error" or self._is_rate_limit_error(e):
                return self._create_error_response(
                    request,
                    f"Rate limit or quota exceeded for {provider.value}: {error_str}",
                    "rate_limit_error"
                )

            return self._create_error_response(
                request,
                f"Error making LLM call: {error_str}",
                error_type
            )

    async def cleanup(self):
        """Cleans up resources."""
        pass


# Factory functions
def create_litellm_client(api_key: str = "", **kwargs) -> LiteLLMClient:
    """Creates LiteLLM client."""
    return LiteLLMClient(api_key=api_key, **kwargs)


def create_litellm_client_from_env(**kwargs) -> LiteLLMClient:
    """Creates LiteLLM client using env vars."""
    return LiteLLMClient(**kwargs)

