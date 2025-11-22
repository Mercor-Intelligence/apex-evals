"""
Model context/window limits.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Fallback limits for common models (in tokens)
_FALLBACK_LIMITS = {
    # Anthropic
    "claude-3-5-sonnet-20241022": 190_000,
    "claude-3-5-haiku-20241022": 190_000,
    "claude-opus-4-1-20250805": 190_000,
    "claude-opus-4-20250514": 190_000,
    "claude-sonnet-4-20250514": 190_000,
    "claude-sonnet-4-5-20250929": 190_000,
    "claude-4": 190_000,
    "claude-4-sonnet-20250722": 190_000,
    "claude-4-haiku-20250722": 190_000,
    "claude-4-opus-20250722": 190_000,
    "claude-3-opus-20240229": 190_000,
    "claude-3-sonnet-20240229": 190_000,
    "claude-3-haiku-20240307": 190_000,
    
    # Google
    "gemini-3-pro-preview": 1_000_000,
    "gemini-2.5-flash": 950_000,
    "gemini-2.5-pro": 950_000,
    "gemini-2.0-flash-exp": 950_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.0-pro": 30_000,
    "gemini-pro": 30_000,
    
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o3-deep-research": 200_000,
    "o4-mini-deep-research-2025-06-26": 128_000,
    "gpt-5": 225_000,
    
    # xAI
    "grok-4-0709": 256_000,
    "grok-code-fast-1": 256_000,
    "grok-3": 131_072,
    "grok-3-mini": 131_072,
    "grok-3-mini-beta": 131_072,
    "grok-beta": 131_072,
    
    # Fireworks
    "qwen3-30b-a3b-thinking-2507": 32_768,
    "qwen3-235b-a22b-thinking-2507": 32_768,
    "llama4-scout-instruct-basic": 128_000,
    "llama4-maverick-instruct-basic": 128_000,
    "deepseek-v3p1": 64_000,
    "deepseek-v3p1-terminus": 64_000,
    "deepseek-r1-0528": 64_000,
}


def get_context_limit(model: str) -> Optional[int]:
    """Gets context limit for a model."""
    return get_context_limit_fallback(model)


async def get_context_limit_async(model: str) -> Optional[int]:
    """Gets context limit asynchronously."""
    return get_context_limit_fallback(model)


def get_context_limit_fallback(model: str) -> Optional[int]:
    """Gets fallback context limit."""
    if not model:
        return None
    
    m = model.lower()
    
    # Direct match
    for key, val in _FALLBACK_LIMITS.items():
        if key.lower() == m:
            logger.debug(f"Found exact context limit for {model}: {val}")
            return val
    
    # Fuzzy match
    for key, val in _FALLBACK_LIMITS.items():
        if key.lower() in m or m in key.lower():
            logger.debug(f"Found fuzzy context limit for {model} (matched {key}): {val}")
            return val
    
    logger.debug(f"No context limit found for model {model}")
    return None


def add_model_limit(model: str, limit: int) -> None:
    """Adds/updates model limit."""
    _FALLBACK_LIMITS[model] = limit
    logger.info(f"Added/updated context limit for {model}: {limit}")


def list_known_models() -> dict[str, int]:
    """Lists known models and limits."""
    return _FALLBACK_LIMITS.copy()

