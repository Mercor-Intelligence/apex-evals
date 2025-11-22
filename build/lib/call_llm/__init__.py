"""
LLM Client SDK - Universal interface for calling multiple LLM providers

This module provides a unified interface for calling LLMs from OpenAI, Anthropic,
Google, xAI and custom providers using LiteLLM.
"""

from .base import (
    LLMProvider,
    LLMRole,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    FileAttachment,
    ProcessedImage,
    BaseLLMProvider
)

from .litellm_client import (
    LiteLLMClient,
    create_litellm_client,
    create_litellm_client_from_env,
    clean_think_tags
)

from .model_limits import (
    get_context_limit,
    get_context_limit_async,
    get_context_limit_fallback
)

__all__ = [
    # Base models and enums
    "LLMProvider",
    "LLMRole",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "FileAttachment",
    "ProcessedImage",
    "BaseLLMProvider",
    
    # LiteLLM client
    "LiteLLMClient",
    "create_litellm_client",
    "create_litellm_client_from_env",
    "clean_think_tags",
    
    # Model limits
    "get_context_limit",
    "get_context_limit_async",
    "get_context_limit_fallback",
]

__version__ = "0.1.0"

