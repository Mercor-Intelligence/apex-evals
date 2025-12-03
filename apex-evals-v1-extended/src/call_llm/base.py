"""
Base models and interfaces.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from datetime import datetime, UTC
from enum import Enum
import uuid


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    FIREWORKS = "fireworks"
    CUSTOM = "custom"


class LLMRole(Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FileAttachment(BaseModel):
    """File attachment metadata."""
    url: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    content: Optional[bytes] = None
    base64_data: Optional[str] = None


class ProcessedImage(BaseModel):
    """Processed image data."""
    base64_data: str
    mime_type: str = "image/png"
    page_number: Optional[int] = None


class LLMMessage(BaseModel):
    """Conversation message."""
    role: LLMRole
    content: str
    attachments: Optional[List[FileAttachment]] = None
    processed_images: Optional[List[ProcessedImage]] = None


class LLMRequest(BaseModel):
    """LLM API request parameters."""
    # Core parameters
    model: str
    messages: List[LLMMessage]

    # Common parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Thinking parameters
    enable_thinking: Optional[bool] = None
    thinking_tokens: Optional[int] = None

    # Web search option
    web_search_enabled: bool = False
    stop: Optional[Union[str, List[str]]] = None

    provider_params: Optional[Dict[str, Any]] = {}

    # Model configuration
    model_configs: Optional[Dict[str, Any]] = None

    # Custom model configuration
    is_custom_model: bool = False
    custom_model_config: Optional[Dict[str, Any]] = None
    
    # API key
    api_key: Optional[str] = None
    
    # Request metadata
    request_id: Optional[str] = None
    timeout: int = 3000

    def model_post_init(self, __context: Dict[str, Any]) -> None:
        """Generates ID if missing."""
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


class LLMUsage(BaseModel):
    """Usage and cost stats."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    per_token_cost: Optional[float] = None


class LLMResponse(BaseModel):
    """LLM API response."""
    # Core data
    content: str
    model: str
    provider: LLMProvider

    # Reference
    request_id: str

    # Stats
    usage: Optional[LLMUsage] = None
    response_time_ms: int

    # Raw data
    raw_response: Optional[Dict[str, Any]] = None

    # Status
    success: bool = True
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None

    def model_post_init(self, __context: Dict[str, Any]) -> None:
        """Sets creation time if missing."""
        if not self.created_at:
            self.created_at = datetime.now(UTC)


class BaseLLMProvider(ABC):
    """Abstract base provider."""

    def __init__(self, api_key: str, **kwargs):
        """Initializes provider."""
        self.api_key = api_key
        self.provider_config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> LLMProvider:
        """Returns provider enum."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Returns supported models."""
        pass

    @abstractmethod
    async def call_llm(self, request: LLMRequest) -> LLMResponse:
        """Executes LLM call."""
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Validates model support."""
        pass

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> List[str]:
        """Returns supported models (static)."""
        pass

    @classmethod
    @abstractmethod
    def is_model_supported(cls, model: str) -> bool:
        """Validates model support (static)."""
        pass

    # Capabilities
    @classmethod
    def supports_files_api(cls, model: str) -> bool:
        """Checks file API support."""
        return False

    def prepare_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Formats messages for provider."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def create_error_response(
        self, request: LLMRequest, error: str, error_type: str = "unknown"
    ) -> LLMResponse:
        """Creates error response."""
        return LLMResponse(
            content="",
            model=request.model,
            provider=self.provider_name,
            request_id=request.request_id,
            response_time_ms=0,
            success=False,
            error=error,
            error_type=error_type,
        )

