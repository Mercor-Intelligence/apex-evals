from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class Attachment(BaseModel):
    """Document attachment."""
    filename: str = Field(..., description="Display name")
    url: str = Field(..., description="Public URL")


class ModelConfig(BaseModel):
    """Model configuration."""
    model_id: str = Field(..., description="Model identifier")
    output_fields: List[str] = Field(default_factory=list, description="Output labels")
    number_of_runs: int = Field(default=1, ge=1, description="Run count")
    max_input_tokens: Optional[int] = Field(default=None, description="Max input tokens")
    max_tokens: Optional[int] = Field(default=None, description="Max output tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    use_tools: bool = Field(default=False, description="Enable tools")
    is_custom_model: bool = Field(default=False, description="Is custom model")
    custom_model_config: Optional[Dict[str, Any]] = Field(default=None, description="Custom config")
    model_configs: Optional[Dict[str, Any]] = Field(default=None, description="Additional model params (e.g. reasoning_effort)")
    enable_thinking: Optional[bool] = Field(default=None, description="Enable thinking")
    thinking_tokens: Optional[int] = Field(default=None, description="Thinking budget")
    api_key: Optional[str] = Field(default=None, description="API key")
    
    @field_validator('output_fields')
    @classmethod
    def validate_output_fields(cls, value, info):
        if value:
            return value
        model_id = info.data.get('model_id')
        return [model_id] if model_id else []


class GenerationTask(BaseModel):
    """Generation task input."""
    prompt: str = Field(..., description="Prompt text")
    models: List[ModelConfig] = Field(..., description="Models list")

    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    retries: int = Field(default=3, ge=0, le=10, description="Retry count")

    attachments: Optional[List[Attachment]] = Field(default=None, description="Attachments")
    parsing_method: Optional[str] = Field(default="reducto", description="Parsing method")
    cache_parsed_documents: bool = Field(default=True, description="Cache parsing")

    @model_validator(mode="after")
    def set_defaults(self):
        if not self.models:
            raise ValueError("At least one model configuration is required.")
        return self


class GenerationResult(BaseModel):
    """Generation result."""
    results: List[Dict[str, Any]]
    completed: int
    failed: int
    total_tokens: int
    total_cost: float

