from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
import json


class GradingModelConfig(BaseModel):
    """Grading model config."""
    model_id: str = Field(default="gemini-2.5-flash", description="Grading model")
    max_tokens: int = Field(default=1000000, description="Max tokens")
    temperature: float = Field(default=0.01, ge=0.0, le=2.0, description="Temperature")
    api_key: Optional[str] = Field(default=None, description="API key")
    use_tools: bool = Field(default=False, description="Enable tools")


class GradingTask(BaseModel):
    """Grading task payload."""
    solution: str = Field(..., description="Response to grade")
    rubric: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Rubric definition")
    grading_model: GradingModelConfig = Field(default_factory=GradingModelConfig, description="Grading model config")
    grading_prompt_template: Optional[str] = Field(
        default=None,
        description="Prompt template or path",
    )
    response_images: Optional[List[str]] = Field(default=None, description="Image URLs")
    
    @model_validator(mode="after")
    def normalize(self):
        if isinstance(self.rubric, str):
            try:
                parsed = json.loads(self.rubric)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Rubric string is not valid JSON: {exc}") from exc
            self.rubric = parsed
        
        if isinstance(self.rubric, list):
            rubric_dict: Dict[str, Any] = {}
            for entry in self.rubric:
                if isinstance(entry, dict):
                    rubric_dict.update(entry)
            self.rubric = rubric_dict
        
        if not isinstance(self.rubric, dict) or not self.rubric:
            raise ValueError("Rubric must resolve to a non-empty dict of criteria.")
        
        return self


class GradingResult(BaseModel):
    """Grading result."""
    points_earned: float
    points_possible: int
    percentage_score: float
    criteria_results: List[Dict[str, Any]]
    grading_error: Optional[str]
    execution_time_seconds: float
    total_grading_tokens: int
    total_grading_cost: float

