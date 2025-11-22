"""
Validation utilities.
"""

import os
import logging
from typing import Iterable, Optional, Any
from dotenv import load_dotenv

from errors import UserInputError

load_dotenv()

logger = logging.getLogger(__name__)


class ValidationError(UserInputError):
    """Validation error."""
    
    category = "CONFIG VALIDATION ERROR"


class ConfigValidator:
    """Runtime config validator."""
    
    @staticmethod
    def validate_api_keys_for_models(models: Iterable[Any]) -> None:
        """Checks API key availability."""
        if not models:
            return
        
        missing_keys = []
        
        for model in models:
            model_id = getattr(model, "model_id", None) or (model.get("model_id") if isinstance(model, dict) else None)
            api_key = getattr(model, "api_key", None) or (model.get("api_key") if isinstance(model, dict) else None)
            
            if not model_id:
                continue
            
            if api_key:
                continue
            
            if model_id.startswith(("gpt", "o1", "o3")) and not os.getenv("OPENAI_API_KEY"):
                missing_keys.append(f"OPENAI_API_KEY (for {model_id})")
            elif model_id.startswith("claude") and not os.getenv("ANTHROPIC_API_KEY"):
                missing_keys.append(f"ANTHROPIC_API_KEY (for {model_id})")
            elif model_id.startswith("gemini") and not os.getenv("GOOGLE_API_KEY"):
                missing_keys.append(f"GOOGLE_API_KEY (for {model_id})")
            elif model_id.startswith("grok") and not os.getenv("XAI_API_KEY"):
                missing_keys.append(f"XAI_API_KEY (for {model_id})")
        
        if missing_keys:
            error_msg = "Missing required API keys:\n"
            error_msg += "\n".join(f"  - {key}" for key in missing_keys)
            error_msg += "\n\nSet the variables above or provide `api_key` directly in the request."
            raise ValidationError(error_msg)
        
        logger.info("API key validation passed")
    
    @staticmethod
    def validate_parser_api_key(parsing_method: Optional[str], attachments_present: bool) -> None:
        """Checks parser API key."""
        if not attachments_present:
            return
        
        if parsing_method == "reducto" and not os.getenv("REDUCTO_API_KEY"):
            raise ValidationError(
                "Attachment parsing is enabled but REDUCTO_API_KEY is not set.\n"
                "Either provide the key or skip attachments."
            )
        
        logger.info("Parser API key validation passed")


def validate_environment() -> dict[str, bool]:
    """Checks environment keys."""
    return {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "xAI": bool(os.getenv("XAI_API_KEY")),
        "Reducto": bool(os.getenv("REDUCTO_API_KEY")),
    }


def print_environment_status() -> None:
    """Prints API key status."""
    status = validate_environment()
    
    print("\n" + "=" * 80)
    print("API KEY ENVIRONMENT STATUS")
    print("=" * 80)
    
    for provider, available in status.items():
        icon = "[OK]" if available else "[  ]"
        status_text = "Available" if available else "Not set"
        print(f"  {icon} {provider:12} : {status_text}")
    
    print("=" * 80 + "\n")

