#!/usr/bin/env python3
"""
Centralized configuration management for ACE evaluation benchmark.

This module provides a single source of truth for all configuration,
loading credentials from environment variables and providing validation.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """
    Centralized configuration management.

    Loads all credentials and configuration from environment variables.
    Provides validation methods to ensure required credentials are present.
    """

    def __init__(self):
        """Initialize configuration by loading from environment."""
        self._load_credentials()

    def _load_credentials(self):
        """Load all credentials from environment variables."""
        # Supabase Database (required for pipeline operations)
        self.SUPABASE_URL: Optional[str] = os.getenv('SUPABASE_URL')
        self.SUPABASE_KEY: Optional[str] = os.getenv('SUPABASE_KEY')

        # Model API Keys (optional, depend on which models you're using)
        self.GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
        self.OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
        self.ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')

        # Service API Keys
        self.FIRECRAWL_API_KEY: Optional[str] = os.getenv('FIRECRAWL_API_KEY')
        self.SEARCHAPI_API_KEY: Optional[str] = os.getenv('SEARCHAPI_API_KEY')  # For YouTube transcripts

    def validate_supabase(self, required=True):
        """
        Validate that Supabase credentials are present.

        Args:
            required: If False, skip validation (for optional Supabase usage)

        Raises:
            ValueError: If required=True and SUPABASE_URL or SUPABASE_KEY is not set
        """
        if not required:
            return  # Skip validation when Supabase is optional

        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment.\n"
                "Create a .env file with your credentials or set environment variables.\n"
                "See .env.example for template."
            )

    def validate_model_key(self, provider: str):
        """
        Validate that a specific model provider key is present.

        Args:
            provider: Model provider name ('gemini', 'openai', 'anthropic')

        Raises:
            ValueError: If the required API key is not set
        """
        key_map = {
            'gemini': ('GEMINI_API_KEY', self.GEMINI_API_KEY),
            'openai': ('OPENAI_API_KEY', self.OPENAI_API_KEY),
            'anthropic': ('ANTHROPIC_API_KEY', self.ANTHROPIC_API_KEY),
        }

        if provider in key_map:
            key_name, key_value = key_map[provider]
            if not key_value:
                raise ValueError(
                    f"{key_name} must be set in environment to use {provider} models.\n"
                    f"Add {key_name}=your-api-key to your .env file."
                )

    def validate_firecrawl(self):
        """
        Validate that Firecrawl API key is present.

        Raises:
            ValueError: If FIRECRAWL_API_KEY is not set
        """
        if not self.FIRECRAWL_API_KEY:
            raise ValueError(
                "FIRECRAWL_API_KEY must be set in environment for web scraping.\n"
                "Add FIRECRAWL_API_KEY=your-api-key to your .env file."
            )

    def has_searchapi(self) -> bool:
        """
        Check if SearchAPI key is available (optional).

        Returns:
            bool: True if SEARCHAPI_API_KEY is set, False otherwise
        """
        return bool(self.SEARCHAPI_API_KEY)

    def has_supabase(self) -> bool:
        """
        Check if Supabase credentials are available (optional).

        Returns:
            bool: True if both SUPABASE_URL and SUPABASE_KEY are set, False otherwise
        """
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)


# Global config instance
# Import this in other modules: from configs.config import config
config = Config()


if __name__ == '__main__':
    """Test configuration loading."""
    print("Configuration Test")
    print("=" * 60)

    # Test credential loading
    print(f"SUPABASE_URL: {'✓ Set' if config.SUPABASE_URL else '✗ Not set'}")
    print(f"SUPABASE_KEY: {'✓ Set' if config.SUPABASE_KEY else '✗ Not set'}")
    print(f"GEMINI_API_KEY: {'✓ Set' if config.GEMINI_API_KEY else '✗ Not set'}")
    print(f"OPENAI_API_KEY: {'✓ Set' if config.OPENAI_API_KEY else '✗ Not set'}")
    print(f"ANTHROPIC_API_KEY: {'✓ Set' if config.ANTHROPIC_API_KEY else '✗ Not set'}")
    print(f"FIRECRAWL_API_KEY: {'✓ Set' if config.FIRECRAWL_API_KEY else '✗ Not set'}")
    print(f"SEARCHAPI_API_KEY: {'✓ Set' if config.SEARCHAPI_API_KEY else '✗ Not set'}")

    print("\n" + "=" * 60)

    # Test validation
    try:
        config.validate_supabase()
        print("✓ Supabase credentials validated")
    except ValueError as e:
        print(f"✗ Supabase validation failed: {e}")

