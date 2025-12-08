#!/usr/bin/env python3
"""
Model Provider Abstraction Layer

Handles different AI provider APIs (Gemini, OpenAI, Claude) and normalizes
their responses into a unified format for downstream processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from google import genai
from google.genai import types
import json
import os

from configs.logging_config import setup_logging
logger = setup_logging(__name__)
try:
    import anthropic
except ImportError:
    pass
try:
    from openai import OpenAI
except ImportError:
    pass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Import config for credential management (same directory)
try:
    from config import config
except ImportError:
    # When imported as package
    from .config import config


# API Configuration Constants
# These match current provider defaults - DO NOT CHANGE without testing

# Gemini provider settings

# Gemini model-specific settings
GEMINI_MODEL_CONFIGS = {
    'gemini-2.5-flash': {
        'temperature': 0.7,
        'thinking_budget': 24576,     # 24k thinking tokens
        'use_thinking_level': False,
    },
    'gemini-2.5-pro': {
        'temperature': 0.7,
        'thinking_budget': 32768,     # 32k thinking tokens
        'use_thinking_level': False,
    },
    'gemini-3-pro': {
        'temperature': 1.0,           # Gemini 3 Pro uses temperature 1.0
        'thinking_level': 'high',     # Use thinking_level instead of thinking_budget
        'use_thinking_level': True,   # Gemini 3 uses thinking_level API
    },
}

# OpenAI provider settings
OPENAI_REASONING_EFFORT = "high"      # Reasoning effort level

# Anthropic provider settings (model-specific)
ANTHROPIC_MODEL_CONFIGS = {
    'sonnet-4.5': {
        'max_tokens': 64000,      # Sonnet 4.5 limit
        'budget_tokens': 63999,   # max thinking budget
    },
    'opus-4.1': {
        'max_tokens': 32000,      # Opus 4.1 limit
        'budget_tokens': 31999,   # max thinking budget
    },
    'opus-4.5': {
        'max_tokens': 64000,      # Opus 4.5 limit (same as sonnet-4.5)
        'budget_tokens': 63999,   # max thinking budget (same as sonnet-4.5)
    },
}


# Model-to-Provider Registry
MODEL_REGISTRY = {
    # OpenAI models
    'gpt-5': 'openai',
    'gpt-5.1': 'openai',
    'o3': 'openai',
    'o3-pro': 'openai',

    # Gemini models
    'gemini-2.5-pro': 'gemini',
    'gemini-2.5-flash': 'gemini',
    'gemini-3-pro': 'gemini',

    # Anthropic Claude models
    'sonnet-4.5': 'anthropic',
    'opus-4.1': 'anthropic',
    'opus-4.5': 'anthropic',
}

# Note: OpenAI Responses API uses base model names with tools=[{"type": "web_search"}]
# All models support web search/grounding through the Responses API


def get_provider_for_model(model_name: str) -> str:
    """
    Get the provider name for a given model

    Args:
        model_name: Model name (e.g. 'gpt-5', 'gemini-2.5-pro')

    Returns:
        str: Provider name ('openai', 'gemini', 'anthropic')

    Raises:
        ValueError: If model is not recognized
    """
    provider = MODEL_REGISTRY.get(model_name)
    if not provider:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return provider


def get_provider_instance(model_name: str) -> 'BaseProvider':
    """
    Get a provider instance for a given model

    Args:
        model_name: Model name (e.g. 'gpt-5', 'gemini-2.5-pro')

    Returns:
        BaseProvider instance
    """
    provider_name = get_provider_for_model(model_name)

    if provider_name == 'openai':
        return OpenAIProvider()
    elif provider_name == 'gemini':
        return GeminiProvider()
    elif provider_name == 'anthropic':
        return ClaudeProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


class BaseProvider(ABC):
    """Abstract base class for AI model providers"""

    @abstractmethod
    def make_api_call(self, prompt: str, model_name: str) -> dict:
        """
        Make grounded API call to the provider

        Args:
            prompt: User prompt/query
            model_name: Specific model to use

        Returns:
            dict: Raw API response
        """
        pass

    @abstractmethod
    def parse_response(self, raw_response: dict, model_name: str) -> Dict[str, Any]:
        """
        Parse provider-specific response into normalized format

        Args:
            raw_response: Raw API response
            model_name: Model that generated the response

        Returns:
            dict: Normalized format with keys:
                - groundingChunks: List of source URLs/chunks
                - groundingSupports: List of citations with indices
                - responseText: Generated text
                - direct_grounding: Raw provider response (for storage)
        """
        pass


class GeminiProvider(BaseProvider):
    """Google Gemini API provider"""

    # Map internal model names to Gemini API model names
    MODEL_NAME_MAP = {
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-3-pro': 'gemini-3-pro-preview',  # Preview version required
    }

    def __init__(self):
        # Load Gemini API key from config
        config.validate_model_key('gemini')
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    def make_api_call(self, prompt, model_name):
        """Make Gemini grounded API call with model-specific thinking config"""
        # Set up grounding tool
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        # Map internal model name to API model name
        api_model_name = self.MODEL_NAME_MAP.get(model_name, model_name)

        # Get model-specific configuration
        if model_name not in GEMINI_MODEL_CONFIGS:
            raise ValueError(f"Unknown Gemini model: {model_name}. Supported models: {list(GEMINI_MODEL_CONFIGS.keys())}")
        model_config = GEMINI_MODEL_CONFIGS[model_name]
        temperature = model_config['temperature']

        # Configure thinking based on model
        # Gemini 3 Pro uses thinking_level, others use thinking_budget
        if model_config.get('use_thinking_level'):
            # Gemini 3 Pro: Use thinking_level parameter
            thinking_config = types.ThinkingConfig(
                thinking_level=model_config['thinking_level']
            )
        else:
            # Gemini 2.5 models: Use thinking_budget parameter
            thinking_config = types.ThinkingConfig(
                thinking_budget=model_config['thinking_budget']
            )

        # Build generation config
        generation_config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=temperature,
            thinking_config=thinking_config
        )

        # Make the call
        response = self.client.models.generate_content(
            model=api_model_name,  # Use mapped API model name
            contents=prompt,
            config=generation_config
        )

        return response

    def parse_response(self, raw_response, model_name):
        """Parse Gemini response to normalized format"""
        # Extract response text
        response_text = raw_response.text if hasattr(raw_response, 'text') else ''

        # Check for grounding metadata
        if not (hasattr(raw_response, 'candidates') and raw_response.candidates):
            logger.warning("No candidates in Gemini response")
            return {
                'groundingChunks': [],
                'groundingSupports': [],
                'responseText': response_text,
                'webSearchQueries': [],
                'direct_grounding': None
            }

        candidate = raw_response.candidates[0]
        if not (hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata):
            logger.warning("No grounding metadata in Gemini response")
            return {
                'groundingChunks': [],
                'groundingSupports': [],
                'responseText': response_text,
                'webSearchQueries': [],
                'direct_grounding': None
            }

        gm = candidate.grounding_metadata

        # Build normalized format
        grounding_json = {
            'responseText': response_text,
            'groundingChunks': [],
            'groundingSupports': [],
            'webSearchQueries': []
        }

        # Extract web search queries (handle None safely)
        if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
            try:
                grounding_json['webSearchQueries'] = list(gm.web_search_queries)
            except (TypeError, AttributeError):
                grounding_json['webSearchQueries'] = []

        # Extract chunks
        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                grounding_json['groundingChunks'].append({
                    'web': {
                        'title': chunk.web.title if hasattr(chunk, 'web') else '',
                        'uri': chunk.web.uri if hasattr(chunk, 'web') else ''
                    }
                })

        # Extract supports (citations)
        if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
            for support in gm.grounding_supports:
                grounding_json['groundingSupports'].append({
                    'groundingChunkIndices': list(support.grounding_chunk_indices) if hasattr(support, 'grounding_chunk_indices') else [],
                    'segment': {
                        'startIndex': support.segment.start_index if hasattr(support, 'segment') else 0,
                        'endIndex': support.segment.end_index if hasattr(support, 'segment') else 0,
                        'text': support.segment.text if hasattr(support, 'segment') else ''
                    }
                })

        # Store raw Gemini grounding metadata (for backward compatibility and storage)
        grounding_json['direct_grounding'] = {
            'groundingChunks': grounding_json['groundingChunks'],
            'groundingSupports': grounding_json['groundingSupports'],
            'webSearchQueries': grounding_json['webSearchQueries']
        }

        return grounding_json


class OpenAIProvider(BaseProvider):
    """OpenAI API provider"""

    def __init__(self):
        config.validate_model_key('openai')
        self.api_key = config.OPENAI_API_KEY

    def make_api_call(self, prompt, model_name):
        """Make OpenAI API call with web search/grounding enabled using Responses API"""
        if 'OpenAI' not in globals():
            raise ImportError(
                "OpenAI package not installed.\n"
                "Install with: uv add openai  OR  pip install openai"
            )

        # Add timeout to prevent infinite hangs (25 minutes for model reasoning)
        from httpx import Timeout
        client = OpenAI(
            api_key=self.api_key,
            timeout=Timeout(1500.0, connect=10.0)  # 25 minutes total, 10s connect
        )

        print(f"   Using OpenAI Responses API with model: {model_name}")
        print(f"   Web search enabled: Yes")
        print(f"   Timeout: 1500s (25 minutes)")

        # Use Responses API with web_search tool and reasoning effort
        response = client.responses.create(
            model=model_name,  # Use base model name (e.g., gpt-5, gpt-5.1, o3, o3-pro)
            input=prompt,
            reasoning={"effort": OPENAI_REASONING_EFFORT},
            tools=[{"type": "web_search"}]  # Enable web search/grounding
        )

        # Convert to dict for easier processing
        return response.model_dump()

    def parse_response(self, raw_response, model_name):
        """Parse OpenAI Responses API response to normalized format"""
        # Responses API format: output array with message type
        # Structure: output[].content[].text and annotations

        content = ''
        annotations = []

        # Find the message output
        for output_item in raw_response.get('output', []):
            if output_item.get('type') == 'message':
                # Found the message, now find output_text in content
                for content_item in output_item.get('content', []):
                    if content_item.get('type') == 'output_text':
                        content = content_item.get('text', '')
                        annotations = content_item.get('annotations', [])
                        break
                break

        # Step 1: Extract unique URLs → groundingChunks
        # Responses API format: annotations directly contain url, title, start_index, end_index
        url_to_index = {}
        chunks = []

        for ann in annotations:
            if ann.get('type') != 'url_citation':
                continue

            # Responses API: fields are at top level of annotation
            url = ann.get('url', '')
            title = ann.get('title', url)

            if url and url not in url_to_index:
                url_to_index[url] = len(chunks)
                chunks.append({
                    'web': {
                        'uri': url,
                        'title': title
                    }
                })

        # Step 2: Create groundingSupports from annotations
        supports = []

        for ann in annotations:
            if ann.get('type') != 'url_citation':
                continue

            # Responses API: fields are at top level
            start = ann.get('start_index', 0)
            end = ann.get('end_index', 0)
            url = ann.get('url', '')

            if not url or url not in url_to_index:
                continue

            # Extract cited text using character indices
            cited_text = content[start:end] if start < len(content) and end <= len(content) else ''
            chunk_idx = url_to_index[url]

            supports.append({
                'groundingChunkIndices': [chunk_idx],
                'segment': {
                    'startIndex': start,
                    'endIndex': end,
                    'text': cited_text
                }
            })

        # Build normalized format
        normalized = {
            'groundingChunks': chunks,
            'groundingSupports': supports,
            'responseText': content,
            'webSearchQueries': [],  # OpenAI doesn't expose search queries
            'direct_grounding': raw_response  # Store raw OpenAI response
        }

        return normalized


class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider"""

    # Map internal model names to Anthropic API model names
    MODEL_NAME_MAP = {
        'sonnet-4.5': 'claude-sonnet-4-5',
        'opus-4.1': 'claude-opus-4-1',
        'opus-4.5': 'claude-opus-4-5',
    }

    def __init__(self):
        config.validate_model_key('anthropic')
        self.api_key = config.ANTHROPIC_API_KEY

    def make_api_call(self, prompt, model_name):
        """
        Make Claude API call with web search enabled.
        Uses server-side web_search_20250305 tool which auto-executes searches.
        """
        if 'anthropic' not in globals():
            raise ImportError(
                "Anthropic package not installed.\n"
                "Install with: uv add anthropic  OR  pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        # Map internal model name to API model name
        api_model_name = self.MODEL_NAME_MAP.get(model_name, model_name)

        # Get model-specific configuration (fallback to default if model not found)
        model_config = ANTHROPIC_MODEL_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown Anthropic model: {model_name}. Supported models: {list(ANTHROPIC_MODEL_CONFIGS.keys())}")

        print(f"   Using Anthropic model: {api_model_name}")
        print(f"   Max tokens: {model_config['max_tokens']} (thinking budget: {model_config['budget_tokens']})")
        print(f"   Web search enabled: Yes (server-side tool)")

        response = client.messages.create(
            model=api_model_name,
            max_tokens=model_config['max_tokens'],
            temperature=1.0, # Required Temp when thinking is enabled
            thinking={
                "type": "enabled",
                "budget_tokens": model_config['budget_tokens']
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search"
            }],
            timeout=1500.0  # 25 minutes for extended thinking + web search
        )

        # Note: stop_reason will be "pause_turn" because web_search is server-executed
        # The response already contains web_search_tool_result blocks with search results

        # Convert to dict
        return response.model_dump()

    def parse_response(self, raw_response, model_name):
        """
        Parse Claude response to normalized format.
        Only extracts cited sources (from citations) and web search queries.
        Does NOT include uncited search results from web_search_tool_result blocks.
        """
        content_list = raw_response.get('content', [])

        response_text = ""
        chunks = []
        url_to_index = {}
        supports = []
        web_search_queries = []

        # Iterate through content blocks to extract data
        for block in content_list:
            block_type = block.get('type')

            # 1. Extract Web Search Queries
            if block_type == 'server_tool_use' and block.get('name') == 'web_search':
                query = block.get('input', {}).get('query')
                if query:
                    web_search_queries.append(query)

            # 2. SKIP web_search_tool_result blocks - we only care about cited sources
            # (Uncited search results are not useful for our pipeline)

            # 3. Extract Text and Citations (ONLY cited sources)
            elif block_type == 'text':
                text_content = block.get('text', '')

                # Calculate start/end indices for this block relative to full response
                start_index = len(response_text)
                response_text += text_content
                end_index = len(response_text)

                # Process citations attached to this text block
                citations = block.get('citations', [])
                if citations:
                    for citation in citations:
                        url = citation.get('url')
                        # Check if this citation type has URL
                        # example: type: "web_search_result_location"

                        if url:
                            # Ensure URL is in chunks (add if missing)
                            if url not in url_to_index:
                                url_to_index[url] = len(chunks)
                                chunks.append({
                                    'web': {
                                        'uri': url,
                                        'title': citation.get('title', 'Untitled')
                                    }
                                })

                            chunk_idx = url_to_index[url]

                            # Create support entry
                            # Note: The citation applies to this text block.
                            # We use the full block text as the segment.
                            supports.append({
                                'groundingChunkIndices': [chunk_idx],
                                'segment': {
                                    'startIndex': start_index,
                                    'endIndex': end_index,
                                    'text': text_content
                                }
                            })

        normalized = {
            'responseText': response_text,
            'groundingChunks': chunks,
            'groundingSupports': supports,
            'webSearchQueries': web_search_queries,
            'direct_grounding': raw_response
        }

        # Simple summary output (consistent with other providers)
        print(f"   ✓ Parsed {len(chunks)} sources, {len(supports)} citations from {len(web_search_queries)} queries")

        return normalized


if __name__ == '__main__':
    # Test model registry
    print("="*60)
    print("MODEL PROVIDER REGISTRY TEST")
    print("="*60)

    for model_name, provider_name in MODEL_REGISTRY.items():
        print(f"{model_name:25} → {provider_name}")

    print("\n" + "="*60)
    print("PROVIDER LOOKUP TEST")
    print("="*60)

    test_models = ['gpt-5', 'gemini-2.5-pro', 'sonnet-4.5']
    for model in test_models:
        try:
            provider = get_provider_for_model(model)
            instance = get_provider_instance(model)
            print(f"✅ {model:25} → {provider:15} ({instance.__class__.__name__})")
        except Exception as e:
            print(f"❌ {model:25} → Error: {e}")
