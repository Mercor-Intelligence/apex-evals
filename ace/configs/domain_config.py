#!/usr/bin/env python3
"""
Domain Configuration for Multi-Domain Pipeline Support

Defines domain-specific settings for Shopping, Food, Gaming, and DIY domains.
Supports all model providers (Gemini, OpenAI, Anthropic).
"""

from typing import Dict, List, Any
try:
    from model_providers import MODEL_REGISTRY
except ImportError:
    from .model_providers import MODEL_REGISTRY


# Base domain configurations (without table names - for model-based routing)
DOMAIN_BASE_CONFIG = {
    'Shopping': {
        'criterion_type_column': 'Criteria type',  # New unified column name
        'description_column': 'Description',
        'non_grounding_types': [
            'Meets quantity requirement',
            'Product is in a set list/recommends specific product',
            'Recommends buying habit',
            'Other'
        ],
        'exclude_types': ['Geographic availability'],
        'has_shop_vs_product': True
    },
    'Food': {
        'criterion_type_column': 'Criteria type',  # New unified column name
        'description_column': 'Description',
        'non_grounding_types': [
            'Meets quantity/duration requirement',
            'Meets dietary requirements',
            'Meets dish feature requirements',
            'Provides dietary information',
            'Meets preparation or cooking requirement',
            'Provides shopping/ingredient list',
            'Other',
            'Product is in a set list/recommends specific dish',
            'Meets serving/portion requirement',
            'Provides preparation instructions'
        ],
        'exclude_types': [],
        'has_shop_vs_product': False
    },
    'Gaming': {
        'criterion_type_column': 'Criteria type',  # New unified column name
        'description_column': 'Description',
        'non_grounding_types': [
            'Meets quantity requirement',
            'Game/strategy is in a set list/recommends specific game/strategy',
            'Provides instruction for using recommended strategy',
            'Other',
            'Provides game/strategy explanation'
        ],
        'exclude_types': [],
        'has_shop_vs_product': False
    },
    'DIY': {
        'criterion_type_column': 'Criteria type',  # New unified column name
        'description_column': 'Description',
        'non_grounding_types': [
            'Describes specific procedural steps',
            'Recommends consulting a professional',
            'Provides safety warnings',
            'Specifies necessary materials or tools',
            'Provides general DIY guidance and tips',
            'Provides step-by-step instructions'
        ],
        'exclude_types': [],
        'has_shop_vs_product': False
    }
}


def get_domain_config_for_model(domain: str, model_name: str) -> Dict[str, Any]:
    """
    Get domain configuration with dynamic table names based on model

    This is the new model-based configuration function that generates
    table names dynamically using the pattern: tasks_{domain}_{model}

    Args:
        domain: Domain name ('Shopping', 'Food', 'Gaming', or 'DIY')
        model_name: Model name (e.g. 'gpt-5', 'gemini-2.5-pro')

    Returns:
        dict: Domain configuration with keys:
            - task_table: Generated task table name
            - criteria_table: Generated criteria table name
            - criterion_type_column: Column name for criterion type
            - description_column: Column name for description
            - non_grounding_types: List of non-grounding criterion types
            - exclude_types: List of criterion types to exclude
            - has_shop_vs_product: Whether domain has shop vs product feature

    Raises:
        ValueError: If domain is not recognized

    Examples:
        >>> get_domain_config_for_model('Shopping', 'gpt-5')
        {'task_table': 'tasks_shopping_gpt-5', 'criteria_table': 'criteria_shopping_gpt-5', ...}

        >>> get_domain_config_for_model('Food', 'gemini-2.5-pro')
        {'task_table': 'tasks_food_gemini-2.5-pro', 'criteria_table': 'criteria_food_gemini-2.5-pro', ...}
    """
    if domain not in DOMAIN_BASE_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_BASE_CONFIG.keys())}")

    # Validate model name against registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    domain_lower = domain.lower()

    # Get base configuration (criterion types, columns, settings)
    config = DOMAIN_BASE_CONFIG[domain].copy()

    # Generate dynamic table names
    # Format: tasks_{domain}_{model} and criteria_{domain}_{model}
    # Keep hyphens in model names as-is (Supabase handles them)
    config['task_table'] = f"tasks_{domain_lower}_{model_name}"
    config['criteria_table'] = f"criteria_{domain_lower}_{model_name}"

    return config


def list_domains() -> List[str]:
    """List all available domains"""
    return list(DOMAIN_BASE_CONFIG.keys())


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MODEL-BASED CONFIGURATION TEST (New API)")
    print("="*60)

    test_models = [
        ('Shopping', 'gpt-5'),
        ('Shopping', 'gemini-2.5-pro'),
        ('Food', 'gemini-2.5-flash'),
        ('Gaming', 'sonnet-4.5'),
        ('DIY', 'o3-pro')
    ]

    for domain, model in test_models:
        config = get_domain_config_for_model(domain, model)
        print(f"\n{domain} + {model}:")
        print(f"  Task Table: {config['task_table']}")
        print(f"  Criteria Table: {config['criteria_table']}")
        print(f"  Has Shop vs. Product: {config['has_shop_vs_product']}")

    print("\n" + "="*60)

