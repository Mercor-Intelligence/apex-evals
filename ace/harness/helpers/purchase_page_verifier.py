#!/usr/bin/env python3
"""
Purchase Page Verifier
Validates that URLs are direct product purchasing pages
"""

import os
import sys
from firecrawl import FirecrawlApp

# Add project root to path FIRST (2 levels up from helpers/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import config

# New Gemini SDK
from google import genai
from google.genai import types

# Verifier Configuration Constants
VERIFIER_MODEL = 'gemini-2.5-pro'  # Use Pro for consistent autograding
VERIFIER_TEMPERATURE = 0.0  # Deterministic classification

# Initialize Gemini client (fail-fast if API key not set)
config.validate_model_key('gemini')
_client = genai.Client(api_key=config.GEMINI_API_KEY)
_gen_config = types.GenerateContentConfig(temperature=VERIFIER_TEMPERATURE)


def _generate(prompt: str) -> str:
    """Generate content using Gemini"""
    result = _client.models.generate_content(
        model=VERIFIER_MODEL,
        contents=prompt,
        config=_gen_config
    )
    return result.text.strip()


def verify_purchase_link(product_name, url, firecrawl_app, verification_mode='Product'):
    """
    Verify if URL is a valid product purchasing page or vendor page

    Modes:
        - 'Product': Strict - must be direct product page with purchase button
        - 'Shop': Relaxed - can be vendor/store page where user can browse

    Returns:
        dict: {'is_valid': bool, 'reason': str, 'page_type': str}
    """
    try:
        # Scrape the URL
        result = firecrawl_app.scrape(
            url,
            formats=['markdown'],
            proxy='auto',
            only_main_content=False
        )

        markdown = result.markdown if hasattr(result, 'markdown') else ''

        if not markdown or len(markdown) < 100:
            return {
                'is_valid': False,
                'reason': 'Page failed to load or has no content',
                'page_type': 'error'
            }

        # Different validation criteria based on mode
        if verification_mode == 'Shop':
            validation_prompt = f"""VALID store/vendor page:
- Direct link to the vendor's store or product category page
- Shows products from {product_name} (the vendor/brand)
- User can browse and potentially purchase from this vendor
- Includes vendor branding, product listings, or shop page

INVALID pages:
- Unrelated vendor or store
- Generic marketplace search results (unless {product_name} is the seller)
- Review/comparison pages without vendor link
- Dead links or error pages"""
        else:  # Product mode (default)
            validation_prompt = f"""VALID product purchasing page:
- Direct product page for {product_name}
- Has "Add to Cart", "Buy Now", or similar purchase button
- User can directly purchase THIS specific product

INVALID pages:
- Product list or search results (even if they link to purchase pages)
- Review/comparison pages (without direct purchase)
- General shop homepage
- Category pages with multiple products"""

        # Use Gemini to classify page
        prompt = f"""You are classifying a webpage to determine if it's a valid {'store/vendor' if verification_mode == 'Shop' else 'product purchasing'} page.

Product/Vendor name: {product_name}

Page content:
{markdown}

Classify this page:

{validation_prompt}

Return ONLY valid JSON:
{{
  "is_purchase_page": true/false,
  "page_type": "product_page|vendor_page|list|search_results|review|general",
  "reason": "Brief explanation why valid or invalid"
}}"""

        text = _generate(prompt)

        # Parse JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        import json
        result_data = json.loads(text)

        return {
            'is_valid': result_data.get('is_purchase_page', False),
            'reason': result_data.get('reason', ''),
            'page_type': result_data.get('page_type', 'unknown')
        }

    except Exception as e:
        return {
            'is_valid': False,
            'reason': f'Error verifying link: {str(e)}',
            'page_type': 'error'
        }


def verify_gaming_link(recommendation_name, url, firecrawl_app, criterion_description):
    """
    Verify if URL meets the Gaming criterion's link requirement

    Args:
        recommendation_name: Name of game/strategy/approach
        url: URL to verify
        firecrawl_app: Firecrawl instance
        criterion_description: Full criterion description (e.g., "Provides official Nintendo links for all games.")

    Returns:
        dict: {'is_valid': bool, 'reason': str, 'page_type': str}
    """
    try:
        # Scrape the URL
        result = firecrawl_app.scrape(
            url,
            formats=['markdown'],
            proxy='auto',
            only_main_content=False
        )

        markdown = result.markdown if hasattr(result, 'markdown') else ''

        if not markdown or len(markdown) < 10:
            return {
                'is_valid': False,
                'reason': 'Page failed to load or has no content',
                'page_type': 'error'
            }

        # Ask LLM to verify against the criterion description directly
        prompt = f"""You are verifying if a webpage meets a link requirement for a Gaming criterion.

Game/Item: {recommendation_name}
URL: {url}
Criterion: {criterion_description}

Page content:
{markdown}

Based on the criterion description, does this webpage meet the link requirement?

EXAMPLES:
- If criterion says "official Nintendo links" → Check if URL is from nintendo.com or official Nintendo domain and content confirms it's official and to the actual product, not a general page.
- If criterion says "Steam store links" → Check if URL is from steampowered.com or store.steampowered.com
- If criterion says "BoardGameGeek entries" → Check if URL is from boardgamegeek.com
- If criterion says "official links" → Check if it's from the official/authoritative source for that game (not third-party) and to the actual product, not a general page.
- If criterion says "purchase links" → Check if it's a page where the game can be purchased
- If criterion says "gameplay video links" → Check if it's a video page showing gameplay

Verify both:
1. URL domain matches the requirement
2. Page content is relevant to the game/item

And then only return true if it passes both of those checks.

Return ONLY valid JSON:
{{
  "is_valid": true/false,
  "page_type": "official|store|forum|video|wiki|third_party|other",
  "reason": "Brief explanation why it does or doesn't meet the criterion requirement. Mention the domain and content type."
}}"""

        text = _generate(prompt)

        # Parse JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        import json
        result_data = json.loads(text)

        return {
            'is_valid': result_data.get('is_valid', False),
            'reason': result_data.get('reason', ''),
            'page_type': result_data.get('page_type', 'unknown')
        }

    except Exception as e:
        return {
            'is_valid': False,
            'reason': f'Error verifying link: {str(e)}',
            'page_type': 'error'
        }


if __name__ == '__main__':
    # Test
    import sys

    if len(sys.argv) < 3:
        print("Usage: python purchase_page_verifier.py <product-name> <url>")
        print("       python purchase_page_verifier.py <product-name> <url> --gaming 'criterion description'")
        sys.exit(1)

    product = sys.argv[1]
    url = sys.argv[2]

    config.validate_firecrawl()
    firecrawl = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

    # Check if Gaming mode
    if '--gaming' in sys.argv:
        criterion_desc = sys.argv[sys.argv.index('--gaming') + 1]
        result = verify_gaming_link(product, url, firecrawl, criterion_desc)
        print(f"\nGame: {product}")
        print(f"URL: {url}")
        print(f"Criterion: {criterion_desc}")
    else:
        result = verify_purchase_link(product, url, firecrawl)
        print(f"\nProduct: {product}")
        print(f"URL: {url}")

    print(f"\nResult: {'VALID' if result['is_valid'] else 'INVALID'}")
    print(f"Page type: {result['page_type']}")
    print(f"Reason: {result['reason']}")

