#!/usr/bin/env python3
"""
Autograder for LLM Product Recommendations
Evaluates recommendations against criteria using two-stage verification
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from firecrawl import FirecrawlApp
from typing import Dict, List, Optional, Any

# Add project root to path FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from configs.logging_config import setup_logging
from configs.config import config
from helpers.purchase_page_verifier import verify_purchase_link, verify_gaming_link

logger = setup_logging(__name__)

# New Gemini SDK imports
from google import genai
from google.genai import types

# Autograder Configuration Constants
AUTOGRADER_MODEL = 'gemini-2.5-pro'  # Gemini Pro for all autograding
AUTOGRADER_TEMPERATURE = 0.0  # Deterministic evaluation


class Autograder:
    """Automated grading system for LLM recommendations"""

    def __init__(self):
        # Initialize Gemini client with new SDK
        config.validate_model_key('gemini')
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.gen_config = types.GenerateContentConfig(temperature=AUTOGRADER_TEMPERATURE)
        self.logs = []
        config.validate_firecrawl()
        self.firecrawl = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

    def _generate(self, prompt: str) -> str:
        """Generate content using Gemini - single place for all LLM calls"""
        result = self.client.models.generate_content(
            model=AUTOGRADER_MODEL,
            contents=prompt,
            config=self.gen_config
        )
        return result.text.strip()

    def log(self, message, level='info'):
        """Add log entry"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.logs.append(entry)

        if level == 'info':
            logger.info(f"  {message}")
        elif level == 'success':
            logger.info(f"  {message}")
        elif level == 'warning':
            logger.warning(f"  {message}")
        elif level == 'error':
            logger.error(f"  {message}")
        else:
            logger.info(f"  {message}")

    def check_response_text(self, criterion, response_text, products):
        """
        Stage 1: Check if products meet criterion based on response text

        Returns:
            dict: {
                'all_pass': bool,
                'reasoning': str,
                'product_results': [{'product': str, 'pass': bool, 'reason': str}]
            }
        """
        self.log(f"Stage 1: Checking response text for criterion: {criterion['description']}")

        recommendation_names = [p['product_name'] for p in products]

        prompt = f"""You are evaluating a response against a specific criterion.

        Criterion to evaluate: {criterion['description']}

        Response Text:
        {response_text}

        **CANONICAL RECOMMENDATION LIST (use EXACTLY these) if applicable:**
        {', '.join(recommendation_names) if recommendation_names else '(No recommendations identified)'}

        CRITICAL EVALUATION RULES:
        1. **ZERO BACKGROUND KNOWLEDGE**: You must IGNORE any knowledge you have about these recommendations/products and only grade the response text based on the criterion. Pretend you have NEVER heard of them before.
        2. **ONLY STATEMENTS ABOUT THE Recommendations/Products/Services/etc. COUNT**: A criterion is met for a recommendation/product ONLY if the response text DIRECTLY states the required information for that specific recommendation/product or if it is obviously inferrable.
        3. **USE THE CANONICAL LIST**: You MUST evaluate EXACTLY the recommendations listed above if applicable.
           - Do NOT add recommendations that aren't in the list
           - Do NOT remove recommendations from the list
           - Do NOT rename or modify the recommendation names
           - If the list is empty, the response provided NO recommendations
           
           **WHAT COUNTS AS A RECOMMENDATION:**
           - A recommendation is a specific item the response suggests the user should consider/buy/use
           - Recommendations mentioned only as "examples of what NOT to buy" or "for comparison" are NOT recommendations
           - Recommendations mentioned only to explain why they don't work are NOT recommendations  
           - A "meal plan" or "weekly plan" is ONE recommendation, not multiple separate meals
           - A "recipe" is ONE recommendation, not individual ingredients
           - A "DIY project" step by step guide is ONE recommendation, not individual steps or materials as different recommendations.
           
           If you believe the canonical list is WRONG (e.g., includes non-recommendations or misses actual recommendations),
           still evaluate the canonical list as given, but note your concern in the reasoning.

        4. **ZERO BACKGROUND KNOWLEDGE**: You must COMPLETELY IGNORE any knowledge you have about these recommendations/products. Pretend you have NEVER heard of them before.

        5. **ONLY EXPLICIT STATEMENTS COUNT**: A criterion is met for a recommendation ONLY if the response text DIRECTLY states the required information for that specific recommendation or if it is obviously inferrable.
        Example:
        - If the criterion says "supports 2+ players" → The response MUST explicitly say "multiplayer", "2 players", "co-op", "local multiplayer" or similar
        - If the criterion says "compatible with X" → The response MUST explicitly mention compatibility with X
        - If something is IMPLIED, HINTED AT, or requires INFERENCE → It does NOT count

        When checking recommendation features, it might meet additional features. This is fine. For example:
        Criterion: "Recommends only mixed martial arts (MMA) gloves"
        Recommendation: "Gloves for MMA and Boxing"
        Evaluation: meets_criterion = TRUE -- because the recommended gloves are advertised for MMA. Multi-purpose is OK.

        6. **STRICT PER-PRODUCT CHECKING**: For each recommendation, look at ONLY the text that describes that specific ecommendation/product. Do NOT use:
        - Context from other recommendations
        - General statements about the overall response
        - Your knowledge about what these items typically are
        - Assumptions based on names

        7. **DETERMINE QUANTIFIER**:
        - "at least one RECOMMENDATION/PRODUCT" or "one RECOMMENDATION/PRODUCT" → required_pass_count: 1, evaluation_type: per_product_any
        - "at least 2 RECOMMENDATION/PRODUCTS" or "at least 3 RECOMMENDATION/PRODUCTS" → required_pass_count: 2 or 3, evaluation_type: per_product_any
        - "ALL" or "only" or "each" → required_pass_count: -1, evaluation_type: per_product_all
        - "Recommends [TYPE]" (e.g., "Recommends N64 games", "Recommends red recommendations/products", "Recommends shoes with...") → required_pass_count: -1, evaluation_type: per_product_all (implicit ALL)
        - "Recommends a [FEATURE]" or "Recommends a [TYPE] with [FEATURE]" (e.g., "Recommends a product under $50", "Recommends a laptop with 16GB RAM") → required_pass_count: -1, evaluation_type: per_product_all (implicit ALL - every recommendation must have that feature)
        - "Provides X items" or "recommends Y recommendations/products" (quantity check) → evaluation_type: holistic

        **IMPORTANT:** "one or more of the following FEATURES" refers to features PER RECOMMENDATION/PRODUCT, not number of recommendations/products. This is still per_product_all.
        
        **IMPORTANT:** "Recommends a [product/feature description]" means ALL recommendations must match that description. The "a" is singular grammar, NOT a quantifier. For example, "Recommends a budget laptop" means ALL recommended laptops must be budget laptops.

        **DEFAULT:** If NO explicit quantifier like "at least" or "one of" is present, assume ALL must pass → required_pass_count: -1

        SPECIAL INSTRUCTION FOR CONDITIONAL CRITERIA:
        If this criterion starts with "If" (conditional), treat "available in X", "comes in X", "offered in X" as EQUIVALENT to "recommends X". Stock caveats like "confirm availability" or "check stock" do NOT make it fail.

        SCORING:
        - Criteria with "at least N": Pass if N or more recommendations/products meet it (e.g., if "at least 2" then Pass if 2+ recommendations/products meet)
        - Criteria with "only" or "all": Pass only if EVERY recommendation/product meets it
        - Criteria about quantity/count: Evaluate the overall response, not individual products

        **CRITICAL - NO VACUOUS TRUTH**:
        If the response contains NO recommendations/products, or failse to provide a recommendaiton then:
        - Criteria that CHECK features of recommendations (e.g., "Recommends only X with feature Y") → FAIL (pass=false)
        - Criteria that REQUIRE providing something (e.g., "Provides links", "States price") → FAIL (pass=false)
        - The user asked for recommendations - providing NONE means criteria cannot be met
        - Do NOT pass criteria vacuously just because "there are no violations if there are no products"
        - Exception: Criteria explicitly about quantity (e.g., "Recommends at least one") are evaluated separately as holistic
        - BE REASONABLE AND LOGICAL. If it should pass, pass, if it should fail, fail. Be Reasonable and logical.
        EXAMPLE OF STRICT EVALUATION:
        Criterion: "Recommends only games that support 2 or more players locally"
        Recommendation/Product: "Mario Kart - A fun racing game where you compete for first place"
        Evaluation: meets_criterion = FALSE (no mention of multiplayer, 2+ players, or local play)

        Recommendation/Product: "Super Smash Bros - Battle with up to 4 players locally"
        Evaluation: meets_criterion = TRUE (explicitly states "up to 4 players locally")

        KEY EXAMPLES:

        Feature Lists: "Recommends shoes with one or more of: non-compressible sole, raised heel, straps"
        - Evaluation: per_product_all (ALL shoes must have at least ONE of the listed features)
        - "Shoe 1: firm sole" → TRUE (has non-compressible sole)
        - "Shoe 2: elevated heel" → TRUE (has raised heel)
        - "Shoe 3: flat, soft sole" → FALSE (has none of the features)
        - Result: FAIL (not all shoes have at least one feature)

        Conditional: "If recommending item A, must have feature B"
        - "Item A - available in feature B" → TRUE
        - "Item A - while available in feature B, check stock" → TRUE (caveat ignored)
        - "Item A" (no mention of B) → FALSE
        - "Item C" (not item A) → TRUE (condition not met)

        Only: "Recommends only MMA gloves"
        - "MMA/Boxing gloves" → TRUE (is MMA, multi-purpose OK)
        - "Boxing gloves" (no MMA) → FALSE

        Strict: "Games with 4+ players"
        - "Fun party game" → FALSE (no player count mentioned)
        
        Return ONLY valid JSON:
        {{
        "pass": true/false,
        "reasoning": "Clear explanation. For each recommendation, state whether the criterion is EXPLICITLY stated in its description.",
        "evaluation_type": "holistic|per_product_all|per_product_any",
        "required_pass_count": <number or -1>,
        "recommendations_checked": [{{"recommendation_name": "EXACT name from canonical list", "meets_criterion": true/false}}],
        "violation_found": true/false
        }}
        
        **IMPORTANT**: The "recommendations_checked" array MUST contain EXACTLY the items from the CANONICAL RECOMMENDATION LIST above.
        Use the EXACT names provided. If the canonical list has 3 items, your array must have 3 items with those exact names.

        SET required_pass_count based on the criterion wording:
        - -1 = ALL recommendation/products must pass (for "only", "all", "each", OR when no specific number given)
        - 1 = At least 1 recommendation/product must pass (for "one of", "at least one")
        - 2 = At least 2 recommendation/products must pass (for "at least 2")
        - 3 = At least 3 recommendation/products must pass (for "at least 3")
        - etc.

        Examples:
        - "Recommends only red items" → required_pass_count: -1 (ALL)
        - "At least 2 games under $50" → required_pass_count: 2
        - "One of the options includes X" → required_pass_count: 1"""

        try:
            response = self._generate(prompt)

            # Parse JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()

            data = json.loads(response)

            # Log results
            eval_type = data.get('evaluation_type', 'unknown')
            violation = data.get('violation_found', False)
            all_pass = data.get('pass', False)
            # Support both old and new key names
            products_checked = data.get('recommendations_checked', data.get('recommendation/products_checked', []))
            required_count = data.get('required_pass_count', -1)  # NEW: Extract required pass count

            status = "✓" if all_pass else "✗"
            # Print directly without emoji prefix since status symbol is already included
            print(f"    {status} Evaluation ({eval_type}): {data.get('reasoning', '')[:100]}...")

            # Log individual product checks if available
            if products_checked and len(products_checked) > 0:
                for product_check in products_checked:
                    recommendation_name = product_check.get('recommendation_name', 'Unknown')
                    meets = product_check.get('meets_criterion', False)
                    symbol = "✓" if meets else "✗"
                    # Print directly without emoji prefix since status symbol is already included
                    print(f"      {symbol} {recommendation_name}: {'Meets criterion' if meets else 'Does not meet criterion'}")

            if violation:
                self.log(f"  Violation detected - criterion fails", 'warning')

            return {
                'all_pass': all_pass,
                'reasoning': data.get('reasoning', ''),
                'evaluation_type': eval_type,
                'violation_found': violation,
                'recommendation/products_checked': products_checked,
                'required_pass_count': required_count  # NEW: Include in return
            }

        except Exception as e:
            # If it's a rate limit error, re-raise to fail the entire task for retry
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                self.log(f"Rate limit hit - failing task for retry: {e}", 'error')
                raise  # Re-raise to fail the task
            
            self.log(f"Error in response text check: {e}", 'error')
            return {
                'all_pass': False,
                'reasoning': f"Error during check: {str(e)}",
                'evaluation_type': 'error',
                'violation_found': False
            }

    def check_grounded_sources(self, criterion, product_map, all_sources, evaluation_type='per_product_all', required_count=-1):
        """
        Stage 2: Check if products meet criterion based on grounded sources
        A product PASSES if criterion is true in AT LEAST ONE of its sources

        For Gaming/other domains with no products: Does holistic grounding check

        Args:
            criterion: The criterion being evaluated
            product_map: List of products/recommendations with source mappings
            all_sources: All grounding sources
            evaluation_type: How to evaluate (per_product_all, per_product_any, holistic)
            required_count: How many must pass (-1 = all, 1 = at least one, 2 = at least two, etc.)

        Returns:
            dict: {
                'all_pass': bool,
                'reasoning': str,
                'product_results': [{'product': str, 'pass': bool, 'reason': str}]
            }
        """
        self.log(f"Stage 2: Checking grounded sources for criterion: {criterion['description']}")

        product_results = []
        all_pass = True

        # CASE 1: No products (Gaming domain - approaches/strategies without specific product names)
        if not product_map or len(product_map) == 0:
            self.log(f"  No product map - performing holistic grounding check")

            # Get all source content
            source_contents = []
            for source in all_sources:
                if source.get('webpage_content', {}).get('markdown'):
                    source_contents.append({
                        'title': source['source_title'],
                        # 'content': source['webpage_content']['markdown'][:30000]  # Limit size per source
                        'content': source['webpage_content']['markdown']   # No limit size
                    })

            if not source_contents:
                self.log(f"  No grounding sources available - cannot verify", 'warning')
                return {
                    'all_pass': False,
                    'reasoning': "No grounding sources available for verification",
                    'product_results': []
                }

            source_text = "\n".join([f"Source: {s['title']}\n{s['content']}\n---" for s in source_contents])

            prompt = f"""You are verifying if a criterion is supported by grounding sources.

                        Criterion to verify: {criterion['description']}

                        Grounding Source Content:
                        {source_text}

                        IMPORTANT: Based ONLY on the source content above, is the criterion's claim supported or verifiable?
                        - If the sources contain information that supports or validates the criterion: pass = true
                        - If the sources do not mention or support the criterion: pass = false

                        Return ONLY valid JSON:
                        {{
                        "pass": true/false,
                        "reason": "Brief explanation citing which source(s) support or contradict the criterion"
                        }}"""

            try:
                response = self._generate(prompt)

                if '```json' in response:
                    response = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    response = response.split('```')[1].split('```')[0].strip()

                check_result = json.loads(response)

                all_pass = check_result.get('pass', False)
                reasoning = check_result.get('reason', '')

                status = "✓" if all_pass else "✗"
                self.log(f"  {status} Holistic grounding check: {reasoning[:100]}...",
                        'success' if all_pass else 'error')

                return {
                    'all_pass': all_pass,
                    'reasoning': f"Holistic grounding verification: {reasoning}",
                    'product_results': []
                }

            except Exception as e:
                # If it's a rate limit error, re-raise to fail the entire task for retry
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    self.log(f"Rate limit hit - failing task for retry: {e}", 'error')
                    raise  # Re-raise to fail the task
                
                self.log(f"  Error in holistic grounding check: {e}", 'error')
                return {
                    'all_pass': False,
                    'reasoning': f"Error during holistic grounding check: {str(e)}",
                    'product_results': []
                }

        # CASE 2: Has products (Shopping domain - specific products with mapped sources)
        for product in product_map:
            product_name = product['product_name']
            source_indices = product.get('source_indices', [])

            self.log(f"  Checking {product_name} (sources: {source_indices})")

            # Get markdown content for this product's sources
            source_contents = []
            for idx in source_indices:
                source = next((s for s in all_sources if s['source_number'] == idx + 1), None)
                if source and source.get('webpage_content', {}).get('markdown'):
                    source_contents.append({
                        'title': source['source_title'],
                        # 'content': source['webpage_content']['markdown'][:50000]  # Limit size
                        'content': source['webpage_content']['markdown']   # No limit size

                    })

            # Ask Gemini to verify against sources
            source_text = "\n".join([f"Source: {s['title']}\n{s['content']}\n---" for s in source_contents])

            prompt = f"""

            Your job is to check that our grading of a model's response is correct, based on the source material.

            We gave the model a prompt asking for recommendations. It then gave a response with suitable recommendations.
            We graded whether the response meets the following criterion, and found that it passed: {criterion['description']}
            The recommendation, product, or shop that passed is called: {product_name}

            We now want you to check whether the recommendation actually passes the criterion given the source material.
            For instance, if the criterion is "The response recommends a veterinarian" and the recommendation, product, or shop in the response is "St Barnard's vets", you need to check the source material to confirm that "St Barnard's vets" actually is a veterinarian.
            This is important for making sure that the model has not hallucinated in its response. We want to make sure that our grading of the model's responses is correct and the model is not making anything up.

            Another example:
            Criterion: "Recommends food themed games" and the recommendation, product, or shop returned in the response is "Overcooked! All you can eat", you need to check the source material to confirm that the "Overcooked! All you can eat" actually is a food themed game.

            Another example:
            Criterion: "Recommends badminton rackets under $25" and the recommendation, product, or shop returned in the response is "Puma Xr1234 Pro badminton racket", you need to check the source material to confirm that the Puma Xr1234 Pro badminton racket actually is a badminton racket and is under $25.

            Another example:
            Criterion: "Provides prices for each item" and the recommendation, product, or shop returned in the response is "XBox 360 controller", you need to check the source material to confirm that the XBox 360 controller actuallyhas a price.

            Source content:
                {source_text}

                The model response returned: {product_name}. The response passed this criterion: {criterion['description']}. Based on ONLY the source content, should it have passed?

                The recommendation, product, or shop PASSES if the criterion is true in AT LEAST ONE source (it does NOT need to be supported by all sources).

                Return ONLY valid JSON:
                {{
                "pass": true/false,
                "reason": "Brief explanation citing which source(s) confirmed or denied the criterion"
                }}"""

            try:
                response = self._generate(prompt)

                if '```json' in response:
                    response = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    response = response.split('```')[1].split('```')[0].strip()

                check_result = json.loads(response)

                product_results.append({
                    'product_name': product_name,
                    'pass': check_result.get('pass', False),
                    'reason': check_result.get('reason', ''),
                    'sources_checked': [s['title'] for s in source_contents]
                })

                # Log result
                status = "✓" if check_result['pass'] else "✗"
                self.log(f"    {status} {product_name}: {check_result['reason']}",
                        'success' if check_result['pass'] else 'error')

                if not check_result.get('pass', False):
                    all_pass = False

            except Exception as e:
                # If it's a rate limit error, re-raise to fail the entire task for retry
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    self.log(f"Rate limit hit - failing task for retry: {e}", 'error')
                    raise  # Re-raise to fail the task
                
                self.log(f"    Error checking {product_name}: {e}", 'error')
                product_results.append({
                    'product_name': product_name,
                    'pass': False,
                    'reason': f"Error: {str(e)}",
                    'sources_checked': []
                })
                all_pass = False

        # NEW: Smart scoring logic based on evaluation_type and required_count
        pass_count = sum(1 for p in product_results if p['pass'])
        total_count = len(product_results)

        # Determine overall pass/fail based on evaluation_type and required_count
        if evaluation_type == 'per_product_any':
            # "At least N" requirement
            if required_count == -1:
                # Edge case: treat as "all"
                all_pass = (pass_count == total_count)
            else:
                # "At least N" requirement
                all_pass = (pass_count >= required_count)

            reasoning = f"Source verification ({pass_count}/{total_count} passed): "
            if all_pass:
                req_text = f"≥{required_count}" if required_count != -1 else f"all {total_count}"
                reasoning += f"Met requirement (needed {req_text})"
            else:
                req_text = f"≥{required_count}" if required_count != -1 else f"all {total_count}"
                reasoning += f"Did not meet requirement (needed {req_text})"

        elif evaluation_type == 'per_product_all':
            # "All" or "only" - need ALL to pass
            all_pass = (pass_count == total_count) if product_results else False
            reasoning = f"Source verification ({pass_count}/{total_count} passed): "
            if all_pass:
                reasoning += "All recommendations verified in sources"
            else:
                reasoning += f"{total_count - pass_count} recommendation(s) failed source verification"

        else:
            # Holistic or unknown - default behavior
            all_pass = all(p['pass'] for p in product_results) if product_results else True
            reasoning = "Source verification: " + ('All verified' if all_pass else 'Some failed')

        # Add detailed individual product results to reasoning
        if product_results:
            reasoning += "\n\nIndividual verification results:"
            for pr in product_results:
                status = "✓ PASS" if pr['pass'] else "✗ FAIL"
                reasoning += f"\n  {status}: {pr['product_name']}"
                reasoning += f"\n    → {pr['reason'][:150]}"
                if pr.get('sources_checked'):
                    reasoning += f"\n    Sources: {', '.join(pr['sources_checked'])}"

        return {
            'all_pass': all_pass,
            'reasoning': reasoning,
            'product_results': product_results
        }

    def extract_links_for_product(self, product_name, response_text):
        """Extract URLs from response text for a specific product"""
        # Use Gemini to extract links mentioned for this product
        prompt = f"""Extract ALL URLs mentioned for product: {product_name}

        Response text:
        {response_text}

        Find URLs (http://, https://) mentioned in context of {product_name} specifically..

        Return ONLY valid JSON:
        {{
        "urls": ["url1", "url2", ...]
        }}"""

        try:
            text = self._generate(prompt)

            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            data = json.loads(text)
            return data.get('urls', [])
        except Exception as e:
            # If it's a rate limit error, re-raise to fail the entire task for retry
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                self.log(f"Rate limit hit - failing task for retry: {e}", 'error')
                raise  # Re-raise to fail the task
            
            # No fallback - cannot reliably extract product-specific URLs with regex
            self.log(f"  Failed to extract links for {product_name}: {e}", 'warning')
            return []

    def grade_link_criterion(self, criterion, response_text, product_map, sources, shop_vs_product='Product', domain='Shopping'):
        """
        Grade "Provides link(s)" criterion with link verification

        Stage 1: Check if links are provided in response
        Stage 2: Verify links meet requirement

        Args:
            shop_vs_product: 'Product' or 'Shop' (Shopping domain only)
            domain: 'Shopping' or 'Gaming'
        """
        mode_text = f"{shop_vs_product} mode" if domain == 'Shopping' else f"{domain} domain"
        self.log(f"Special handling: Link criterion ({mode_text})")

        # Fail fast: No recommendations identified means criterion cannot be met
        if not product_map:
            self.log("No recommendations identified - cannot evaluate link criterion", 'error')
            return {
                'criterion_id': criterion.get('criterion_id', criterion['id']),
                'description': criterion['description'],
                'type': criterion.get('type', ''),
                'score': 0,
                'stage_reached': 'response_text',
                'stage_1_result': {
                    'all_pass': False,
                    'reasoning': 'No recommendations identified in response - cannot evaluate links',
                    'product_results': []
                },
                'stage_2_result': None,
                'reasoning': 'Failed: No recommendations identified to check for links'
            }

        # Stage 1: Extract links for each product
        product_link_results = []
        all_have_links = True

        for product in product_map:
            product_name = product['product_name']
            links = self.extract_links_for_product(product_name, response_text)

            product_link_results.append({
                'product_name': product_name,
                'links_found': links,
                'has_links': len(links) > 0
            })

            if len(links) == 0:
                all_have_links = False
                self.log(f"  No links found for {product_name}", 'warning')
            else:
                self.log(f"  Found {len(links)} link(s) for {product_name}", 'success')

        if not all_have_links:
            return {
                'criterion_id': criterion.get('criterion_id', criterion['id']),
                'description': criterion['description'],
                'type': criterion.get('type', ''),  # Include criterion type
                'score': 0,
                'stage_reached': 'response_text',
                'stage_1_result': {
                    'all_pass': False,
                    'reasoning': 'One or more products missing purchase links',
                    'product_results': product_link_results
                },
                'stage_2_result': None,
                'reasoning': 'Failed: Not all products have purchase links in response'
            }

        # Stage 2 Grounding: Verify each product's links
        self.log("Stage 2: Verifying purchase links...")

        all_products_pass = True
        link_verification_results = []

        for product_data in product_link_results:
            product_name = product_data['product_name']
            links = product_data['links_found']

            # Product passes if at least ONE link is valid
            product_passes = False
            link_checks = []

            for url in links:
                self.log(f"  Verifying {url} for {product_name}...")

                # Route by domain
                if domain == 'Gaming':
                    # Gaming: Verify against criterion description
                    verification = verify_gaming_link(product_name, url, self.firecrawl, criterion['description'])
                else:
                    # Shopping: Verify is purchasing page
                    verification = verify_purchase_link(product_name, url, self.firecrawl, shop_vs_product)

                link_checks.append({
                    'url': url,
                    'is_valid': verification['is_valid'],
                    'page_type': verification['page_type'],
                    'reason': verification['reason']
                })

                if verification['is_valid']:
                    success_msg = f"Valid link (meets criterion)" if domain == 'Gaming' else f"Valid purchase page"
                    product_passes = True
                    self.log(f"    {success_msg}", 'success')
                    break  # One valid link is enough
                else:
                    self.log(f"    Invalid: {verification['reason']}", 'warning')

            link_verification_results.append({
                'product_name': product_name,
                'pass': product_passes,
                'links_checked': link_checks
            })

            if not product_passes:
                all_products_pass = False

        score = 1 if all_products_pass else -1

        return {
            'criterion_id': criterion.get('criterion_id', criterion['id']),
            'description': criterion['description'],
            'type': criterion.get('type', ''),  # Include criterion type
            'score': score,
            'stage_reached': 'link_verification',
            'stage_1_result': {
                'all_pass': True,
                'reasoning': 'All products have links',
                'product_results': product_link_results
            },
            'stage_2_result': {
                'all_pass': all_products_pass,
                'reasoning': f"Link verification: {'All valid' if all_products_pass else 'One or more invalid'}",
                'product_results': link_verification_results
            },
            'reasoning': f"Links provided and {'all verified' if all_products_pass else 'verification failed'}"
        }

    def grade_non_grounding_criterion(self, criterion, response_text):
        """
        Grade non-grounding criteria (single stage, no source verification)

        These criteria types don't require grounding source verification:
        - Meets quantity requirement
        - Product is in a set list/recommends specific product
        - Recommends buying habit
        - Other

        Checks response text only with strict per-recommendation validation.
        Returns score: 1 if pass, 0 if fail
        """
        self.log(f"Non-grounding criterion (response text only)")

        # Use the SAME improved prompt as check_response_text (for consistency)
        prompt = f"""You are evaluating a response against a specific criterion.

        Criterion to evaluate: {criterion['description']}

        Response Text:
        {response_text}

        SPECIAL INSTRUCTION FOR CONDITIONAL CRITERIA:
        If this criterion starts with "If" (conditional), treat "available in X", "comes in X", "offered in X" as EQUIVALENT to "recommends X". Stock caveats like "confirm availability" or "check stock" do NOT make it fail.

        CRITICAL EVALUATION RULES:

        1. **ZERO BACKGROUND KNOWLEDGE**: You must COMPLETELY IGNORE any knowledge you have about these recommendation/products. Pretend you have NEVER heard of them before.

        2. **ONLY STATEMENTS ABOUT THE RECOMMENDATION/PRODUCT COUNT**: A criterion is met for a recommendation/product ONLY if the response text DIRECTLY states the required information for that specific recommendation/product or if it is obviously inferrable.
        Example:
        - If the criterion says "supports 2+ players" → The response MUST explicitly say "multiplayer", "2 players", "co-op", "local multiplayer" or similar
        - If the criterion says "compatible with X" → The response MUST explicitly mention compatibility with X
        - If something is IMPLIED, HINTED AT, or requires INFERENCE → It does NOT count

        Multi-purpose recommendation/products are fine:
        Criterion: "Recommends only mixed martial arts (MMA) gloves"
        Recommendation/Product: "Gloves for MMA and Boxing"
        Evaluation: meets_criterion = TRUE (advertised for MMA, multi-sport OK)

        3. **DETERMINE SCOPE**:
        - HOLISTIC: Applies to the overall response (e.g., "recommends exactly 3 items")
        - PER_RECOMMENDATION: Applies to each individual recommendation (e.g., "all items must be X")

        SCORING:
        - Criteria with "only" or "all": Pass only if EVERY recommendation answer or product meets it
        - Criteria about quantity/count: Evaluate the overall response

        KEY EXAMPLES:

        Conditional: "If recommending item A, must have feature B"
        - "Item A - available in feature B" → TRUE
        - "Item A - while available in feature B, check stock" → TRUE
        - "Item A" (no mention of B) → FALSE
        - "Item C" (not item A) → TRUE (condition not met)

        Only: "Recommends only MMA gloves"
        - "MMA/Boxing gloves" → TRUE
        - "Boxing gloves" (no MMA) → FALSE

        Overall be intuitive and logical, if it should pass, pass, if it should fail, fail. Be Reasonable and logical.

        Return ONLY valid JSON:
        {{
        "pass": true/false,
        "reasoning": "Clear explanation of why pass or fail.",
        "evaluation_type": "holistic|per_recommendation",
        "violation_found": true/false
        }}"""

        try:
            response = self._generate(prompt)

            # Parse JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()

            data = json.loads(response)

            all_pass = data.get('pass', False)

            # Log results
            eval_type = data.get('evaluation_type', 'unknown')
            violation = data.get('violation_found', False)

            status = "✓" if all_pass else "✗"
            # Print directly without emoji prefix since status symbol is already included
            print(f"    {status} Evaluation ({eval_type}): {data.get('reasoning', '')[:100]}...")

            if violation:
                self.log(f"  Violation detected - criterion fails", 'warning')

            score = 1 if all_pass else 0

            stage_1_result = {
                'all_pass': all_pass,
                'reasoning': data.get('reasoning', ''),
                'evaluation_type': eval_type,
                'violation_found': violation
            }

            self.log(f"Criterion {criterion['id']} {'PASSED' if all_pass else 'FAILED'} (Score: {score})",
                    'success' if all_pass else 'error')

            return {
                'criterion_id': criterion.get('criterion_id', criterion['id']),
                'description': criterion['description'],
                'type': criterion.get('type', ''),  # Include criterion type
                'score': score,
                'stage_reached': 'response_text_only',
                'stage_1_result': stage_1_result,
                'stage_2_result': None,  # No stage 2 for non-grounding
                'reasoning': f"Non-grounding check: {data.get('reasoning', '')}"
            }

        except Exception as e:
            # If it's a rate limit error, re-raise to fail the entire task for retry
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                self.log(f"Rate limit hit - failing task for retry: {e}", 'error')
                raise  # Re-raise to fail the task
            
            self.log(f"Error in non-grounding criterion check: {e}", 'error')
            return {
                'criterion_id': criterion.get('criterion_id', criterion['id']),
                'description': criterion['description'],
                'type': criterion.get('type', ''),  # Include criterion type
                'score': 0,
                'stage_reached': 'response_text_only',
                'stage_1_result': {
                    'all_pass': False,
                    'reasoning': f"Error during check: {str(e)}",
                    'evaluation_type': 'error',
                    'violation_found': False
                },
                'stage_2_result': None,
                'reasoning': f"Error: {str(e)}"
            }

    def grade_criterion(self, criterion, response_text, product_map, sources, shop_vs_product, domain):
        """
        Grade a single criterion with routing based on type

        Args:
            criterion: Criterion dict with id, description, type, etc.
            response_text: Model's response text
            product_map: List of products with source mappings
            sources: All grounding sources
            shop_vs_product: 'Product' or 'Shop' - affects link verification mode (Shopping only)
            domain: 'Shopping', 'Gaming', 'Food', or 'DIY' - affects verification logic

        Returns:
            dict: Full grading result with score (0, 1, or -1)
        """
        self.log(f"\n{'='*60}", 'info')
        self.log(f"Grading Criterion {criterion['id']}: {criterion['description']}", 'info')
        criterion_type = criterion.get('type', 'standard')
        self.log(f"Type: {criterion_type}", 'info')
        self.log(f"{'='*60}", 'info')

        # Route based on criterion type
        if criterion_type == 'Provides link(s)':
            return self.grade_link_criterion(criterion, response_text, product_map, sources, shop_vs_product, domain)

        # Non-grounding criteria (use database column)
        elif criterion.get('grounded_status') == 'Not Grounded':
            return self.grade_non_grounding_criterion(criterion, response_text)

        # Grounded criteria (features, pricing)

        # Stage 1: Response text check
        response_check = self.check_response_text(criterion, response_text, product_map)

        if not response_check['all_pass']:
            # Failed at response text stage
            self.log(f"Criterion {criterion['id']} FAILED at Stage 1 (response text)", 'error')
            return {
                'criterion_id': criterion.get('criterion_id', criterion['id']),
                'description': criterion['description'],
                'type': criterion.get('type', ''),  # Include criterion type
                'score': 0,
                'stage_reached': 'response_text',
                'stage_1_result': response_check,
                'stage_2_result': None,
                'reasoning': f"Failed response text check: {response_check['reasoning']}"
            }

        self.log(f"Stage 1 passed - all products meet criterion in response text", 'success')
        # For now, hardcode to per_product_all for all criteria, so we ensure that every product must meets the criterion
        
        # evaluation_type = 'per_product_all'
        # required_count = -1
        # Stage 2: Grounded sources check
        # Use evaluation_type and required_count from Stage 1 response check
        evaluation_type = response_check.get('evaluation_type', 'per_product_all')
        required_count = response_check.get('required_pass_count', -1)
        source_check = self.check_grounded_sources(criterion, product_map, sources, evaluation_type, required_count)

        if source_check['all_pass']:
            self.log(f"Criterion {criterion['id']} PASSED - verified in sources (Score: 1)", 'success')
            score = 1
        else:
            self.log(f"Criterion {criterion['id']} FAILED at Stage 2 - not verified in sources (Score: -1)", 'error')
            score = -1

        return {
            'criterion_id': criterion.get('criterion_id', criterion['id']),
            'description': criterion['description'],
            'type': criterion.get('type', ''),  # Include criterion type
            'score': score,
            'stage_reached': 'grounded_sources',
            'stage_1_result': response_check,
            'stage_2_result': source_check,
            'reasoning': f"Stage 1: {response_check['reasoning']} | Stage 2: {source_check['reasoning']}"
        }

    def grade_all(self, input_file, output_file='./data/extracted/autograder-results.json', domain='Shopping', model_name=None):
        """
        Grade all criteria for a test case

        Args:
            input_file: Path to scraped sources JSON
            output_file: Path to save autograder results
            domain: Domain name ('Shopping', 'Food', 'Gaming') - affects grading rules
            model_name: Model name (e.g. 'gemini-2.5-pro') - used to determine correct table for Shop vs. Product lookup

        Input: grounding-sources.json with query, responseText, productSourceMap, criteria, sources
        Output: autograder-results.json with scores and detailed results
        """
        import time
        start_time = time.time()

        print(f"\nStarting Autograder ({domain} Domain)")
        print(f"{'='*60}\n")

        # Load input
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        task_id = data.get('task_id')
        test_id = data.get('test_id', '')
        query = data.get('query', '')
        response_text = data.get('responseText', '')
        product_map = data.get('productSourceMap', [])
        criteria = data.get('criteria', [])
        sources = data.get('sources', [])
        pipeline_timing = data.get('pipeline_timing', {})

        # Read Shop vs. Product from local file (Shopping domain only)
        shop_vs_product = data.get('shop_vs_product', None) if domain == 'Shopping' else None

        self.log(f"Loaded test case")
        self.log(f"  Domain: {domain}")
        self.log(f"  Products: {len(product_map)}")
        self.log(f"  Criteria: {len(criteria)}")
        self.log(f"  Sources: {len(sources)}")
        if domain == 'Shopping':
            self.log(f"  Shop vs. Product: {shop_vs_product}")

        if not criteria:
            self.log("No criteria found in input file!", 'error')
            return

        # Grade each criterion in parallel for speed
        results = [None] * len(criteria)  # Preserve order
        timings = [None] * len(criteria)  # Preserve order
        scores = []
        scores_with_types = []
        criterion_timings = []

        def grade_single_criterion(idx, criterion):
            """Grade a single criterion and return (idx, result, time)"""
            criterion_start = time.time()
            result = self.grade_criterion(criterion, response_text, product_map, sources, shop_vs_product, domain)
            criterion_time = time.time() - criterion_start
            return (idx, result, criterion_time)

        # Run criteria grading in parallel (100 workers)
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = {
                executor.submit(grade_single_criterion, idx, criterion): idx
                for idx, criterion in enumerate(criteria)
            }

            for future in as_completed(futures):
                idx, result, criterion_time = future.result()
                results[idx] = result
                timings[idx] = criterion_time

                # Log completion
                self.log(f"✓ Completed criterion {idx+1}/{len(criteria)}", 'info')

        # Extract scores and timings in order
        for idx, result in enumerate(results):
            scores.append(result['score'])
            scores_with_types.append([
                result['score'],
                criteria[idx].get('type', 'standard'),
                criteria[idx].get('hurdle_tag', 'Not')  # Add hurdle tag
            ])
            criterion_timings.append({
                'criterion_id': result.get('criterion_id', criteria[idx]['id']),
                'time_seconds': round(timings[idx], 2)
            })

        # Create output
        output = {}

        # Add IDs at top if available
        if task_id is not None:
            output['task_id'] = task_id
        if test_id:
            output['test_id'] = test_id

        # Calculate total time
        total_time = time.time() - start_time

        # Build timing object with pipeline timing included
        timing_data = {
            'autograder_total_seconds': round(total_time, 2),
            'per_criterion': criterion_timings
        }

        # Include grounding pipeline timing if available
        if pipeline_timing:
            timing_data['grounding_pipeline_total_seconds'] = pipeline_timing.get('total_seconds')
            timing_data['grounding_pipeline_scraping_seconds'] = pipeline_timing.get('scraping_seconds')
            timing_data['grounding_pipeline_processing_seconds'] = pipeline_timing.get('processing_seconds')

        # Calculate Total Score and Total Hurdle Score
        total_score = sum(scores)

        # Calculate Total Hurdle Score (same logic as supabase_writer.py)
        hurdle_scores = [score[0] for score in scores_with_types if len(score) >= 3 and score[2] == 'Hurdle']

        if not hurdle_scores:
            # No hurdle criteria
            total_hurdle_score = 0
        elif any(score <= 0 for score in hurdle_scores):
            # ANY hurdle failed (score 0 or -1) → Total Hurdle Score = 0
            total_hurdle_score = 0
        else:
            # ALL hurdles passed (all are 1) → Total Hurdle Score = Total Score
            total_hurdle_score = total_score

        # Add rest of output (domain-aware)
        output.update({
            'query': query,
            'criteria': criteria,  # Include full criteria list for reference
            'num_criteria': len(criteria),
            'criteria_scores': scores_with_types,  # [score, type, hurdle_tag] for each criterion
            'criteria_scores_only': scores,  # Just scores for backward compatibility
            'total_score': total_score,
            'total_hurdle_score': total_hurdle_score,
            'timing': timing_data,
            'summary': {
                'pass_count': sum(1 for s in scores if s == 1),
                'fail_response_count': sum(1 for s in scores if s == 0),
                'fail_source_count': sum(1 for s in scores if s == -1),
                'total': len(scores),
                'hurdle_count': len(hurdle_scores),
                'hurdle_pass_count': sum(1 for s in hurdle_scores if s == 1)
            },
            'detailed_results': results
        })

        # Add num_products only for Shopping domain (Food doesn't have products)
        if domain == 'Shopping':
            output['num_products'] = len(product_map)

        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.log(f"\nResults saved to: {output_file}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"Autograding Summary")
        print(f"{'='*60}\n")
        print(f"Criteria Scores (with types):")
        for i, score_data in enumerate(scores_with_types, 1):
            score = score_data[0]
            ctype = score_data[1]
            hurdle = score_data[2] if len(score_data) > 2 else 'Not'
            timing = criterion_timings[i-1]['time_seconds']
            hurdle_marker = "🚧 HURDLE" if hurdle == 'Hurdle' else ""
            print(f"  {i}. Score: {score:2} | Type: {ctype:40} | {hurdle_marker:10} | Time: {timing}s")
        print(f"\nAggregate:")
        print(f"  Pass (1):           {output['summary']['pass_count']}")
        print(f"  Fail Response (0):  {output['summary']['fail_response_count']}")
        print(f"  Fail Source (-1):   {output['summary']['fail_source_count']}")
        print(f"\nScoring:")
        print(f"  Total Score:        {output['total_score']}")
        print(f"  Total Hurdle Score: {output['total_hurdle_score']}")
        print(f"  Hurdle Criteria:    {output['summary']['hurdle_count']} ({output['summary']['hurdle_pass_count']} passed)")
        print(f"\nTiming:")
        if pipeline_timing:
            print(f"  Grounding Pipeline: {pipeline_timing.get('total_seconds', 0):.2f}s")
            print(f"    - Scraping: {pipeline_timing.get('scraping_seconds', 0):.2f}s")
        print(f"  Autograder: {total_time:.2f}s")
        if pipeline_timing:
            total_combined = pipeline_timing.get('total_seconds', 0) + total_time
            print(f"  Combined Total: {total_combined:.2f}s")
        print(f"\n✅ Autograding complete!")

        return output


def main():
    if len(sys.argv) < 2:
        print("❌ Error: No input file provided")
        print("\nUsage: python harness/autograder.py <grounding-sources-file> [output-file]")
        print("Example: python harness/autograder.py data/extracted/grounding-sources.json")
        sys.exit(1)

    input_file = sys.argv[1]

    # Default: infer from input file name
    if 'results/' in input_file and '/2_scraped_sources.json' in input_file:
        output_file = input_file.replace('/2_scraped_sources.json', '/3_autograder_results.json')
    else:
        output_file = './data/extracted/autograder-results.json'

    # Override if specified
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)

    grader = Autograder()
    grader.grade_all(input_file, output_file)


if __name__ == '__main__':
    main()

