#!/usr/bin/env python3
"""
Make a grounded call using any supported model provider (Gemini, OpenAI, Claude)
Stage 1 of Pipeline: Only interacts with Model API
"""

# Load environment variables from .env file first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

import json
import os
import sys

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.logging_config import setup_logging
from configs.model_providers import get_provider_for_model, get_provider_instance

logger = setup_logging(__name__)

# Parse arguments - REQUIRE all inputs
if len(sys.argv) < 3:
    logger.error("Error: Missing required arguments")
    logger.error("\nUsage:")
    logger.error("  python3 make-grounded-call.py <test_case.json> <output.json> --model <model>")
    logger.error("\nExample:")
    logger.error("  python3 harness/make-grounded-call.py task_312/0_test_case.json output.json --model gemini-2.5-pro")
    logger.error("\nRequired arguments:")
    logger.error("  test_case.json: Must contain {\"prompt\": \"...\", \"criteria\": [...]}")
    logger.error("  output.json: Where to save the grounded response")
    logger.error("  --model: Model name (gpt-5, gemini-2.5-pro, sonnet-4.5, etc.)")
    sys.exit(1)

# Get required positional arguments
test_case_file = sys.argv[1]
output_file = sys.argv[2]

# Get REQUIRED --model flag (no default!)
if '--model' not in sys.argv:
    print("❌ Error: --model flag is required")
    print("   Example: --model gemini-2.5-pro")
    sys.exit(1)

model_idx = sys.argv.index('--model')
if model_idx + 1 >= len(sys.argv):
    print("❌ Error: --model flag requires a model name")
    sys.exit(1)

model_name = sys.argv[model_idx + 1]

# Validate test case file exists
if not os.path.exists(test_case_file):
    print(f"❌ Error: Test case file not found: {test_case_file}")
    sys.exit(1)

# Load test case from JSON
try:
    with open(test_case_file, 'r') as f:
        test_case = json.load(f)
except json.JSONDecodeError as e:
    print(f"❌ Error: Invalid JSON in test case file: {e}")
    sys.exit(1)

# Extract and validate required fields
query = test_case.get('prompt')
if not query:
    print("❌ Error: Test case must have 'prompt' field")
    sys.exit(1)

# Extract optional fields
criteria = test_case.get('criteria', [])
task_id = test_case.get('task_id')
test_id = test_case.get('test_id')  # Optional, just metadata
shop_vs_product = test_case.get('shop_vs_product')  # Shopping domain only

try:
    provider_name = get_provider_for_model(model_name)
    print(f"[*] Making grounded call using {model_name} ({provider_name})...\n")
except ValueError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Use query directly (Specified Prompt already contains any needed instructions)
print(f"Query (using Specified Prompt): {query[:200]}...\n")

# Get provider instance
print(f"Initializing {provider_name} provider...")
try:
    provider = get_provider_instance(model_name)
except ValueError as e:
    # Print to stderr so runner.py can capture it
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# Make the API call
print(f"Making API call to {model_name}...")
try:
    raw_response = provider.make_api_call(query, model_name)
except Exception as e:
    print(f"ERROR: API call failed: {e}", file=sys.stderr)
    sys.exit(1)

# Parse to normalized format
print(f"Parsing response to normalized format...")
grounding_json = provider.parse_response(raw_response, model_name)
print(f"✅ Parsed {len(grounding_json.get('groundingChunks', []))} chunks, {len(grounding_json.get('groundingSupports', []))} supports")

# Add query and metadata
grounding_json['query'] = query
grounding_json['model'] = model_name
grounding_json['provider'] = provider_name

print(f"✅ Response received!\n")
print("=" * 80)
print(grounding_json['responseText'])
print("=" * 80)

# Add metadata (already have from test_case loaded above)
if task_id:
    grounding_json['task_id'] = task_id
if test_id:
    grounding_json['test_id'] = test_id  # Optional metadata, pass through if present

# Add criteria
if criteria:
    grounding_json['criteria'] = criteria

# Add shop_vs_product (Shopping domain only)
if shop_vs_product:
    grounding_json['shop_vs_product'] = shop_vs_product

# Save to file
print(f"\nSaving grounding metadata to: {output_file}")

# Create directory if needed
os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(grounding_json, f, indent=2, ensure_ascii=False)

# Summary
api_source_count = len(grounding_json.get('groundingChunks', []))

print(f"\n✅ Found {api_source_count} API grounding sources")
print(f"✅ Found {len(grounding_json.get('groundingSupports', []))} citations")
if criteria:
    print(f"✅ Included {len(criteria)} criteria for autograding")
print(f"\n[*] Next: python3 harness/grounding-pipeline.py {output_file}")
