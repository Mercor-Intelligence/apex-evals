#!/usr/bin/env python3
"""
Re-run the full evaluation pipeline (Grounding + Autograder) on an existing response.

This script:
1. Fetches the EXISTING response text from Supabase (it does NOT call the model again).
2. Re-runs the Grounding Pipeline:
   - Extracts products and links from the response text.
   - Scrapes all sources (API provided + Text extracted).
   - Generates a fresh `2_scraped_sources.json`.
3. Re-runs the Autograder:
   - Grades the fresh scraped data against the criteria.
   - Generates a fresh `3_autograder_results.json`.
4. Updates Supabase with the new results.

Use this when:
- You've improved the grounding/scraping logic.
- You've updated the autograder logic.
- You want to re-verify a task without paying for a new model inference.
"""

import json
import os
import sys
import tempfile
import time

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.logging_config import setup_logging
logger = setup_logging(__name__)

# Make Supabase optional
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # Type hint placeholder

# Add scripts and core to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'harness'))
sys.path.insert(0, os.path.join(project_root, 'configs'))

# Import from configs
from configs.config import config
from configs.domain_config import get_domain_config_for_model
from configs.model_providers import get_provider_for_model

# Import from scripts
from autograder import Autograder

# Supabase (optional - disabled by default, use --supabase flag to enable)
USE_SUPABASE = False
supabase = None
logger.warning("Running without Supabase (local files only mode)")


def fetch_task_response(task_id, domain, model_name, run_number=None):
    """
    Fetch existing response text and test case

    Reads from Supabase if available, otherwise from local files.
    """
    # If no Supabase, read from local files
    if not USE_SUPABASE:
        from local_file_reader import read_grounded_response
        return read_grounded_response(task_id, domain, model_name, run_number)

    # Supabase path
    domain_config = get_domain_config_for_model(domain, model_name)
    task_table = domain_config['task_table']
    criteria_table = domain_config['criteria_table']

    # Use generic 'Description' as the description column since criteria_shopping_gemini-2.5-flash table uses 'Criterion' column name
    # but the domain config defaults to 'Description' and the table mapping for Gemini/Shopping maps to 'Criterion Type (Shopping)'.
    # We'll dynamically check the column names or fall back to 'Criterion'.

    suffix = f" - {run_number}" if run_number else ""

    print(f"Fetching task {task_id} from {task_table}...")

    # Fetch task data
    task_result = supabase.table(task_table).select('*').eq('"Task ID"', task_id).execute()

    if not task_result.data:
        raise ValueError(f"Task {task_id} not found in {task_table}")

    task_data = task_result.data[0]

    # Get response text
    response_text = task_data.get(f'Response Text{suffix}', '')
    if not response_text:
        raise ValueError(f"Task {task_id} has no Response Text{suffix}. Generate response first!")

    # Get direct grounding JSON (if exists) - Needed for API chunks
    # Try explicit column first, then check if it was stored in a different casing or field
    direct_grounding = (
        task_data.get(f'Direct Grounding{suffix}') or
        task_data.get(f'direct_grounding{suffix}') or
        {}
    )

    # Fetch criteria
    print(f"Fetching criteria from {criteria_table}...")
    criteria_result = supabase.table(criteria_table).select('*').eq('"Task ID"', task_id).execute()

    if not criteria_result.data:
        raise ValueError(f"No criteria found for task {task_id} in {criteria_table}")

    # Build test case (needed for autograder)
    # We re-construct what '0_test_case.json' would look like
    # Check for 'Description' or 'Criterion' column
    first_row = criteria_result.data[0]
    desc_col = 'Description' if 'Description' in first_row else 'Criterion'

    test_case_data = {
        'task_id': task_id,
        'query': task_data.get('Specified Prompt', ''),
        'criteria': [
            {
                'id': c['Criterion ID'],
                'criterion_id': c['Criterion ID'],  # Add for autograder
                'description': c.get(desc_col, c.get('Criterion', 'No description')),
                # Add other fields if needed by autograder (e.g. type)
                'type': c.get(f'Criterion Type ({domain})', ''),
                'hurdle_tag': c.get('Hurdle Tag', 'Not'),  # Add hurdle tag
                'grounded_status': c.get('Criterion Grounding Check', '')  # Add grounded status
            }
            for c in criteria_result.data
        ]
    }

    print(f"✅ Loaded:")
    print(f"   - Response text: {len(response_text)} chars")
    print(f"   - Criteria: {len(test_case_data['criteria'])}")
    print(f"   - Direct grounding (API chunks): {'Yes' if direct_grounding else 'No'}")

    return response_text, direct_grounding, test_case_data


def run_grounding_pipeline(response_text, test_case_data, direct_grounding=None):
    """
    Run grounding pipeline (Stage 2)
    This handles product mapping, link extraction, and scraping.
    """
    print(f"\nRunning Grounding Pipeline (Stage 2)...")

    # Create input for grounding pipeline (simulating 1_grounded_response.json)
    # We pass the Raw Response Text and API Chunks.
    # The pipeline script will do the extraction and mapping.
    grounded_input = {
        'query': test_case_data.get('prompt', test_case_data.get('query', '')),
        'responseText': response_text,
        'criteria': test_case_data['criteria'],
        'task_id': test_case_data.get('task_id'),
        'direct_grounding': direct_grounding,
        # Required fields for pipeline to work
        'groundingChunks': direct_grounding.get('groundingChunks', []) if direct_grounding else [],
        'groundingSupports': direct_grounding.get('groundingSupports', []) if direct_grounding else []
    }

    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_grounded_input.json', delete=False) as tmp:
        json.dump(grounded_input, tmp, indent=2)
        grounded_file = tmp.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='_scraped_sources.json', delete=False) as tmp:
        scraped_file = tmp.name

    try:
        # Call grounding-pipeline.py
        import subprocess
        grounding_script = os.path.join(project_root, 'harness', 'grounding-pipeline.py')

        # Run the script
        result = subprocess.run(
            ['python3', grounding_script, grounded_file, scraped_file],
            stderr=subprocess.PIPE,  # Capture errors only, stdout flows to terminal
            text=True,
            timeout=900  # 15 minutes for scraping
        )

        if result.returncode != 0:
            error_msg = result.stderr or 'Unknown error'
            raise RuntimeError(f"Grounding pipeline failed: {error_msg[:500]}")

        # Load results
        with open(scraped_file, 'r') as f:
            scraped_data = json.load(f)

        print(f"✅ Grounding complete:")
        print(f"   - Products Found: {len(scraped_data.get('productSourceMap', []))}")
        print(f"   - Sources Scraped: {len(scraped_data.get('sources', []))}")

        return scraped_data

    finally:
        # Clean up
        if os.path.exists(grounded_file):
            os.unlink(grounded_file)
        if os.path.exists(scraped_file):
            os.unlink(scraped_file)


def run_autograder(scraped_data, domain, model_name=None):
    """
    Run autograder (Stage 3) on scraped data
    """
    print(f"\nRunning Autograder (Stage 3)...")

    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_scraped_for_grading.json', delete=False) as tmp_in:
        json.dump(scraped_data, tmp_in, indent=2)
        input_file = tmp_in.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='_autograder_results.json', delete=False) as tmp_out:
        output_file = tmp_out.name

    try:
        grader = Autograder()
        grader.grade_all(input_file, output_file, domain=domain, model_name=model_name)

        with open(output_file, 'r') as f:
            results = json.load(f)

        scores = results.get('criteria_scores_only', [])
        total_score = results.get('total_score')

        print(f"✅ Autograder complete:")
        print(f"   - Total Score: {total_score}")
        print(f"   - Scores: {scores}")

        return results

    finally:
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def update_supabase(task_id, domain, model_name, run_number, scraped_data, autograder_results):
    """
    Update Supabase with new results (preserving Response Text)
    """
    if not USE_SUPABASE:
        print(f"\n⚠️  Skipping Supabase update (not available)")
        return

    print(f"\nUpdating Supabase...")

    config = get_domain_config_for_model(domain, model_name)
    task_table = config['task_table']
    criteria_table = config['criteria_table']

    suffix = f" - {run_number}" if run_number else ""

    # Get scores from autograder results (already calculated)
    scores = autograder_results.get('criteria_scores_only', [])
    total_score = autograder_results.get('total_score')
    total_hurdle_score = autograder_results.get('total_hurdle_score')
    criteria_scores = autograder_results.get('criteria_scores', [])

    # Prepare Score Overview
    score_overview = {
        'summary': autograder_results.get('summary', {}),
        'detailed_results': autograder_results.get('detailed_results', []),
        'criteria_scores': criteria_scores,
        'num_criteria': autograder_results.get('num_criteria', 0)
    }

    # Prepare Product Source Map (Enhanced with URLs)
    product_source_map_raw = scraped_data.get('productSourceMap', [])
    grounding_source_meta_data = scraped_data.get('sources', [])

    enhanced_product_map = []
    for product in product_source_map_raw:
        # Create a clean product object
        enhanced_product = {
            'product_name': product.get('product_name', ''),
            'source_indices': product.get('source_indices', []),
            'source_urls': []
        }

        # Add URLs
        for idx in product.get('source_indices', []):
            # Find matching source (source_number is 1-based)
            source = next((s for s in grounding_source_meta_data if s.get('source_number') == idx + 1), None)
            if source:
                enhanced_product['source_urls'].append(source.get('source_link', ''))

        enhanced_product_map.append(enhanced_product)

    # 1. Update Task Table
    # We ONLY update the derived columns, NEVER the Response Text
    task_update = {
        f"Product Source Map{suffix}": enhanced_product_map,
        f"Grounding Source Meta Data{suffix}": grounding_source_meta_data,
        f"Failed Grounded Sites{suffix}": scraped_data.get('failed_grounded_sites', []),
        f"Scores{suffix}": scores,
        f"Total Score{suffix}": total_score,
        f"Total Hurdle Score{suffix}": total_hurdle_score,
        f"Score Overview{suffix}": score_overview
    }

    supabase.table(task_table).update(task_update).eq('"Task ID"', task_id).execute()

    # 2. Update Criteria Table
    detailed_results = autograder_results.get('detailed_results', [])
    non_grounding_types = config['non_grounding_types']

    for result in detailed_results:
        criterion_id = result['criterion_id']
        score = result['score']
        criterion_type = result.get('type', '')

        # Determine failure step
        if score == 1:
            failure_step = "Passed"
        elif score == 0:
            failure_step = "Stage 1"
        elif score == -1:
            failure_step = "Stage 2"
        else:
            failure_step = "Unknown"

        # Determine grounding check
        if domain == 'Food':
            grounding_check = "Not Grounded"
        else:
            grounding_check = "Not Grounded" if criterion_type in non_grounding_types else "Grounded"

        criteria_update = {
            f"Score{suffix}": score,
            f"Reasoning{suffix}": result,
            f"Failure Step{suffix}": failure_step,
            "Criterion Grounding Check": grounding_check
        }

        # Retry loop
        for attempt in range(3):
            try:
                supabase.table(criteria_table).update(criteria_update).eq('"Criterion ID"', criterion_id).execute()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    print(f"❌ Failed to update criterion {criterion_id}: {e}")

    print(f"✅ Database Updated successfully!")


def regrade_task(task_id, domain, model_name, run_number, dry_run=False):
    """
    Main regrade logic
    """
    try:
        start_time = time.time()

        print("="*60)
        print(f"REGRADE TASK (Re-run Grounding + Autograder)")
        print("="*60)
        print(f"Task ID: {task_id}")
        print(f"Domain: {domain}")
        print(f"Model: {model_name}")
        print(f"Run: {run_number}")
        print("="*60 + "\n")

        # 1. Fetch existing response
        response_text, direct_grounding, test_case_data = fetch_task_response(task_id, domain, model_name, run_number)

        # 2. Run Grounding Pipeline (This creates the Product Map and Scrapes Sources)
        scraped_data = run_grounding_pipeline(response_text, test_case_data, direct_grounding)

        # 3. Run Autograder
        autograder_results = run_autograder(scraped_data, domain, model_name)

        # 4. Save results to local files
        if not dry_run:
            provider_name = get_provider_for_model(model_name)
            results_dir = os.path.join(project_root, 'results', provider_name, model_name, domain, f'run_{run_number}', f'task_{task_id}')
            os.makedirs(results_dir, exist_ok=True)

            # Write 2_scraped_sources.json
            scraped_file = os.path.join(results_dir, '2_scraped_sources.json')
            with open(scraped_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=2, ensure_ascii=False)

            # Write 3_autograder_results.json
            autograder_file = os.path.join(results_dir, '3_autograder_results.json')
            with open(autograder_file, 'w', encoding='utf-8') as f:
                json.dump(autograder_results, f, indent=2, ensure_ascii=False)

            print(f"\nUpdated local files in {results_dir}")

        # 5. Update Supabase
        if not dry_run:
            update_supabase(task_id, domain, model_name, run_number, scraped_data, autograder_results)
        else:
            print("\n⚠️  Dry Run: Skipping database update and local file writes.")

        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ REGRADE COMPLETE ({elapsed:.1f}s)")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ Error during regrade: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Regrade a task (Re-run Grounding + Autograder)')
    parser.add_argument('task_id', type=int, help='Task ID')
    parser.add_argument('domain', choices=['Shopping', 'Gaming', 'Food', 'DIY'], help='Domain')
    parser.add_argument('model', help='Model name (e.g., gemini-2.5-pro)')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('--dry-run', action='store_true', help='Do not update database')
    parser.add_argument('--supabase', action='store_true', help='Enable Supabase (default: local files only)')

    args = parser.parse_args()

    # Enable Supabase if flag is set
    if args.supabase:
        global USE_SUPABASE, supabase
        if config.has_supabase() and SUPABASE_AVAILABLE:
            USE_SUPABASE = True
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            print("--supabase flag: Enabling Supabase operations")
        else:
            print("⚠️  --supabase flag set but Supabase not available")

    success = regrade_task(args.task_id, args.domain, args.model, args.run, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
