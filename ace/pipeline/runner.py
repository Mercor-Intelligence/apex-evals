#!/usr/bin/env python3
"""
Simple batch pipeline - Just calls the original working scripts in parallel
No reimplementation, no bugs!
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Add project root to path FIRST (before importing from configs)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from configs.logging_config import setup_logging
from configs.domain_config import get_domain_config_for_model
from configs.model_providers import get_provider_for_model

logger = setup_logging(__name__)

# Make Supabase optional
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # Type hint placeholder

# Load configuration
from configs.config import config

# Supabase (optional - disabled by default, use --supabase flag to enable)
USE_SUPABASE = False  # Default OFF - use --supabase flag to enable
supabase = None

# Constants
SUBPROCESS_TIMEOUT = 1800  # 30 minutes for API calls and scraping
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.1
UTM_PARAMS_TO_REMOVE = ['utm_source', 'utm_medium', 'utm_campaign']


def deduplicate_urls(urls):
    """
    Remove duplicate URLs by normalizing (removing UTM tracking params)

    Args:
        urls: List of URLs to deduplicate

    Returns:
        list: Deduplicated URLs
    """
    seen_normalized = {}
    deduplicated_urls = []

    for url in urls:
        if not url:
            continue

        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            # Remove UTM tracking parameters
            for param in UTM_PARAMS_TO_REMOVE:
                query_params.pop(param, None)

            clean_query = urlencode(query_params, doseq=True)
            normalized_parts = list(parsed)
            normalized_parts[4] = clean_query
            normalized_url = urlunparse(normalized_parts)

            if normalized_url not in seen_normalized:
                seen_normalized[normalized_url] = url
                deduplicated_urls.append(url)
        except:
            if url not in seen_normalized:
                seen_normalized[url] = url
                deduplicated_urls.append(url)

    return deduplicated_urls


def enhance_product_map(product_source_map_raw, grounding_source_meta_data):
    """
    Enhance product source map by adding URLs from source metadata

    Args:
        product_source_map_raw: Raw product source map with indices
        grounding_source_meta_data: Source metadata with URLs

    Returns:
        list: Enhanced product map with URLs and deduplicated sources
    """
    enhanced_product_map = []

    for product in product_source_map_raw:
        source_urls = []
        for idx in product.get('source_indices', []):
            # Find source by index (source_number is 1-based, indices are 0-based)
            source = next((s for s in grounding_source_meta_data if s.get('source_number') == idx + 1), None)
            if source:
                url = source.get('source_link', '')
                source_urls.append(url)
            else:
                source_urls.append('')

        # Deduplicate URLs
        deduplicated_urls = deduplicate_urls(source_urls)

        # Create product entry with desired key order: product_name, source_urls, source_indices
        enhanced_product = {
            'product_name': product.get('product_name', ''),
            'source_urls': deduplicated_urls,
            'source_indices': product.get('source_indices', [])
        }
        enhanced_product_map.append(enhanced_product)

    return enhanced_product_map


def extract_direct_grounding(data):
    """
    Extract direct grounding from data, checking multiple key names for backward compatibility

    Args:
        data: Data dictionary to search

    Returns:
        Direct grounding data or None
    """
    return (data.get('direct_grounding') or
            data.get('gemini_direct_grounding') or
            data.get('openai_direct_grounding') or
            data.get('claude_direct_grounding'))


def determine_failure_step(score):
    """
    Determine failure step from score

    Args:
        score: Criterion score (1, 0, -1)

    Returns:
        str: Failure step description
    """
    if score == 1:
        return "Passed"
    elif score == 0:
        return "Stage 1"
    elif score == -1:
        return "Stage 2"
    else:
        return "Unknown"


def retry_with_backoff(func, max_attempts=MAX_RETRY_ATTEMPTS):
    """
    Execute function with exponential backoff retry logic

    Args:
        func: Callable to execute
        max_attempts: Maximum retry attempts

    Returns:
        Result of successful function call

    Raises:
        Exception: If all attempts fail
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            else:
                raise


def build_common_task_data(scraped_data, enhanced_product_map, grounding_source_meta_data, suffix):
    """
    Build common task data dictionary shared between write functions

    Args:
        scraped_data: Scraped sources data
        enhanced_product_map: Enhanced product source map with URLs
        grounding_source_meta_data: Source metadata
        suffix: Column suffix (e.g., " - 1")

    Returns:
        dict: Common task data
    """
    task_id = scraped_data.get('task_id')

    data = {
        "Task ID": task_id,
        "Specified Prompt": scraped_data.get('query', ''),
        f"Response Text{suffix}": scraped_data.get('responseText', ''),
        f"Product Source Map{suffix}": enhanced_product_map,
        f"Grounding Source Meta Data{suffix}": grounding_source_meta_data,
        "Criteria List": scraped_data.get('criteria', []),
        f"Failed Grounded Sites{suffix}": scraped_data.get('failed_grounded_sites', [])
    }

    # Add direct grounding (provider-agnostic, backward compatible)
    direct_grounding = extract_direct_grounding(scraped_data)
    if direct_grounding:
        data[f"Direct Grounding{suffix}"] = direct_grounding

    # Add shop_vs_product (Shopping domain only, permanent field)
    if scraped_data.get('shop_vs_product'):
        data["Shop vs. Product"] = scraped_data['shop_vs_product']

    return data


def process_single_task_with_original_scripts(task_id, domain, model_name, run_number, skip_grading=False):
    """
    Process one task using the ORIGINAL working scripts

    Args:
        task_id: Task ID to process
        domain: Domain name
        model_name: Model name (e.g. 'gpt-5', 'gemini-2.5-pro')
        run_number: Run number (1-5) for variance testing
        skip_grading: If True, skip autograder (only do grounding + scraping)

    Returns:
        dict: Result with success status
    """
    logger.info(f"Task {task_id} - Run {run_number}" if run_number else f"Task {task_id}")

    start_time = time.time()

    try:
        # Get provider from model name (for directory structure)
        provider_name = get_provider_for_model(model_name)

        # Create directory: results/{provider}/{model}/{domain}/run_{run_number}/task_{id}
        results_dir = os.path.join(project_root, 'results', provider_name, model_name, domain, f'run_{run_number}', f'task_{task_id}')
        os.makedirs(results_dir, exist_ok=True)

        test_case_file = os.path.join(results_dir, '0_test_case.json')
        grounded_response_file = os.path.join(results_dir, '1_grounded_response.json')
        scraped_sources_file = os.path.join(results_dir, '2_scraped_sources.json')
        autograder_results_file = os.path.join(results_dir, '3_autograder_results.json')

        # Step 1: Create test case from database (only if doesn't exist)
        if not os.path.exists(test_case_file):
            if USE_SUPABASE:
                from supabase_reader import get_test_case
                domain_config = get_domain_config_for_model(domain, model_name)
                test_case = get_test_case(task_id, domain=domain, table_name=domain_config['criteria_table'])

                with open(test_case_file, 'w', encoding='utf-8') as f:
                    json.dump(test_case, f, indent=2, ensure_ascii=False)
            else:
                raise FileNotFoundError(
                    f"Test case file not found: {test_case_file}\n"
                    f"Without Supabase, you must run 'python3 pipeline/init_from_dataset.py all {model_name}' first to create test case files."
                )

        # Step 2: Call ORIGINAL make-grounded-call.py with model flag
        make_grounded_script = os.path.join(project_root, 'harness', 'make-grounded-call.py')
        result = subprocess.run(
            ['python3', make_grounded_script, test_case_file, grounded_response_file, '--model', model_name],
            stderr=subprocess.PIPE,  # Capture errors only, stdout flows to terminal
            text=True,
            timeout=SUBPROCESS_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = result.stderr or 'Unknown error'
            logger.error(f"Make-grounded-call.py failed for task {task_id}")
            return {'task_id': task_id, 'run': run_number, 'success': False, 'error': f'Grounded call failed: {error_msg[:500]}', 'stage': 'grounding'}

        # Step 3: Call ORIGINAL grounding-pipeline.py
        grounding_script = os.path.join(project_root, 'harness', 'grounding-pipeline.py')
        result = subprocess.run(
            ['python3', grounding_script, grounded_response_file, scraped_sources_file],
            stderr=subprocess.PIPE,  # Capture errors only, stdout flows to terminal
            text=True,
            timeout=SUBPROCESS_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = result.stderr or 'Unknown error'
            return {'task_id': task_id, 'run': run_number, 'success': False, 'error': f'Scraping failed: {error_msg[:500]}', 'stage': 'scraping'}

        # Step 4: Call ORIGINAL autograder.py (unless skip_grading)
        if not skip_grading:
            sys.path.insert(0, os.path.join(project_root, 'harness'))
            from autograder import Autograder

            grader = Autograder()
            grader.grade_all(scraped_sources_file, autograder_results_file, domain=domain, model_name=model_name)

        # Step 5: Write to database with run number
        if not skip_grading:
            write_results_to_numbered_columns(
                scraped_sources_file,
                autograder_results_file,
                domain,
                model_name,
                run_number
            )
        else:
            write_grounding_to_numbered_columns(
                scraped_sources_file,
                domain,
                model_name,
                run_number
            )

        elapsed = time.time() - start_time
        logger.info(f"Task {task_id} Run {run_number}: Complete in {elapsed:.1f}s")

        # Keep files (don't delete like temp files)
        # Files are saved in results/{domain}/task_{id}/ for later reference

        return {'task_id': task_id, 'run': run_number, 'success': True, 'time': elapsed}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Task {task_id} Run {run_number}: Error - {e}")
        return {'task_id': task_id, 'run': run_number, 'success': False, 'error': str(e), 'time': elapsed}


def write_grounding_to_numbered_columns(scraped_file, domain, model_name, run_number):
    """Write only grounding data (no scores) to numbered columns"""
    if not USE_SUPABASE:
        return  # Skip Supabase write when not available

    suffix = f" - {run_number}" if run_number else ""

    with open(scraped_file, 'r') as f:
        data = json.load(f)

    domain_config = get_domain_config_for_model(domain, model_name)
    task_table = domain_config['task_table']

    # Enhance product_source_map with URLs
    product_source_map_raw = data.get('productSourceMap', [])
    grounding_source_meta_data = data.get('sources', [])
    enhanced_product_map = enhance_product_map(product_source_map_raw, grounding_source_meta_data)

    # Build and write data
    update_data = build_common_task_data(data, enhanced_product_map, grounding_source_meta_data, suffix)
    supabase.table(task_table).upsert(update_data, on_conflict='Task ID').execute()


def write_results_to_numbered_columns(scraped_file, autograder_file, domain, model_name, run_number):
    """Write complete results to numbered columns using ORIGINAL supabase_writer logic"""
    if not USE_SUPABASE:
        return  # Skip Supabase write when not available

    suffix = f" - {run_number}" if run_number else ""

    with open(scraped_file, 'r') as f:
        scraped_data = json.load(f)

    with open(autograder_file, 'r') as f:
        autograder_data = json.load(f)

    task_id = scraped_data.get('task_id')
    domain_config = get_domain_config_for_model(domain, model_name)
    task_table = domain_config['task_table']
    criteria_table = domain_config['criteria_table']

    # Get scores from autograder results (already calculated)
    scores = autograder_data.get('criteria_scores_only', [])
    total_score = autograder_data.get('total_score')
    total_hurdle_score = autograder_data.get('total_hurdle_score')
    criteria_scores = autograder_data.get('criteria_scores', [])

    # Prepare score overview
    score_overview = {
        'summary': autograder_data.get('summary', {}),
        'detailed_results': autograder_data.get('detailed_results', []),
        'criteria_scores': criteria_scores,
        'num_criteria': autograder_data.get('num_criteria', 0)
    }

    # Enhance product_source_map with URLs
    product_source_map_raw = scraped_data.get('productSourceMap', [])
    grounding_source_meta_data = scraped_data.get('sources', [])
    enhanced_product_map = enhance_product_map(product_source_map_raw, grounding_source_meta_data)

    # STEP 1: Write to criteria table FIRST (all-or-nothing)
    # If any criteria write fails, we abort before writing to task table
    detailed_results = autograder_data.get('detailed_results', [])

    criteria_write_results = []

    for result in detailed_results:
        criterion_id = result['criterion_id']
        score = result['score']
        failure_step = determine_failure_step(score)

        # Criterion Grounding Check is pre-filled in input, no need to write it
        update_data = {
            f"Score{suffix}": score,
            f"Reasoning{suffix}": result,
            f"Failure Step{suffix}": failure_step
        }

        # Retry logic for connection errors
        try:
            retry_with_backoff(
                lambda: supabase.table(criteria_table).update(update_data).eq('"Criterion ID"', criterion_id).execute()
            )
            criteria_write_results.append({'criterion_id': criterion_id, 'success': True})
        except Exception as e:
            raise Exception(f"Failed to write criterion {criterion_id} after {MAX_RETRY_ATTEMPTS} attempts: {e}")

    # Check ALL criteria writes succeeded
    all_criteria_succeeded = all(r['success'] for r in criteria_write_results)

    if not all_criteria_succeeded:
        failed_ids = [r['criterion_id'] for r in criteria_write_results if not r['success']]
        raise Exception(f"Criteria writes failed for IDs: {failed_ids}. Aborting task table write.")

    # STEP 2: Only write to task table if ALL criteria writes succeeded
    task_data = build_common_task_data(scraped_data, enhanced_product_map, grounding_source_meta_data, suffix)

    # Add score-specific fields
    task_data[f"Score Overview{suffix}"] = score_overview
    task_data[f"Scores{suffix}"] = scores
    task_data[f"Total Score{suffix}"] = total_score
    task_data[f"Total Hurdle Score{suffix}"] = total_hurdle_score

    # Write to task table (with retry)
    try:
        retry_with_backoff(
            lambda: supabase.table(task_table).upsert(task_data, on_conflict='Task ID').execute()
        )
    except Exception as e:
        raise Exception(f"Failed to write task table after {MAX_RETRY_ATTEMPTS} attempts: {e}")


def get_pending_tasks_from_files(domain, model_name, run_number, force=False, skip_grading=False):
    """
    Get pending tasks from local filesystem (when Supabase not available)

    Scans results/{provider}/{model}/{domain}/run_{N}/ for existing task folders.
    Checks for final output file to determine completion.

    Args:
        domain: Domain name
        model_name: Model name
        run_number: Run number
        force: If True, return all tasks (ignore completion status)
        skip_grading: If True, check for 2_scraped_sources.json; else check 3_autograder_results.json

    Returns:
        list: Task IDs to process
    """
    from local_file_reader import get_pending_tasks as get_pending_from_local

    return get_pending_from_local(domain, model_name, run_number, force, skip_grading)


def get_pending_tasks(domain, model_name, run_number, force=False, skip_grading=False):
    """
    Get tasks that need processing for a specific run

    Reads from Supabase if available, otherwise from local files.

    Args:
        domain: Domain name
        model_name: Model name (e.g. 'gpt-5', 'gemini-2.5-pro')
        run_number: Run number (1-5)
        force: If True, return ALL tasks (ignore existing responses)
        skip_grading: If True, only check for scraping completion (not grading)

    Returns:
        list: Task IDs to process
    """
    if not USE_SUPABASE:
        return get_pending_tasks_from_files(domain, model_name, run_number, force, skip_grading)

    domain_config = get_domain_config_for_model(domain, model_name)
    criteria_table = domain_config['criteria_table']
    task_table = domain_config['task_table']

    # Get all unique task IDs from the criteria table (ACE table)
    # This is the source of truth for which tasks exist
    # IMPORTANT: Paginate to get ALL rows (default limit is 1000)
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        result = supabase.table(criteria_table).select('"Task ID"').range(offset, offset + page_size - 1).execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    all_task_ids = sorted(list(set([row['Task ID'] for row in all_rows])))

    # If force flag is set, return ALL tasks (ignore existing responses)
    if force:
        return all_task_ids

    # Otherwise, only return tasks that haven't been processed yet
    suffix = f" - {run_number}" if run_number else ""
    response_col = f'"Response Text{suffix}"'

    # Get existing responses from task_outputs
    task_result = supabase.table(task_table).select(f'"Task ID", {response_col}').execute()
    completed_tasks = set([row['Task ID'] for row in task_result.data if row.get(f'Response Text{suffix}')])

    # Pending = all tasks that exist in ACE table but don't have responses yet
    pending = [task_id for task_id in all_task_ids if task_id not in completed_tasks]

    return pending


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Batch pipeline using ORIGINAL scripts')
    parser.add_argument('domain', choices=['Shopping', 'Gaming', 'Food', 'DIY'])
    parser.add_argument('--model', nargs='+', required=True, help='Model name(s) - can specify multiple (e.g. --model gemini-2.5-pro gemini-2.5-flash)')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--run', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], required=True)
    parser.add_argument('--skip-autograder', action='store_true', help='Skip autograder (only run grounding + scraping)')
    parser.add_argument('--force', action='store_true', help='Re-run ALL tasks, ignoring existing responses')
    parser.add_argument('--supabase', action='store_true', help='Enable Supabase (default: local files only)')

    args = parser.parse_args()
    models = args.model  # Now a list

    # Enable Supabase if flag is set and credentials available
    if args.supabase:
        global USE_SUPABASE, supabase
        if config.has_supabase() and SUPABASE_AVAILABLE:
            USE_SUPABASE = True
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("--supabase flag: Enabling Supabase writes")
        else:
            logger.warning("--supabase flag set but Supabase not available (missing credentials or package)")

    # Validate all models first
    for model_name in models:
        try:
            get_provider_for_model(model_name)
        except ValueError as e:
            logger.error(f"Error: {e}")
            return

    # Process each model
    all_results = {}
    for model_name in models:
        provider_name = get_provider_for_model(model_name)

        # Validate Supabase table if enabled
        if USE_SUPABASE:
            try:
                domain_config = get_domain_config_for_model(args.domain, model_name)
                criteria_table = domain_config['criteria_table']
                test_result = supabase.table(criteria_table).select('"Task ID"').limit(1).execute()
                if not test_result.data:
                    logger.error(f"Supabase table '{criteria_table}' is empty!")
                    logger.error(f"Run 'python3 pipeline/init_from_dataset.py {args.domain} {model_name} --supabase' first")
                    continue
            except Exception as e:
                logger.error(f"Supabase validation failed for {model_name}: {e}")
                continue

        logger.info("=" * 60)
        logger.info(f"BATCH PIPELINE: {model_name}")
        logger.info("=" * 60)
        logger.info(f"Domain: {args.domain}")
        logger.info(f"Model: {model_name} ({provider_name})")
        logger.info(f"Run Number: {args.run}")
        logger.info(f"Workers: {args.workers}")
        if args.skip_autograder:
            logger.info("Mode: SKIP AUTOGRADER (grounding + scraping only)")
        if args.force:
            logger.warning("FORCE MODE: Re-running ALL tasks (will overwrite existing data)")

        # Get pending tasks
        pending = get_pending_tasks(args.domain, model_name, args.run, force=args.force, skip_grading=args.skip_autograder)

        if not pending:
            logger.info(f"No pending tasks for {model_name}!")
            all_results[model_name] = {'success': 0, 'failed': 0, 'time': 0}
            continue

        logger.info(f"Processing {len(pending)} tasks...")

        # Process in parallel
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_task_with_original_scripts,
                    task_id,
                    args.domain,
                    model_name,
                    args.run,
                    args.skip_autograder
                ): task_id
                for task_id in pending
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Summary for this model
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])

        logger.info(f"BATCH COMPLETE for {model_name}")
        logger.info(f"Success: {success_count}/{len(results)}")
        logger.info(f"Failed: {len(results) - success_count}/{len(results)}")
        logger.info(f"Time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per task)")

        all_results[model_name] = {'success': success_count, 'failed': len(results) - success_count, 'time': elapsed}

        # Show failures
        failures = [r for r in results if not r['success']]
        if failures:
            logger.error("Failed tasks:")
            for f in failures[:10]:
                logger.error(f"  - Task {f['task_id']}: {f.get('error', 'Unknown')}")
            if len(failures) > 10:
                logger.error(f"  ... and {len(failures) - 10} more")

    # Final summary for multiple models
    if len(models) > 1:
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY (All Models)")
        logger.info("=" * 60)
        for model_name, stats in all_results.items():
            total = stats['success'] + stats['failed']
            logger.info(f"  {model_name}: {stats['success']}/{total} succeeded ({stats['time']:.1f}s)")


if __name__ == '__main__':
    main()
