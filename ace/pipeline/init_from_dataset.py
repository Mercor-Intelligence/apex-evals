#!/usr/bin/env python3
"""
Initialize Supabase tables from CSV files in dataset/ folder

Usage:
    python3 init_from_dataset.py <domain> <model> [--overwrite] [--dry-run] [--supabase]

Examples:
    python3 init_from_dataset.py DIY gemini-2.5-pro          # Init DIY for one model
    python3 init_from_dataset.py DIY all                      # Init DIY for all models
    python3 init_from_dataset.py all gemini-2.5-pro          # Init all domains for one model
    python3 init_from_dataset.py all all                      # Init everything
    python3 init_from_dataset.py Food all --overwrite         # Overwrite existing Food data
    python3 init_from_dataset.py all all                       # Create local files only (default)
    python3 init_from_dataset.py all all --supabase            # Also write to database
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path FIRST (before importing from configs)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from configs.logging_config import setup_logging
from configs.domain_config import get_domain_config_for_model
from configs.model_providers import MODEL_REGISTRY, get_provider_for_model

logger = setup_logging(__name__)

# Supabase optional - disabled by default, use --supabase flag to enable
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

# Global Supabase client (initialized in main if needed)
supabase = None

# Available domains and models (models from single source of truth)
AVAILABLE_DOMAINS = ['Shopping', 'Food', 'Gaming', 'DIY']
AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())


def load_csv_data(csv_file, domain):
    """
    Load criteria data from CSV file

    Args:
        csv_file: Path to CSV file
        domain: Domain name (for column name detection)

    Returns:
        list: List of criteria rows as dicts
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    logger.info(f"Reading {csv_file}...")

    criteria = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            criteria.append(row)

    logger.info(f"Found {len(criteria)} criteria rows")
    return criteria


def insert_criteria_to_table(criteria, domain, model, overwrite=False, dry_run=False):
    """
    Insert criteria into Supabase criteria table

    Args:
        criteria: List of criteria dicts from CSV
        domain: Domain name
        model: Model name
        overwrite: If True, use upsert. If False, skip existing.
        dry_run: If True, don't actually insert

    Returns:
        int: Number of rows inserted
    """
    if supabase is None:
        logger.info("Skipping criteria table insert (local files only)")
        return 0

    config = get_domain_config_for_model(domain, model)
    criteria_table = config['criteria_table']

    logger.info(f"Inserting into {criteria_table}...")

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(criteria)} criteria")
        return len(criteria)

    inserted = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(criteria):
        try:
            # Debug: log first row to see column names
            if i == 0:
                logger.debug(f"CSV columns: {list(row.keys())}")

            criterion_id = int(row['Criterion ID'])

            # Check if exists (unless overwrite mode)
            if not overwrite:
                existing = supabase.table(criteria_table).select('"Criterion ID"').eq('"Criterion ID"', criterion_id).execute()
                if existing.data:
                    skipped += 1
                    continue

            # Prepare data (map CSV columns to DB columns)
            data = {
                "Criterion ID": criterion_id,
                "Task ID": int(row['Task ID']),
                "Prompt": row.get('Prompt', ''),
                "Description": row.get('Description', ''),
                "Criterion Grounding Check": row.get('Criterion Grounding Check', ''),
                "Hurdle Tag": row.get('Hurdle Tag', ''),
                f"Criterion Type ({domain})": row.get('Criteria type', ''),  # CSV has unified name, Supabase has domain-specific
                "Specified Prompt": row.get('Specified Prompt', ''),
                "Workflow": row.get('Workflow', '')
            }

            # Add Shop vs. Product for Shopping domain only
            if domain == 'Shopping':
                data["Shop vs. Product"] = row.get('Shop vs. Product', '')

            # Insert or update
            if overwrite:
                supabase.table(criteria_table).upsert(data, on_conflict='Criterion ID').execute()
            else:
                supabase.table(criteria_table).insert(data).execute()

            inserted += 1

            # Progress updates
            if inserted % 100 == 0:
                logger.debug(f"Progress: {inserted}/{len(criteria)} criteria inserted...")

        except Exception as e:
            errors += 1
            if errors <= 3:  # Only show first few errors
                logger.error(f"Error on criterion {row.get('Criterion ID', '?')}: {e}")

    logger.info(f"Inserted: {inserted}, Skipped: {skipped}, Errors: {errors}")
    return inserted


def create_test_case_json_files(criteria, domain, model, runs=[1,2,3,4,5,6,7,8]):
    """
    Create 0_test_case.json files for all tasks in all run directories

    Args:
        criteria: List of criteria dicts from CSV
        domain: Domain name
        model: Model name
        runs: List of run numbers to create directories for
    """
    # Determine provider from model (using centralized registry)
    provider = get_provider_for_model(model)

    # Group criteria by task ID
    if domain == 'Shopping':
        tasks_data = defaultdict(lambda: {'criteria': [], 'prompt': '', 'specified_prompt': '', 'workflow': '', 'shop_vs_product': ''})
    else:
        tasks_data = defaultdict(lambda: {'criteria': [], 'prompt': '', 'specified_prompt': '', 'workflow': ''})

    for row in criteria:
        task_id = int(row['Task ID'])
        criterion_id = int(row['Criterion ID'])

        # Set task-level data (same for all criteria in a task)
        if not tasks_data[task_id]['prompt']:
            tasks_data[task_id]['prompt'] = row.get('Specified Prompt') or row.get('Prompt', '')
            tasks_data[task_id]['specified_prompt'] = row.get('Specified Prompt', '')
            tasks_data[task_id]['workflow'] = row.get('Workflow', '')
            # Extract Shop vs. Product for Shopping domain only
            if domain == 'Shopping':
                tasks_data[task_id]['shop_vs_product'] = row.get('Shop vs. Product', '')

        # Add criterion
        tasks_data[task_id]['criteria'].append({
            'criterion_id': criterion_id,
            'id': len(tasks_data[task_id]['criteria']) + 1,
            'description': row.get('Description', ''),
            'type': row.get('Criteria type', ''),
            'hurdle_tag': row.get('Hurdle Tag', 'Not'),
            'grounded_status': row.get('Criterion Grounding Check', 'Grounded')
        })

    logger.info(f"Creating 0_test_case.json files for {len(tasks_data)} tasks...")

    # Get project root (one level up from this script's directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    created_count = 0
    for task_id, task_data in sorted(tasks_data.items()):
        for run_num in runs:
            # Create directory: results/{provider}/{model}/{domain}/run_{N}/task_{id}/
            results_dir = os.path.join(project_root, 'results', provider, model, domain, f'run_{run_num}', f'task_{task_id}')
            os.makedirs(results_dir, exist_ok=True)

            # Create test case JSON
            test_case = {
                'task_id': task_id,
                'test_id': f'task_{task_id}',
                'prompt': task_data['prompt'],
                'criteria': task_data['criteria'],
                'domain': domain
            }

            # Add shop_vs_product for Shopping domain only
            if domain == 'Shopping' and task_data.get('shop_vs_product'):
                test_case['shop_vs_product'] = task_data['shop_vs_product']

            test_case_file = os.path.join(results_dir, '0_test_case.json')
            with open(test_case_file, 'w', encoding='utf-8') as f:
                json.dump(test_case, f, indent=2, ensure_ascii=False)

            created_count += 1

    logger.info(f"Created {created_count} test case files ({len(tasks_data)} tasks x {len(runs)} runs)")
    return len(tasks_data)


def insert_tasks_to_table(criteria, domain, model, overwrite=False, dry_run=False):
    """
    Extract unique tasks from criteria and insert into task table

    Args:
        criteria: List of criteria dicts from CSV
        domain: Domain name
        model: Model name
        overwrite: If True, use upsert. If False, skip existing.
        dry_run: If True, don't actually insert

    Returns:
        int: Number of tasks inserted
    """
    if supabase is None:
        logger.info("Skipping task table insert (local files only)")
        return 0

    config = get_domain_config_for_model(domain, model)
    task_table = config['task_table']

    # Extract unique tasks
    tasks = {}
    for row in criteria:
        task_id = int(row['Task ID'])
        if task_id not in tasks:
            tasks[task_id] = {
                'prompt': row.get('Prompt', ''),
                'specified_prompt': row.get('Specified Prompt', ''),
                'workflow': row.get('Workflow', '')
            }
            # Add shop_vs_product only for Shopping domain
            if domain == 'Shopping':
                tasks[task_id]['shop_vs_product'] = row.get('Shop vs. Product', '')

    logger.info(f"Inserting into {task_table}...")
    logger.info(f"Found {len(tasks)} unique tasks")

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(tasks)} tasks")
        return len(tasks)

    inserted = 0
    skipped = 0
    errors = 0

    for task_id, task_data in sorted(tasks.items()):
        try:
            # Check if exists (unless overwrite mode)
            if not overwrite:
                existing = supabase.table(task_table).select('"Task ID"').eq('"Task ID"', task_id).execute()
                if existing.data:
                    skipped += 1
                    continue

            # Prepare data
            data = {
                "Task ID": task_id,
                "Prompt": task_data['prompt'],
                "Specified Prompt": task_data['specified_prompt'],
                "Workflow": task_data['workflow']
            }

            # Add Shop vs. Product for Shopping domain only
            if domain == 'Shopping' and task_data.get('shop_vs_product'):
                data["Shop vs. Product"] = task_data['shop_vs_product']

            # Insert or update
            if overwrite:
                supabase.table(task_table).upsert(data, on_conflict='Task ID').execute()
            else:
                supabase.table(task_table).insert(data).execute()

            inserted += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.error(f"Error on task {task_id}: {e}")

    logger.info(f"Inserted: {inserted}, Skipped: {skipped}, Errors: {errors}")
    return inserted


def initialize_domain_model(domain, model, overwrite=False, dry_run=False):
    """
    Initialize one domain-model combination

    Args:
        domain: Domain name (Shopping, Food, Gaming, DIY)
        model: Model name
        overwrite: Replace existing data
        dry_run: Preview without making changes
    """
    # Find CSV file - use path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_file = os.path.join(project_root, "dataset", f"ACE-{domain}.csv")

    if not os.path.exists(csv_file):
        logger.warning(f"CSV file not found: {csv_file}, skipping...")
        return False

    logger.info(f"{'='*60}")
    logger.info(f"Initializing {domain} - {model}")
    logger.info(f"{'='*60}")

    try:
        # Load CSV data
        criteria = load_csv_data(csv_file, domain)

        # Insert criteria
        criteria_count = insert_criteria_to_table(criteria, domain, model, overwrite, dry_run)

        # Insert tasks
        task_count = insert_tasks_to_table(criteria, domain, model, overwrite, dry_run)

        # Create local test case JSON files
        if not dry_run:
            json_count = create_test_case_json_files(criteria, domain, model)
        else:
            json_count = 0

        if supabase is None:
            logger.info(f"Completed: {json_count} test case files created (local files only)")
        else:
            logger.info(f"Completed: {criteria_count} criteria, {task_count} tasks, {json_count} test case files")
        return True

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Initialize database tables and/or local test case files from CSV datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize specific domain and model
    python3 init_from_dataset.py DIY gemini-2.5-pro

    # Initialize all models for one domain
    python3 init_from_dataset.py DIY all

    # Initialize one model for all domains
    python3 init_from_dataset.py all gemini-2.5-pro

    # Initialize everything
    python3 init_from_dataset.py all all

    # Create local files only (default)
    python3 init_from_dataset.py all all
    
    # Also write to Supabase database
    python3 init_from_dataset.py all all --supabase

    # Overwrite existing data
    python3 init_from_dataset.py Food all --overwrite

    # Dry run (preview without changes)
    python3 init_from_dataset.py all all --dry-run
        """
    )

    parser.add_argument('domain', help='Domain name (Shopping/Food/Gaming/DIY) or "all"')
    parser.add_argument('model', help='Model name or "all"')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing data')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--supabase', action='store_true', help='Also write to Supabase database (default: local files only)')

    args = parser.parse_args()

    # Initialize Supabase client only if --supabase flag is set
    global supabase
    if args.supabase:
        if not SUPABASE_AVAILABLE:
            logger.error("supabase package not installed. Install with: pip install supabase")
            sys.exit(1)

        SUPABASE_URL = os.getenv('SUPABASE_URL')
        SUPABASE_KEY = os.getenv('SUPABASE_KEY')

        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
            sys.exit(1)

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase enabled")

    # Determine domains and models to process
    if args.domain.lower() == 'all':
        domains = AVAILABLE_DOMAINS
    else:
        # Match domain case-insensitively (handles "diy" -> "DIY", "shopping" -> "Shopping")
        domain_match = next((d for d in AVAILABLE_DOMAINS if d.lower() == args.domain.lower()), None)
        if domain_match is None:
            logger.error(f"Unknown domain: {args.domain}")
            logger.error(f"Available: {', '.join(AVAILABLE_DOMAINS)}")
            sys.exit(1)
        domains = [domain_match]

    if args.model.lower() == 'all':
        models = AVAILABLE_MODELS
    else:
        if args.model not in AVAILABLE_MODELS:
            logger.error(f"Unknown model: {args.model}")
            logger.error(f"Available: {', '.join(AVAILABLE_MODELS)}")
            sys.exit(1)
        models = [args.model]

    # Print summary
    logger.info("="*60)
    logger.info("INITIALIZATION PLAN")
    logger.info("="*60)
    logger.info(f"Domains: {', '.join(domains)}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Total combinations: {len(domains)} x {len(models)} = {len(domains) * len(models)}")
    if args.supabase:
        logger.info(f"Mode: SUPABASE + LOCAL FILES ({'OVERWRITE' if args.overwrite else 'SKIP EXISTING'})")
    else:
        logger.info("Mode: LOCAL FILES ONLY (default)")
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    logger.info("="*60)

    # Process each combination
    successes = 0
    failures = 0

    for domain in domains:
        for model in models:
            success = initialize_domain_model(domain, model, args.overwrite, args.dry_run)
            if success:
                successes += 1
            else:
                failures += 1

    # Final summary
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Successful: {successes}/{len(domains) * len(models)}")
    logger.info(f"Failed: {failures}/{len(domains) * len(models)}")
    logger.info("="*60)

    if failures > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()