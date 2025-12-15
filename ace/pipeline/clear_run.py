#!/usr/bin/env python3
"""
Clear run data from both Supabase tables AND local files
Supports all domains: Gaming, Shopping, Food
Supports all models: gemini-2.5-pro, gpt-5, gpt-4o, etc.
"""
import os
import shutil
import sys
import time

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.logging_config import setup_logging
logger = setup_logging(__name__)

# Make Supabase optional
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from configs.domain_config import get_domain_config_for_model
from configs.model_providers import get_provider_for_model
from configs.config import config

# Supabase client will be created in clear_supabase_data() after validation


def clear_supabase_data(domain, model_name, run_number):
    """Clear run data from Supabase tables in batches"""
    if not SUPABASE_AVAILABLE:
        raise RuntimeError("Supabase package not installed. Cannot clear database.")

    if not config.has_supabase():
        raise RuntimeError("Supabase credentials not configured. Cannot clear database.")

    config.validate_supabase()

    domain_config = get_domain_config_for_model(domain, model_name)
    task_table = domain_config['task_table']
    criteria_table = domain_config['criteria_table']

    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    logger.info('Clearing Supabase tables...')

    # Clear task_outputs table
    logger.info(f'Clearing {task_table}...')

    columns_to_clear = [
        f'"Response Text - {run_number}"',
        f'"Product Source Map - {run_number}"',
        f'"Grounding Source Meta Data - {run_number}"',
        f'"Direct Grounding - {run_number}"',
        f'"Score Overview - {run_number}"',
        f'"Scores - {run_number}"',
        f'"Total Score - {run_number}"',
        f'"Total Hurdle Score - {run_number}"',
        f'"Failed Grounded Sites - {run_number}"'
    ]

    # Get task IDs
    task_ids = []
    page_size = 1000
    offset = 0

    while True:
        result = supabase.table(task_table).select('"Task ID"').range(offset, offset + page_size - 1).execute()
        if not result.data:
            break
        task_ids.extend([row['Task ID'] for row in result.data])
        if len(result.data) < page_size:
            break
        offset += page_size

    logger.info(f'Found {len(task_ids)} tasks to clear')

    # Update in batches of 50
    BATCH_SIZE = 50
    total_cleared = 0

    for i in range(0, len(task_ids), BATCH_SIZE):
        batch = task_ids[i:i + BATCH_SIZE]

        # Clear all columns for this batch
        update_data = {col.strip('"'): None for col in columns_to_clear}

        try:
            result = supabase.table(task_table).update(update_data).in_('"Task ID"', batch).execute()
            total_cleared += len(result.data)
            logger.debug(f'Progress: {total_cleared}/{len(task_ids)} tasks cleared')
        except Exception as batch_error:
            logger.warning(f'Batch error on tasks {batch[0]}-{batch[-1]}: {batch_error}')

    logger.info(f'Cleared {total_cleared} task rows')

    # Clear criteria table
    logger.info(f'Clearing {criteria_table}...')

    columns_to_clear = [
        f'"Score - {run_number}"',
        f'"Reasoning - {run_number}"',
        f'"Failure Step - {run_number}"'
    ]

    # Get criterion IDs
    criterion_ids = []
    page_size = 1000
    offset = 0

    while True:
        result = supabase.table(criteria_table).select('"Criterion ID"').range(offset, offset + page_size - 1).execute()
        if not result.data:
            break
        criterion_ids.extend([row['Criterion ID'] for row in result.data])
        if len(result.data) < page_size:
            break
        offset += page_size

    logger.info(f'Found {len(criterion_ids)} criteria to clear')

    # Update in batches of 50
    BATCH_SIZE = 50
    total_cleared = 0

    for i in range(0, len(criterion_ids), BATCH_SIZE):
        batch = criterion_ids[i:i + BATCH_SIZE]

        update_data = {col.strip('"'): None for col in columns_to_clear}

        try:
            result = supabase.table(criteria_table).update(update_data).in_('"Criterion ID"', batch).execute()
            total_cleared += len(result.data)
            logger.debug(f'Progress: {total_cleared}/{len(criterion_ids)} criteria cleared')
        except Exception as batch_error:
            logger.warning(f'Batch error: {batch_error}')

    logger.info(f'Cleared {total_cleared} criteria rows')


def clear_local_files(domain, model_name, run_number):
    """Clear local result files for specific run"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Get provider from model name
    provider_name = get_provider_for_model(model_name)

    # Try multiple path patterns to support both old and new structures
    possible_paths = [
        # New structure: results/{provider}/{model}/{domain}/run_{run_number}
        os.path.join(project_root, 'results', provider_name, model_name, domain, f'run_{run_number}'),
        # Old structure (legacy gemini): results/{provider}/{domain}/run_{run_number}
        os.path.join(project_root, 'results', provider_name, domain, f'run_{run_number}'),
        # Google provider format: results/google/{model}/{domain}/run_{run_number}
        os.path.join(project_root, 'results', 'google', model_name, domain, f'run_{run_number}'),
    ]

    results_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            results_dir = path
            break

    if not results_dir:
        logger.info('No local files found. Tried:')
        for path in possible_paths:
            logger.info(f'  - {path}')
        return

    logger.info(f'Clearing local files from {results_dir}...')

    # Count task directories
    task_dirs = [d for d in os.listdir(results_dir) if d.startswith('task_') and os.path.isdir(os.path.join(results_dir, d))]

    if not task_dirs:
        logger.info('No task directories to clear')
        return

    # Files to delete (preserves 0_test_case.json)
    files_to_delete = [
        '1_grounded_response.json',
        '1_model_response.json',
        '2_scraped_sources.json',
        '3_autograder_results.json'
    ]

    total_files_deleted = 0
    for task_dir in task_dirs:
        task_path = os.path.join(results_dir, task_dir)
        for filename in files_to_delete:
            file_path = os.path.join(task_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                total_files_deleted += 1

    logger.info(f'Cleared {total_files_deleted} files from {len(task_dirs)} task directories (preserved 0_test_case.json)')


def main():
    """Main execution"""
    if len(sys.argv) < 3:
        logger.error("Missing arguments")
        logger.error("Usage: python clear_run.py <domain> <run_number> --model <model_name> [--supabase] [--yes]")
        logger.error("Examples:")
        logger.error("  python clear_run.py Shopping 1 --model gpt-5")
        logger.error("  python clear_run.py Gaming 2 --model gemini-2.5-pro --yes")
        logger.error("  python clear_run.py Food 1 --model gpt-5.1 --supabase")
        logger.error("Domains: Gaming, Shopping, Food, DIY")
        logger.error("Run numbers: 1, 2, 3, 4, 5, 6, 7, 8")
        logger.error("Models: gpt-5, gpt-5.1, o3, o3-pro, gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro, opus-4.5, etc.")
        logger.error("Flags:")
        logger.error("  --supabase: Also clear from database (default: local files only)")
        logger.error("  --yes, -y: Skip confirmation prompt")
        sys.exit(1)

    domain = sys.argv[1]

    if domain not in ['Gaming', 'Shopping', 'Food', 'DIY']:
        logger.error(f"Invalid domain '{domain}'")
        logger.error("Valid domains: Gaming, Shopping, Food, DIY")
        sys.exit(1)

    try:
        run_number = int(sys.argv[2])
        if run_number not in [1, 2, 3, 4, 5, 6, 7, 8]:
            raise ValueError()
    except ValueError:
        logger.error(f"Invalid run number '{sys.argv[2]}'")
        logger.error("Valid run numbers: 1, 2, 3, 4, 5, 6, 7, 8")
        sys.exit(1)

    # Check for model flag (required)
    if '--model' not in sys.argv:
        logger.error("--model flag is required")
        logger.error("Example: python clear_run.py Shopping 1 --model gpt-5")
        sys.exit(1)

    model_idx = sys.argv.index('--model')
    if model_idx + 1 >= len(sys.argv):
        logger.error("--model flag requires a model name")
        sys.exit(1)

    model_name = sys.argv[model_idx + 1]

    # Validate model name
    try:
        provider_name = get_provider_for_model(model_name)
        domain_config = get_domain_config_for_model(domain, model_name)
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("Valid models: gpt-5, gpt-5.1, o3, o3-pro, gemini-2.5-pro, gemini-2.5-flash, sonnet-4.5, opus-4.1, etc.")
        sys.exit(1)

    # Check for --yes flag to skip confirmation
    skip_confirmation = '--yes' in sys.argv or '-y' in sys.argv

    # Check for --supabase flag (default: local files only)
    use_supabase = '--supabase' in sys.argv and config.has_supabase() and SUPABASE_AVAILABLE

    # Confirm with user
    logger.info("CLEAR RUN DATA")
    logger.info(f"Model: {model_name}")
    logger.info(f"Provider: {provider_name}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Run Number: {run_number}")

    if use_supabase:
        logger.info(f"Tables: {domain_config['task_table']}, {domain_config['criteria_table']}")
    else:
        logger.info(f"Mode: Local files only (default)")

    logger.warning("This will:")
    if use_supabase:
        logger.warning(f"  1. Clear Run {run_number} data from Supabase tables (batch size: 50)")
        logger.warning(f"  2. Delete run files (preserves 0_test_case.json):")
    else:
        logger.warning(f"  1. Delete run files (preserves 0_test_case.json):")
    logger.warning(f"     Files: 1_grounded_response.json, 1_model_response.json, 2_scraped_sources.json, 3_autograder_results.json")
    logger.warning(f"     From: results/{provider_name}/{model_name}/{domain}/run_{run_number}/task_*/")
    logger.warning("This action cannot be undone!")

    if not skip_confirmation:
        response = input("\nContinue? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            logger.info("Cancelled")
            sys.exit(0)
    else:
        logger.warning("Auto-confirmed with --yes flag")

    logger.info("Starting cleanup...")

    # Step 1: Clear Supabase (only if available and not disabled)
    if use_supabase:
        try:
            clear_supabase_data(domain, model_name, run_number)
        except Exception as e:
            logger.error(f"Error clearing Supabase data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        logger.info('Skipping Supabase (local files only)...')

    # Step 2: Clear local files (always)
    try:
        clear_local_files(domain, model_name, run_number)
    except Exception as e:
        logger.error(f"Error clearing local files: {e}")
        sys.exit(1)

    logger.info(f"RUN {run_number} CLEARED SUCCESSFULLY!")
    logger.info(f"You can now run a fresh pipeline:")
    logger.info(f"  python3 pipeline/runner.py {domain} --model {model_name} --run {run_number} --workers 5")


if __name__ == '__main__':
    main()

