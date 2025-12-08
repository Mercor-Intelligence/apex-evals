#!/usr/bin/env python3
"""
Initialize task_outputs table with Task IDs and Prompts from criteria table
This prepares the table for batch processing.
"""

import os
import sys
from collections import defaultdict

# Add project root to path FIRST (2 levels up from tools/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from supabase import create_client, Client

from configs.logging_config import setup_logging
from configs.domain_config import get_domain_config_for_model

logger = setup_logging(__name__)

# Load configuration
from configs.config import config

# Supabase credentials
config.validate_supabase()

# Initialize Supabase client
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def get_tasks_from_criteria(domain='Gaming', model_name='gemini-2.5-flash'):
    """
    Get unique task IDs and prompts from criteria table

    Args:
        domain: Domain name
        model_name: Model name (e.g. 'gemini-2.5-flash', 'gpt-5')

    Returns:
        dict: {task_id: prompt}
    """
    config = get_domain_config_for_model(domain, model_name)
    criteria_table = config['criteria_table']

    logger.info(f"📖 Reading tasks from {criteria_table}...")

    try:
        # Query all rows with pagination (handle tables > 1000 rows)
        all_rows = []
        page_size = 1000
        offset = 0

        while True:
            result = supabase.table(criteria_table).select(
                '"Task ID", "Prompt"'
            ).order('"Task ID"').range(offset, offset + page_size - 1).execute()

            page_rows = result.data

            if not page_rows:
                break

            all_rows.extend(page_rows)

            if len(page_rows) < page_size:
                # Last page
                break

            offset += page_size
            print(f"   Fetched {offset} rows...")

        print(f"   Total rows fetched: {len(all_rows)}")

        # Group by task ID (each task has multiple criteria rows)
        tasks = {}
        null_prompt_count = 0

        for row in all_rows:
            task_id = row['Task ID']
            prompt = row.get('Prompt')

            if task_id not in tasks:
                # Handle NULL prompts
                if prompt is None or prompt.strip() == '':
                    null_prompt_count += 1
                    # Skip tasks with no prompt
                    continue
                tasks[task_id] = prompt.strip()

        print(f"✅ Found {len(tasks)} unique tasks with prompts")
        if null_prompt_count > 0:
            print(f"⚠️  Skipped {null_prompt_count} tasks with NULL/empty prompts\n")
        else:
            print()

        return tasks

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def initialize_task_outputs(tasks, domain='Gaming', model_name='gemini-2.5-flash', overwrite=False):
    """
    Initialize task_outputs table with Task IDs and Prompts

    Args:
        tasks: dict of {task_id: prompt}
        domain: Domain name
        model_name: Model name (e.g. 'gemini-2.5-flash', 'gpt-5')
        overwrite: If True, overwrite existing rows. If False, only insert new ones.

    Returns:
        int: Number of tasks initialized
    """
    config = get_domain_config_for_model(domain, model_name)
    task_table = config['task_table']

    print(f"Initializing {task_table}...")
    print(f"   Mode: {'OVERWRITE' if overwrite else 'INSERT NEW ONLY'}\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for task_id, prompt in sorted(tasks.items()):
        try:
            # Check if task already exists
            if not overwrite:
                existing = supabase.table(task_table).select('"Task ID"').eq('"Task ID"', task_id).execute()
                if existing.data:
                    skip_count += 1
                    if skip_count <= 5:  # Only print first few
                        print(f"  [skip] Task {task_id}: Already exists, skipping")
                    elif skip_count == 6:
                        print(f"  [skip] ... (skipping remaining existing tasks)")
                    continue

            # Prepare data
            data = {
                "Task ID": task_id,
                "Prompt": prompt
            }

            # Upsert (insert or update)
            result = supabase.table(task_table).upsert(data, on_conflict='Task ID').execute()

            success_count += 1
            if success_count <= 5 or success_count % 50 == 0:
                print(f"  ✅ Task {task_id}: Initialized")

        except Exception as e:
            error_count += 1
            print(f"  ❌ Task {task_id}: Error - {e}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"   ✅ Initialized: {success_count}")
    print(f"   Skipped: {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   Total: {len(tasks)}")
    print(f"{'='*60}\n")

    return success_count


def get_criteria_for_task(task_id, domain='Gaming', model_name='gemini-2.5-flash'):
    """
    Get all criteria for a specific task

    Args:
        task_id: Task ID
        domain: Domain name
        model_name: Model name (e.g. 'gemini-2.5-flash', 'gpt-5')

    Returns:
        list: List of criteria dicts
    """
    config = get_domain_config_for_model(domain, model_name)
    criteria_table = config['criteria_table']
    criterion_type_column = config['criterion_type_column']

    try:
        result = supabase.table(criteria_table).select('*').eq('"Task ID"', task_id).order('"Criterion ID"').execute()

        criteria = []
        for i, row in enumerate(result.data):
            criterion_type_raw = row.get(criterion_type_column)

            # Handle NULL criterion type (some tasks have this)
            if criterion_type_raw is None:
                criterion_type = 'Unknown'
            else:
                criterion_type = criterion_type_raw.strip()

            hurdle_tag = row.get('Hurdle Tag', 'Not')

            criteria.append({
                'criterion_id': int(row['Criterion ID']),
                'id': len(criteria) + 1,  # Use current count for proper indexing
                'description': row['Description'].strip(),
                'type': criterion_type,
                'hurdle_tag': hurdle_tag if hurdle_tag else 'Not',
                'grounded_status': row['Criterion Grounding Check']  # Read from database column
            })

        return criteria

    except Exception as e:
        print(f"❌ Error getting criteria for task {task_id}: {e}")
        return []


def save_criteria_to_table(task_id, criteria, prompt, domain='Gaming', model_name='gemini-2.5-flash'):
    """
    Save criteria list to task_outputs table

    Args:
        task_id: Task ID
        criteria: List of criteria dicts
        prompt: Prompt text (needed for tasks that don't exist yet)
        domain: Domain name
        model_name: Model name (e.g. 'gemini-2.5-flash', 'gpt-5')
    """
    config = get_domain_config_for_model(domain, model_name)
    task_table = config['task_table']

    try:
        # Include both Prompt and Criteria List to handle new inserts
        data = {
            "Task ID": task_id,
            "Prompt": prompt,
            "Criteria List": criteria
        }

        result = supabase.table(task_table).upsert(data, on_conflict='Task ID').execute()
        return True

    except Exception as e:
        print(f"❌ Error saving criteria for task {task_id}: {e}")
        return False


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize task_outputs table with Task IDs and Prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize Gaming tasks (Gemini)
  python initialize_tasks.py Gaming

  # Initialize with overwrite (re-initialize existing tasks)
  python initialize_tasks.py Gaming --overwrite

  # Initialize and save criteria too
  python initialize_tasks.py Gaming --with-criteria

  # Different provider
  python initialize_tasks.py Shopping --provider openai
        """
    )

    parser.add_argument(
        'domain',
        choices=['Shopping', 'Gaming', 'Food'],
        help='Domain to initialize'
    )
    parser.add_argument(
        '--provider',
        default='gemini',
        choices=['gemini', 'openai'],
        help='Model provider (default: gemini)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing tasks'
    )
    parser.add_argument(
        '--with-criteria',
        action='store_true',
        help='Also save criteria list for each task'
    )

    args = parser.parse_args()

    print("="*60)
    print(f"🚀 INITIALIZE TASK OUTPUTS")
    print(f"   Domain: {args.domain}")
    print(f"   Provider: {args.provider.upper()}")
    print("="*60 + "\n")

    # Get tasks from criteria table
    tasks = get_tasks_from_criteria(args.domain, args.provider)

    if not tasks:
        print("❌ No tasks found!")
        sys.exit(1)

    # Initialize task_outputs table
    success = initialize_task_outputs(tasks, args.domain, args.provider, args.overwrite)

    # Optionally save criteria too
    if args.with_criteria and success > 0:
        print("\n📋 Adding criteria lists to tasks...")
        criteria_success = 0

        for task_id in sorted(tasks.keys()):
            prompt = tasks[task_id]
            criteria = get_criteria_for_task(task_id, args.domain, args.provider)
            if criteria:
                if save_criteria_to_table(task_id, criteria, prompt, args.domain, args.provider):
                    criteria_success += 1
                    if criteria_success <= 5 or criteria_success % 50 == 0:
                        print(f"  ✅ Task {task_id}: Added {len(criteria)} criteria")

        print(f"\n✅ Added criteria to {criteria_success}/{len(tasks)} tasks\n")

    print("✅ Initialization complete!")


if __name__ == '__main__':
    main()

