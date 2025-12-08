#!/usr/bin/env python3
"""
Supabase Test Case Reader
Reads test cases from Supabase "ACE TEST" table
"""

import json
import os
import sys
from collections import defaultdict

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

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

# Import from configs
from configs.config import config

# Supabase (optional)
if config.has_supabase() and SUPABASE_AVAILABLE:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
else:
    supabase = None


def execute_supabase_query(query):
    """Execute SQL query via Supabase RPC"""
    try:
        # Use rpc to execute raw SQL
        result = supabase.rpc('exec_sql', {'query': query}).execute()
        return result.data

    except Exception as e:
        logger.error(f"Error executing Supabase query: {e}")
        return None


def parse_supabase_table(table_name='ACE TEST', exclude_types=None, criterion_type_column='Criterion Type (Shopping)'):
    """
    Parse Supabase table and group criteria by task ID

    Args:
        table_name: Name of the ACE table
        exclude_types: List of criterion types to exclude (e.g., ['Geographic availability'])
        criterion_type_column: Name of the criterion type column (domain-specific)

    Returns:
        dict: {task_id: {prompt, criteria}}
    """
    if exclude_types is None:
        exclude_types = []

    try:
        # Query ALL rows from the table (paginate if needed for tables > 1000 rows)
        all_rows = []
        page_size = 1000
        offset = 0

        while True:
            response = supabase.table(table_name).select('*').order('Task ID').order('Criterion ID').range(offset, offset + page_size - 1).execute()
            page_rows = response.data

            if not page_rows:
                break

            all_rows.extend(page_rows)

            if len(page_rows) < page_size:
                # Last page
                break

            offset += page_size

        rows = all_rows
        print(f"🔍 Debug: Retrieved {len(rows) if rows else 0} rows from {table_name}")

        if not rows:
            print(f"⚠️  No data found in {table_name} table")
            print(f"   This might be due to Row Level Security (RLS) policies.")
            print(f"   Check that the anon key has SELECT permissions on this table.")
            return {}

        test_cases = defaultdict(lambda: {'prompt': '', 'criteria': []})

        for row in rows:
            task_id = int(row['Task ID'])
            criterion_id = int(row['Criterion ID'])
            criterion_type_raw = row.get(criterion_type_column)

            # Skip rows with NULL criterion type
            if criterion_type_raw is None:
                continue

            criterion_type = criterion_type_raw.strip()

            # Skip excluded criterion types
            if criterion_type in exclude_types:
                continue

            # Set prompt (same for all criteria in a task)
            # Use Specified Prompt if available, fallback to Prompt
            if not test_cases[task_id]['prompt']:
                test_cases[task_id]['prompt'] = (row.get('Specified Prompt') or row['Prompt']).strip()

            # Add criterion with hurdle tag
            hurdle_tag = row.get('Hurdle Tag', 'Not')  # Default to 'Not' if not present
            test_cases[task_id]['criteria'].append({
                'criterion_id': criterion_id,
                'id': len(test_cases[task_id]['criteria']) + 1,
                'description': row['Description'].strip(),
                'type': criterion_type,
                'hurdle_tag': hurdle_tag if hurdle_tag else 'Not',  # Include hurdle tag
                'grounded_status': row['Criterion Grounding Check']  # Read from database column
            })

        return dict(test_cases)

    except Exception as e:
        print(f"❌ Error reading from Supabase table: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_test_case(task_id, table_name=None, exclude_types=None, domain='Shopping', model_name=None, run_number=1):
    """
    Get a specific test case by task ID

    Reads from Supabase if available, otherwise from local files.

    Args:
        task_id: Task ID to retrieve
        domain: Domain name ('Shopping', 'Food', 'Gaming', 'DIY') - default: 'Shopping'
        table_name: Override domain's default ACE table
        exclude_types: Override domain's default exclusions
        model_name: Model name (required if table_name not provided, e.g. 'gemini-2.5-flash')
        run_number: Run number for local file lookup (default: 1)

    Returns:
        dict: {task_id, test_id, prompt, criteria, domain}
    """
    # If no Supabase, read from local file
    if supabase is None:
        if model_name is None:
            raise ValueError("model_name is required when Supabase is not available")
        from local_file_reader import get_test_case as get_local_test_case
        return get_local_test_case(task_id, domain, model_name, run_number)

    # Supabase path
    from configs.domain_config import get_domain_config_for_model, DOMAIN_BASE_CONFIG

    # Get config - if model_name is provided, use it; otherwise infer from domain base config
    if model_name:
        config = get_domain_config_for_model(domain, model_name)
    else:
        # Fallback to base config (for backward compatibility)
        config = DOMAIN_BASE_CONFIG[domain].copy()
        # If table_name not provided and no model_name, we can't proceed
        if table_name is None:
            raise ValueError("Either table_name or model_name must be provided")

    # Use domain defaults if not overridden
    if table_name is None:
        table_name = config['criteria_table']
    if exclude_types is None:
        exclude_types = config.get('exclude_types', [])

    criterion_type_column = config['criterion_type_column']
    description_column = config.get('description_column', 'Description')  # All tables use "Description"

    # OPTIMIZED: Query only rows for this specific task (not entire table!)
    try:
        response = supabase.table(table_name).select('*').eq('"Task ID"', task_id).execute()
        rows = response.data

        if not rows:
            raise ValueError(f"Task ID {task_id} not found in Supabase {table_name} table")

        print(f"🔍 Debug: Retrieved {len(rows)} rows for Task {task_id} from {table_name}")

        # Extract prompt (use Specified Prompt if available, fallback to Prompt)
        prompt = rows[0].get('Specified Prompt') or rows[0]['Prompt']
        prompt = prompt.strip()

        # Build criteria list
        criteria = []
        for row in rows:
            criterion_type_raw = row.get(criterion_type_column)

            # Skip rows with NULL criterion type
            if criterion_type_raw is None:
                continue

            criterion_type = criterion_type_raw.strip()

            # Skip excluded criterion types
            if criterion_type in exclude_types:
                continue

            # Add criterion with hurdle tag
            hurdle_tag = row.get('Hurdle Tag', 'Not')
            criteria.append({
                'criterion_id': int(row['Criterion ID']),
                'id': len(criteria) + 1,
                'description': row[description_column].strip(),  # Use domain-specific column name
                'type': criterion_type,
                'hurdle_tag': hurdle_tag,
                'grounded_status': row['Criterion Grounding Check']  # Read from database column
            })

        return {
            'task_id': task_id,
            'test_id': f"task_{task_id}",
            'prompt': prompt,
            'criteria': criteria,
            'domain': domain
        }

    except Exception as e:
        print(f"❌ Error reading from Supabase table: {e}")
        raise ValueError(f"Task ID {task_id} not found in Supabase {table_name} table")


def get_task_ids(table_name='ACE TEST', limit=None, domain='Shopping', model_name=None):
    """
    Get list of all task IDs from Supabase table

    Args:
        table_name: Name of the table to query
        limit: Optional limit on number of task IDs to return
        domain: Domain name ('Shopping', 'Gaming', 'Food')
        model_name: Model name (optional, e.g. 'gemini-2.5-flash')

    Returns:
        List of task IDs
    """
    from configs.domain_config import get_domain_config_for_model, DOMAIN_BASE_CONFIG

    # Get config - if model_name is provided, use it; otherwise use base config
    if model_name:
        config = get_domain_config_for_model(domain, model_name)
    else:
        config = DOMAIN_BASE_CONFIG[domain].copy()

    criterion_type_column = config['criterion_type_column']
    exclude_types = config.get('exclude_types', [])

    all_cases = parse_supabase_table(table_name, exclude_types=exclude_types, criterion_type_column=criterion_type_column)
    task_ids = sorted(all_cases.keys())

    if limit:
        task_ids = task_ids[:limit]

    return task_ids


def list_tasks(table_name='ACE TEST'):
    """List all available task IDs in Supabase table"""
    all_cases = parse_supabase_table(table_name)

    print(f"Available tasks in Supabase '{table_name}' table:\n")
    for task_id in sorted(all_cases.keys()):
        num_criteria = len(all_cases[task_id]['criteria'])
        prompt_preview = all_cases[task_id]['prompt'][:80] + "..."
        print(f"  Task {task_id}: {num_criteria} criteria")
        print(f"    {prompt_preview}\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python supabase_reader.py <task-id>")
        print("       python supabase_reader.py --list")
        sys.exit(1)

    if sys.argv[1] == '--list':
        list_tasks()
    else:
        task_id = int(sys.argv[1])
        test_case = get_test_case(task_id)
        print(json.dumps(test_case, indent=2))

