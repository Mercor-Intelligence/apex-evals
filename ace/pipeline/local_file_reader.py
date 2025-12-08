#!/usr/bin/env python3
"""
Local File Reader - Alternative to Supabase for file-based workflow

Reads test cases and results from local JSON files in the results/ directory.
Used by default when Supabase is not enabled via --supabase flag.
"""

import json
import os
import sys

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.model_providers import get_provider_for_model

# Project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_all_task_ids(domain, model_name, run_number=1):
    """
    Get all task IDs that have test case files in local filesystem

    Args:
        domain: Domain name ('Shopping', 'Food', 'Gaming', 'DIY')
        model_name: Model name (e.g., 'gemini-3-pro', 'opus-4.5')
        run_number: Run number (default: 1)

    Returns:
        list: Sorted list of task IDs that have 0_test_case.json files
    """
    provider_name = get_provider_for_model(model_name)
    run_dir = os.path.join(
        project_root, 'results', provider_name, model_name, domain, f'run_{run_number}'
    )

    task_ids = []
    if os.path.exists(run_dir):
        for item in os.listdir(run_dir):
            if item.startswith('task_'):
                test_case_file = os.path.join(run_dir, item, '0_test_case.json')
                if os.path.exists(test_case_file):
                    try:
                        task_id = int(item.replace('task_', ''))
                        task_ids.append(task_id)
                    except ValueError:
                        # Skip invalid directory names
                        continue

    return sorted(task_ids)


def get_test_case(task_id, domain, model_name, run_number=1):
    """
    Read test case from local JSON file

    Args:
        task_id: Task ID
        domain: Domain name
        model_name: Model name
        run_number: Run number (default: 1)

    Returns:
        dict: Test case data with keys: task_id, test_id, prompt, criteria, domain

    Raises:
        FileNotFoundError: If test case file doesn't exist
    """
    provider_name = get_provider_for_model(model_name)
    test_case_file = os.path.join(
        project_root, 'results', provider_name, model_name, domain,
        f'run_{run_number}', f'task_{task_id}', '0_test_case.json'
    )

    if not os.path.exists(test_case_file):
        raise FileNotFoundError(
            f"Test case file not found: {test_case_file}\n"
            f"Run 'python3 pipeline/init_from_dataset.py all {model_name}' first to create test case files."
        )

    with open(test_case_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def task_has_response(task_id, domain, model_name, run_number, check_complete=True):
    """
    Check if task has been processed

    Args:
        task_id: Task ID
        domain: Domain name
        model_name: Model name
        run_number: Run number
        check_complete: If True, check for final output (3_autograder_results.json)
                       If False, check for model response (1_grounded_response.json)

    Returns:
        bool: True if required file exists
    """
    provider_name = get_provider_for_model(model_name)
    results_dir = os.path.join(
        project_root, 'results', provider_name, model_name, domain,
        f'run_{run_number}', f'task_{task_id}'
    )

    if check_complete:
        # Check for final autograder output (most reliable indicator of completion)
        autograder_file = os.path.join(results_dir, '3_autograder_results.json')
        return os.path.exists(autograder_file)
    else:
        # Check for model response only (for partial resumption)
        response_file = os.path.join(results_dir, '1_grounded_response.json')
        return os.path.exists(response_file)


def get_pending_tasks(domain, model_name, run_number, force=False, skip_grading=False):
    """
    Get list of pending tasks from local filesystem

    Args:
        domain: Domain name
        model_name: Model name
        run_number: Run number
        force: If True, return all tasks regardless of completion status
        skip_grading: If True, check for 2_scraped_sources.json instead of 3_autograder_results.json

    Returns:
        list: Task IDs that need processing
    """
    all_task_ids = get_all_task_ids(domain, model_name, run_number)

    if force:
        return all_task_ids

    # Filter out tasks that have completed successfully
    pending = []
    provider_name = get_provider_for_model(model_name)

    for task_id in all_task_ids:
        results_dir = os.path.join(
            project_root, 'results', provider_name, model_name, domain,
            f'run_{run_number}', f'task_{task_id}'
        )

        # Check for final output to determine completion
        if skip_grading:
            # When skipping grading, check if scraping completed
            final_file = os.path.join(results_dir, '2_scraped_sources.json')
        else:
            # Normal mode: check if autograding completed
            final_file = os.path.join(results_dir, '3_autograder_results.json')

        if not os.path.exists(final_file):
            pending.append(task_id)

    return pending


def read_grounded_response(task_id, domain, model_name, run_number):
    """
    Read grounded response from local file (for regrading)

    Args:
        task_id: Task ID
        domain: Domain name
        model_name: Model name
        run_number: Run number

    Returns:
        tuple: (response_text, direct_grounding, test_case_data)

    Raises:
        FileNotFoundError: If required files don't exist
    """
    provider_name = get_provider_for_model(model_name)
    results_dir = os.path.join(
        project_root, 'results', provider_name, model_name, domain,
        f'run_{run_number}', f'task_{task_id}'
    )

    # Read test case
    test_case_file = os.path.join(results_dir, '0_test_case.json')
    if not os.path.exists(test_case_file):
        raise FileNotFoundError(f"Test case not found: {test_case_file}")

    with open(test_case_file, 'r', encoding='utf-8') as f:
        test_case_data = json.load(f)

    # Read grounded response
    response_file = os.path.join(results_dir, '1_grounded_response.json')
    if not os.path.exists(response_file):
        raise FileNotFoundError(
            f"Grounded response not found: {response_file}\n"
            f"Task {task_id} has not been processed yet."
        )

    with open(response_file, 'r', encoding='utf-8') as f:
        response_data = json.load(f)

    response_text = response_data.get('responseText', '')
    direct_grounding = response_data.get('direct_grounding', {})

    print(f"✅ Loaded from local files:")
    print(f"   - Response text: {len(response_text)} chars")
    print(f"   - Criteria: {len(test_case_data.get('criteria', []))}")
    print(f"   - Direct grounding: {'Yes' if direct_grounding else 'No'}")

    return response_text, direct_grounding, test_case_data


if __name__ == '__main__':
    """Test local file reader"""
    import argparse

    parser = argparse.ArgumentParser(description='Test local file reader')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--domain', required=True, help='Domain name')
    parser.add_argument('--run', type=int, default=1, help='Run number')

    args = parser.parse_args()

    print(f"Testing local file reader for {args.model} / {args.domain} / Run {args.run}")
    print("="*60)

    task_ids = get_all_task_ids(args.domain, args.model, args.run)
    print(f"Found {len(task_ids)} tasks: {task_ids}")

    if task_ids:
        # Test reading first task
        first_task = task_ids[0]
        print(f"\nReading task {first_task}...")
        test_case = get_test_case(first_task, args.domain, args.model, args.run)
        print(f"✅ Loaded test case:")
        print(f"   - Prompt: {test_case.get('prompt', '')[:100]}...")
        print(f"   - Criteria: {len(test_case.get('criteria', []))}")

        # Check completion status
        pending = get_pending_tasks(args.domain, args.model, args.run)
        print(f"\nPending tasks: {len(pending)}/{len(task_ids)}")

