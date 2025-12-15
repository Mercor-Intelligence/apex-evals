#!/usr/bin/env python3
"""
Run runner.py for all 10 models in parallel
Simple wrapper - just runs 10 processes at once
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.logging_config import setup_logging
from configs.model_providers import MODEL_REGISTRY

logger = setup_logging(__name__)

# All models to evaluate (from single source of truth in model_providers.py)
# Note: gemini-3-pro requires google-genai >= 1.48.0 (has thinking_level parameter)
MODELS = list(MODEL_REGISTRY.keys())


def run_single_model(domain, model, run_number, workers, runner_flags):
    """Run runner.py for one model with live output"""
    import re
    from configs.model_providers import get_provider_for_model
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    runner_path = os.path.join(script_dir, 'runner.py')
    cmd = [
        sys.executable, runner_path, domain,
        '--model', model,
        '--run', str(run_number),
        '--workers', str(workers)
    ]
    if runner_flags:
        cmd += runner_flags

    logger.info(f"Starting {model}...")
    start = time.time()

    # Let stdout flow through LIVE to terminal, only capture stderr
    # Set LOGLEVEL=INFO to ensure logging output shows
    env = os.environ.copy()
    env['LOGLEVEL'] = 'INFO'
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, env=env)
    elapsed = time.time() - start
    
    # Count completed tasks by checking for 3_autograder_results.json files
    provider_name = get_provider_for_model(model)
    results_dir = os.path.join(project_root, 'results', provider_name, model, domain, f'run_{run_number}')
    
    success_count = 0
    total_count = 0
    
    if os.path.exists(results_dir):
        task_dirs = [d for d in os.listdir(results_dir) if d.startswith('task_')]
        total_count = len(task_dirs)
        for task_dir in task_dirs:
            autograder_file = os.path.join(results_dir, task_dir, '3_autograder_results.json')
            if os.path.exists(autograder_file):
                success_count += 1
    
    # Model is successful if subprocess succeeded and (tasks completed or no tasks to run)
    task_success = (success_count > 0) or (total_count == 0)

    if result.returncode == 0 and task_success:
        logger.info(f"{model} done ({elapsed/60:.1f} min) - {success_count}/{total_count} tasks")
        return {'model': model, 'success': True, 'time': elapsed, 
                'tasks_success': success_count, 'tasks_total': total_count}
    else:
        if result.returncode != 0:
            logger.error(f"{model} crashed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
        else:
            logger.error(f"{model} failed: {success_count}/{total_count} tasks succeeded")
        return {'model': model, 'success': False, 'time': elapsed,
                'tasks_success': success_count, 'tasks_total': total_count}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run runner.py for every model in parallel',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('domain', help='Domain to evaluate (Shopping, Food, Gaming, DIY)')
    parser.add_argument('run_number', type=int, choices=range(1, 9), metavar='run',
                        help='Run number (1-8)')
    parser.add_argument('--workers', type=int, default=100, help='Workers per model')
    parser.add_argument('--runner-flags', nargs=argparse.REMAINDER, default=[],
                        help='Additional flags to pass to each runner.py invocation (e.g., --supabase)')

    args = parser.parse_args()

    domain = args.domain
    run_number = args.run_number
    workers = args.workers
    runner_flags = args.runner_flags

    logger.info(f"Running all {len(MODELS)} models in parallel")
    logger.info(f"Domain: {domain}, Run: {run_number}, Workers/model: {workers}")
    if runner_flags:
        logger.info(f"Runner extra flags: {' '.join(runner_flags)}")

    with ProcessPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = [
            executor.submit(run_single_model, domain, model, run_number, workers, runner_flags)
            for model in MODELS
        ]
        results = [f.result() for f in as_completed(futures)]

    # Aggregate counts
    models_success = sum(1 for r in results if r['success'])
    total_tasks_success = sum(r.get('tasks_success', 0) for r in results)
    total_tasks = sum(r.get('tasks_total', 0) for r in results)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Models: {models_success}/{len(MODELS)} succeeded")
    print(f"Tasks:  {total_tasks_success}/{total_tasks} succeeded")
    print("="*60)

    for r in results:
        tasks_info = f"({r.get('tasks_success', 0)}/{r.get('tasks_total', 0)} tasks)"
        if r['success']:
            logger.info(f"✅ {r['model']} {tasks_info}")
        else:
            logger.error(f"❌ {r['model']} {tasks_info}")
