#!/usr/bin/env python3
"""
Clear All Runs Script
Clears data for all 8 runs sequentially for a given model and domain
Calls clear_run.py for each run with --yes flag to skip confirmations
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.logging_config import setup_logging
logger = setup_logging(__name__)


def clear_single_run(domain, model, run_number, use_supabase=False):
    """
    Clear a single run using clear_run.py

    Args:
        domain: Domain name (Shopping, Gaming, Food, DIY)
        model: Model name (gemini-2.5-pro, etc.)
        run_number: Run number (1-8)
        use_supabase: If True, also clear from Supabase (default: local files only)

    Returns:
        dict: {'success': bool, 'run_number': int, 'error': str}
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clear_script = os.path.join(script_dir, 'clear_run.py')

    cmd = [
        'python3',
        clear_script,
        domain,
        str(run_number),
        '--model', model,
        '--yes'  # Skip confirmation
    ]

    # Add --supabase flag if requested
    if use_supabase:
        cmd.append('--supabase')

    logger.info(f"\n{'='*70}")
    logger.info(f"Clearing Run {run_number}...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,  # stdout flows to terminal
            text=True,
            timeout=600  # 10 minutes max per run (generous for large datasets)
        )

        if result.returncode == 0:
            return {
                'success': True,
                'run_number': run_number
            }
        else:
            return {
                'success': False,
                'run_number': run_number,
                'error': f'Exit code {result.returncode}: {result.stderr[:200] if result.stderr else "Unknown"}'
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'run_number': run_number,
            'error': 'Timeout after 10 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'run_number': run_number,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Clear all 8 runs sequentially for a given model and domain'
    )
    parser.add_argument(
        'domain',
        choices=['Shopping', 'Gaming', 'Food', 'DIY'],
        help='Domain to clear'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model name (e.g., gemini-2.5-pro, gemini-2.5-flash, gpt-5)'
    )
    parser.add_argument(
        '--start-run',
        type=int,
        default=1,
        choices=range(1, 9),
        help='Start from this run number (default: 1)'
    )
    parser.add_argument(
        '--end-run',
        type=int,
        default=8,
        choices=range(1, 9),
        help='End at this run number (default: 8)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Require confirmation before starting (safety check)'
    )
    parser.add_argument(
        '--supabase',
        action='store_true',
        help='Also clear from Supabase (default: local files only)'
    )

    args = parser.parse_args()

    # Validate run range
    if args.start_run > args.end_run:
        print(f"❌ Error: start-run ({args.start_run}) must be <= end-run ({args.end_run})")
        sys.exit(1)

    # Print header
    print("\n" + "="*70)
    print("CLEAR ALL RUNS")
    print("="*70)
    print(f"Domain:       {args.domain}")
    print(f"Model:        {args.model}")
    print(f"Runs:         {args.start_run} to {args.end_run}")
    print(f"Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n⚠️  WARNING: This will clear ALL data for the specified runs!")
    print("This includes:")
    if args.supabase:
        print("  - Supabase table data (response text, scores, grounding metadata)")
    print("  - Local result files (all task directories)")
    if not args.supabase:
        print("\nMode: Local files only (default)")
    print("\n❗ This action CANNOT be undone!")

    # Confirmation check
    if args.confirm:
        print("\n" + "="*70)
        response = input("Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("❌ Cancelled - confirmation failed")
            sys.exit(0)
        print("="*70)

    overall_start = datetime.now()
    results = []

    # Clear each run sequentially
    for run_num in range(args.start_run, args.end_run + 1):
        result = clear_single_run(
            domain=args.domain,
            model=args.model,
            run_number=run_num,
            use_supabase=args.supabase
        )
        results.append(result)

        if result['success']:
            print(f"\n✅ Run {run_num} cleared successfully")
        else:
            print(f"\n❌ Run {run_num} failed: {result.get('error', 'Unknown error')}")
            # Continue to next run even if this one failed

    # Print final summary
    overall_end = datetime.now()
    elapsed = (overall_end - overall_start).total_seconds()

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Domain:   {args.domain}")
    print(f"Model:    {args.model}")
    print(f"Runs:     {args.start_run} to {args.end_run}")
    print("="*70)

    successful_clears = 0

    for result in results:
        run_num = result['run_number']
        status = "✅" if result['success'] else "❌"
        error_msg = f" ({result.get('error', '')})" if not result['success'] else ""
        print(f"{status} Run {run_num}{error_msg}")

        if result['success']:
            successful_clears += 1

    print("="*70)
    print(f"Successful:  {successful_clears}/{len(results)}")
    print(f"Failed:      {len(results) - successful_clears}/{len(results)}")
    print(f"Total Time:  {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Completed:   {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if successful_clears == len(results):
        print("\n✅ All runs cleared successfully!")
        print(f"\nYou can now run a fresh pipeline:")
        print(f"  python3 pipeline/runner.py {args.domain} --model {args.model}")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(results) - successful_clears} run(s) failed to clear")
        sys.exit(1)


if __name__ == '__main__':
    main()

