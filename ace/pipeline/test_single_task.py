#!/usr/bin/env python3
"""
Run single task on a model (Test Script)
Directly calls runner.py (default: local files only, use --supabase to enable database)
"""

import sys
import os
import argparse

# Add project root to path (for configs imports in runner.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import runner
from runner import process_single_task_with_original_scripts

def main():
    parser = argparse.ArgumentParser(
        description='Test a single task with a specific model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_single_task.py 715 gemini-2.5-pro 1 Shopping
  python3 test_single_task.py 312 sonnet-4.5 1 Food
  python3 test_single_task.py 100 gpt-5 2 Gaming --supabase
        """
    )

    parser.add_argument('task_id', type=int, help='Task ID from dataset')
    parser.add_argument('model', help='Model name (e.g., gemini-2.5-pro, gpt-5, sonnet-4.5)')
    parser.add_argument('run', type=int, choices=range(1, 9), metavar='run', help='Run number (1-8)')
    parser.add_argument('domain', nargs='?', default='Shopping',
                       choices=['Shopping', 'Food', 'Gaming', 'DIY'],
                       help='Domain name (default: Shopping)')
    parser.add_argument('--supabase', action='store_true',
                       help='Enable Supabase (default: local files only)')
    parser.add_argument('--skip-autograder', action='store_true',
                       help='Skip autograder (only run grounding + scraping)')

    args = parser.parse_args()

    # Enable Supabase if flag is set
    if args.supabase:
        from configs.config import config
        from configs.domain_config import get_domain_config_for_model
        try:
            from supabase import create_client
            if config.has_supabase():
                runner.USE_SUPABASE = True
                runner.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                print("--supabase flag: Enabling Supabase")
                
                # Validate Supabase connection and data
                try:
                    domain_config = get_domain_config_for_model(args.domain, args.model)
                    criteria_table = domain_config['criteria_table']
                    
                    # Test connection by querying for this specific task
                    test_result = runner.supabase.table(criteria_table).select('"Task ID"').eq('"Task ID"', args.task_id).execute()
                    
                    if not test_result.data:
                        print(f"⚠️  Task {args.task_id} not found in Supabase table '{criteria_table}'")
                        print(f"   Run 'python3 pipeline/init_from_dataset.py {args.domain} {args.model} --supabase' first")
                        print(f"   Or remove --supabase flag to use local files")
                        sys.exit(1)
                    
                    print(f"   Supabase verified (task {args.task_id} exists in {criteria_table})")
                except Exception as e:
                    print(f"⚠️  Supabase validation failed: {e}")
                    print("   Check your SUPABASE_URL and SUPABASE_KEY in .env")
                    sys.exit(1)
            else:
                print("⚠️  --supabase flag set but credentials not available")
                sys.exit(1)
        except ImportError:
            print("⚠️  --supabase flag set but supabase package not installed")
            sys.exit(1)

    print(f"[*] Testing Task {args.task_id}")
    print(f"Model: {args.model}")
    print(f"Run: {args.run}")
    print(f"Domain: {args.domain}")
    if not args.supabase:
        print(f"Mode: Local files only (default)")

    result = process_single_task_with_original_scripts(
        args.task_id,
        args.domain,
        args.model,
        args.run,
        skip_grading=args.skip_autograder
    )

    if result['success']:
        print("\n✅ SUCCESS!")
        if not args.skip_autograder:
            print(f"Results saved in: results/{args.domain}/run_{args.run}/task_{args.task_id}/")
    else:
        print("\n❌ FAILED!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        if result.get('stage'):
            print(f"Failed at stage: {result['stage']}")
        sys.exit(1)

if __name__ == "__main__":
    main()

