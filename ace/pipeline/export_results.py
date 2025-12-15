#!/usr/bin/env python3
"""
Export task results to properly formatted CSV files.
Fixes:
1. Proper CSV quoting and escaping
2. Workflow values from dataset files
3. Clean formatting that CSV editors can recognize
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from configs.config import config
from configs.model_providers import MODEL_REGISTRY, get_provider_for_model

# Constants
ALL_MODELS = list(MODEL_REGISTRY.keys())
ALL_DOMAINS = ['Shopping', 'Gaming', 'Food', 'DIY']
RESULTS_DIR = os.path.join(project_root, 'results')
DATASET_DIR = os.path.join(project_root, 'dataset')
OUTPUT_DIR = os.path.join(project_root, 'exported_results')


def load_workflows_from_dataset(domain):
    """
    Load workflow values from dataset CSV file.
    
    Args:
        domain: Domain name
        
    Returns:
        dict: {task_id: workflow_value}
    """
    dataset_file = os.path.join(DATASET_DIR, f'ACE-{domain}.csv')
    workflows = {}
    
    if not os.path.exists(dataset_file):
        print(f"  [!] Dataset file not found: {dataset_file}")
        return workflows
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get('Task ID')
                workflow = row.get('Workflow', '')
                if task_id:
                    workflows[task_id] = workflow
        
        # Get unique workflows (multiple rows per task, one per criterion)
        unique_workflows = {}
        for task_id, workflow in workflows.items():
            if task_id not in unique_workflows:
                unique_workflows[task_id] = workflow
        
        return unique_workflows
    
    except Exception as e:
        print(f"  [!] Error loading workflows from {dataset_file}: {e}")
        return {}


def normalize_quotes(text):
    """
    Replace smart/curly quotes with straight quotes for CSV compatibility.
    Some CSV viewers get confused by smart quotes.
    """
    if not isinstance(text, str):
        return text
    
    # Replace smart double quotes with straight quotes
    text = text.replace('"', '"').replace('"', '"')
    # Replace smart single quotes with straight apostrophes
    text = text.replace(''', "'").replace(''', "'")
    # Replace other problematic quotes
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('‹', "'").replace('›', "'")
    
    return text


def serialize_for_csv(value):
    """
    Properly serialize a value for CSV.
    - None/empty: return empty string
    - Numbers (int/float): keep as numbers
    - Strings: normalize quotes, then return (csv module handles quoting/escaping)
    - Dict/list: convert to clean JSON string
    
    CRITICAL: Don't add extra quotes or escaping - the csv module handles it.
    """
    if value is None or value == '':
        return ''
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Keep numbers as numbers
        return value
    if isinstance(value, str):
        # Normalize smart quotes to straight quotes for better compatibility
        value = normalize_quotes(value)
        # Return string as-is; csv module will handle all escaping
        return value
    if isinstance(value, (dict, list)):
        # Convert to compact JSON
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    # Convert everything else to string
    return str(value)


def extract_grounded_links(enhanced_product_source_map):
    """
    Extract source URLs from Product Source Map.
    Returns newline-separated URLs.
    """
    if not enhanced_product_source_map:
        return ''
    
    urls = []
    if isinstance(enhanced_product_source_map, list):
        for product in enhanced_product_source_map:
            if isinstance(product, dict):
                source_urls = product.get('source_urls', [])
                if isinstance(source_urls, list):
                    urls.extend([url for url in source_urls if url])
    
    unique_urls = list(dict.fromkeys(urls))
    return '\n'.join(unique_urls)


def get_local_task_data(task_id, domain, model_name, run_number):
    """
    Get task data from local JSON files for a specific run.
    
    Returns:
        dict with data for this run, or None if missing
    """
    provider = get_provider_for_model(model_name)
    
    base_dir = os.path.join(
        RESULTS_DIR,
        provider,
        model_name,
        domain,
        f'run_{run_number}',
        f'task_{task_id}'
    )
    
    # Files to read
    test_case_file = os.path.join(base_dir, '0_test_case.json')
    grounded_file = os.path.join(base_dir, '1_grounded_response.json')
    scraped_file = os.path.join(base_dir, '2_scraped_sources.json')
    autograder_file = os.path.join(base_dir, '3_autograder_results.json')
    
    result = {
        'response_text': '',
        'product_source_map': None,
        'grounded_links': '',
        'score_overview': None,
        'scores': None,
        'total_score': '',
        'total_hurdle_score': '',
        'test_case': None
    }
    
    # Read test case
    if os.path.exists(test_case_file):
        try:
            with open(test_case_file, 'r', encoding='utf-8') as f:
                result['test_case'] = json.load(f)
        except Exception as e:
            pass
    
    # Read grounded response
    if os.path.exists(grounded_file):
        try:
            with open(grounded_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle null responseText
                response_text = data.get('responseText')
                result['response_text'] = response_text if response_text is not None else ''
        except Exception as e:
            pass
    
    # Read scraped sources
    if os.path.exists(scraped_file):
        try:
            with open(scraped_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                product_source_map_raw = data.get('productSourceMap', [])
                sources_list = data.get('sources', [])
                
                # Enhance Product Source Map with source_urls
                enhanced_product_map = []
                for product in product_source_map_raw:
                    enhanced_product = {
                        'product_name': product.get('product_name', ''),
                        'source_indices': product.get('source_indices', []),
                        'source_urls': []
                    }
                    
                    # Add URLs by looking up source_indices
                    for idx in product.get('source_indices', []):
                        source = next((s for s in sources_list if s.get('source_number') == idx + 1), None)
                        if source:
                            enhanced_product['source_urls'].append(source.get('source_link', ''))
                    
                    enhanced_product_map.append(enhanced_product)
                
                result['product_source_map'] = enhanced_product_map
                result['grounded_links'] = extract_grounded_links(enhanced_product_map)
        except Exception as e:
            pass
    
    # Read autograder results
    if os.path.exists(autograder_file):
        try:
            with open(autograder_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                result['score_overview'] = {
                    'summary': data.get('summary', {}),
                    'detailed_results': data.get('detailed_results', []),
                    'criteria_scores': data.get('criteria_scores', []),
                    'num_criteria': data.get('num_criteria', 0)
                }
                
                result['scores'] = data.get('criteria_scores_only', [])
                result['total_score'] = data.get('total_score', '')
                result['total_hurdle_score'] = data.get('total_hurdle_score', '')
        except Exception as e:
            pass
    
    return result


def get_all_task_ids_for_domain(domain):
    """Get all task IDs for a domain by scanning results directory"""
    task_ids = set()
    
    model = ALL_MODELS[0]
    provider = get_provider_for_model(model)
    
    domain_dir = os.path.join(RESULTS_DIR, provider, model, domain, 'run_1')
    
    if not os.path.exists(domain_dir):
        return []
    
    for item in os.listdir(domain_dir):
        if item.startswith('task_'):
            try:
                task_id = int(item.replace('task_', ''))
                task_ids.add(task_id)
            except ValueError:
                continue
    
    return sorted(list(task_ids))


def export_domain(domain):
    """Export all tasks for a domain"""
    
    print(f"\n{'='*80}")
    print(f"{domain.upper()} DOMAIN")
    print(f"{'='*80}")
    
    # Load workflows from dataset
    print(f"Loading workflows from dataset...")
    workflows = load_workflows_from_dataset(domain)
    print(f"  [+] Loaded {len(workflows)} unique task workflows")
    
    # Get all task IDs
    task_ids = get_all_task_ids_for_domain(domain)
    
    if not task_ids:
        print(f"  [!] No tasks found in results for {domain}")
        return False
    
    print(f"  Found {len(task_ids)} tasks")
    
    # Prepare CSV headers
    headers = ['Model Name', 'Task ID', 'Specified Prompt', 'Workflow', 'Criteria List']
    
    for run in range(1, 9):
        headers.extend([
            f'Response Text - {run}',
            f'Product Source Map - {run}',
            f'Grounded Links - {run}',
            f'Score Overview - {run}',
            f'Scores - {run}',
            f'Total Score - {run}',
            f'Total Hurdle Score - {run}',
        ])
    
    # Collect all rows
    all_rows = []
    
    for task_id in task_ids:
        print(f"  Task {task_id}...", end=' ')
        
        task_rows = []
        for model_name in ALL_MODELS:
            # Get data from run 1 first (for test case info)
            run1_data = get_local_task_data(task_id, domain, model_name, 1)
            
            if not run1_data or not run1_data['test_case']:
                continue
            
            test_case = run1_data['test_case']
            
            # Build row
            row = {
                'Model Name': model_name,
                'Task ID': task_id,  # Keep as integer
                'Specified Prompt': serialize_for_csv(test_case.get('prompt', '')),
                'Workflow': workflows.get(str(task_id), ''),
                'Criteria List': serialize_for_csv(test_case.get('criteria', [])),
            }
            
            # Add data for all 8 runs
            for run in range(1, 9):
                run_data = get_local_task_data(task_id, domain, model_name, run)
                
                row[f'Response Text - {run}'] = serialize_for_csv(run_data['response_text'])
                row[f'Product Source Map - {run}'] = serialize_for_csv(run_data['product_source_map'])
                row[f'Grounded Links - {run}'] = serialize_for_csv(run_data['grounded_links'])
                row[f'Score Overview - {run}'] = serialize_for_csv(run_data['score_overview'])
                row[f'Scores - {run}'] = serialize_for_csv(run_data['scores'])
                row[f'Total Score - {run}'] = serialize_for_csv(run_data['total_score'])
                row[f'Total Hurdle Score - {run}'] = serialize_for_csv(run_data['total_hurdle_score'])
            
            task_rows.append(row)
        
        all_rows.extend(task_rows)
        print(f"[+] ({len(task_rows)} models)")
    
    # Write to CSV with proper formatting
    if all_rows:
        output_file = os.path.join(OUTPUT_DIR, f"{domain}_results.csv")
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:  # utf-8-sig adds BOM
            writer = csv.DictWriter(
                csvfile, 
                fieldnames=headers,
                quoting=csv.QUOTE_MINIMAL,  # Match dataset format - only quote when needed
                doublequote=True,  # Escape internal quotes by doubling (CSV standard: " becomes "")
                lineterminator='\n'  # Unix line endings
            )
            writer.writeheader()
            writer.writerows(all_rows)
        
        file_size = os.path.getsize(output_file)
        print(f"\n  [+] Wrote {len(all_rows)} rows to {output_file}")
        print(f"     File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        return True
    else:
        print(f"  [!] No data to export for {domain}")
        return False


def main():
    """Main export function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Export task results to clean CSV files')
    parser.add_argument(
        '--domains',
        nargs='+',
        choices=ALL_DOMAINS,
        default=ALL_DOMAINS,
        help='Domain(s) to export (default: all domains)'
    )
    parser.add_argument(
        '--domain',
        choices=ALL_DOMAINS,
        help='Single domain to export (shorthand for --domains)'
    )
    
    args = parser.parse_args()
    
    # Determine which domains to export
    if args.domain:
        domains_to_export = [args.domain]
    else:
        domains_to_export = args.domains
    
    print("="*80)
    print("EXPORT RESULTS TO CSV FILES")
    print("="*80)
    print(f"Source directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Domains to export: {', '.join(domains_to_export)}")
    print(f"Models per task: {len(ALL_MODELS)}")
    print(f"Runs per model: 8")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export each domain
    success_count = 0
    for domain in domains_to_export:
        if export_domain(domain):
            success_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("[+] EXPORT COMPLETE")
    print("="*80)
    print(f"Successfully exported: {success_count}/{len(domains_to_export)} domains")
    print(f"\nOutput files in {OUTPUT_DIR}/:")
    
    for domain in domains_to_export:
        filename = os.path.join(OUTPUT_DIR, f"{domain}_results.csv")
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  [+] {domain}_results.csv ({file_size:,} bytes)")
        else:
            print(f"  [-] {domain}_results.csv (not created)")
    
    print("="*80)


if __name__ == '__main__':
    main()

