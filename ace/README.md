# ACE: AI Consumer Index

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An evaluation benchmark for testing grounded AI responses across multiple domains.

## What is ACE?

ACE (AI Consumer Index) is a comprehensive evaluation benchmark designed to test AI models with web search/grounding capabilities. It evaluates how well models generate recommendations backed by verifiable web sources across four consumer-focused domains.

**Key Features:**
- **Multi-Domain Testing**: Shopping, Food, Gaming, and DIY domains

## Supported Models

ACE supports the latest AI models from major providers:

### Google Gemini
- `gemini-2.5-pro` - Gemini 2.5 Pro with Google Search grounding
- `gemini-2.5-flash` - Gemini 2.5 Flash with Google Search grounding
- `gemini-3-pro` - Gemini 3 Pro with Google Search grounding

### OpenAI
- `gpt-5` - GPT-5 with web search
- `gpt-5.1` - GPT-5.1 with web search
- `o3` - O3 with web search
- `o3-pro` - O3 Pro with web search

### Anthropic
- `sonnet-4.5` - Claude Sonnet 4.5 with web search
- `opus-4.1` - Claude Opus 4.1 with web search
- `opus-4.5` - Claude Opus 4.5 with web search

## Project Architecture

### File Structure

```
ACE/
├── configs/                # Configuration and provider abstraction
│   ├── config.py          # Centralized credential management
│   ├── model_providers.py # Multi-provider abstraction layer
│   └── domain_config.py   # Domain-specific configurations
├── dataset/               # CSV datasets
│   ├── ACE-Shopping.csv
│   ├── ACE-Food.csv
│   ├── ACE-Gaming.csv
│   └── ACE-DIY.csv
├── harness/               # Core evaluation harness
│   ├── make-grounded-call.py   # Stage 1: Grounded API calls
│   ├── grounding-pipeline.py  # Stage 2: Scraping & mapping
│   ├── autograder.py          # Stage 3: Evaluation
│   └── helpers/              # YouTube, Reddit, purchase verification
├── pipeline/              # Scripts to run the harness
│   ├── runner.py             # Execute all tasks for a run in parallel
│   ├── init_from_dataset.py  # Database initialization
│   ├── test_single_task.py # Single task testing
│   ├── clear_run.py          # Clear specific run
│   ├── clear_all_runs.py     # Clear all runs
│   ├── regrade_task.py       # Re-grade tasks
│   ├── supabase_reader.py    # Read test cases from Supabase
│   └── local_file_reader.py  # Read test cases from local files
├── supabase-setup/        # Database setup scripts
│   ├── create_tables.sql     # Create all 80 tables
│   └── create_rls_policies.sql # Enable RLS and create policies
├── results/               # Output (provider/model/domain/run_N/task_ID/)
├── pyproject.toml        # Python dependencies
├── uv.lock               # Lock file for uv package manager
├── setup_xurls.sh        # Setup script for xurls binary
└── LICENSE               # MIT License
```

### Dataset Structure

The `dataset/` folder contains CSV files with evaluation tasks:

**Files:**
- `ACE-Shopping.csv` - Shopping domain (product recommendations)
- `ACE-Food.csv` - Food domain (recipes, restaurants)
- `ACE-Gaming.csv` - Gaming domain (game recommendations, strategies)
- `ACE-DIY.csv` - DIY domain (home improvement, repair guides)

**CSV Schema:**

Each CSV in `dataset/` contains the following columns:
- `Criterion ID` — *Primary Key* Unique identifier for each evaluation criterion within a task (integer)
- `Task ID` — Unique identifier for each task (integer)
- `Specified Prompt` — The user query or prompt that will be given to models
- `Criterion Type` — Indicates if the criterion is "grounding" or "non-grounding"
- `Description` — Brief description of the evaluation criterion (explains what is being measured)
- `Answer` — The reference answer used for comparison or grounding verification (may be empty for some domains and criteria)
- Shopping Domain only:  
    - `Product/Shop` — Indicates whether the item in question is asking for "product" or a "shop". This column is only present in ACE-Shopping.csv.


## Dependencies

**Python**: 3.11+

**AI Providers**:
- `anthropic` - Anthropic Claude API
- `openai` - OpenAI API
- `google-genai` - Google Gemini API

**Services**:
- `firecrawl-py` - Web scraping (required)
- `Searchapi-io` - Youtube Transcript scraping (required)

**Database**:
- `supabase` - Database backend (optional)

**Utilities**:
- `requests`, `python-dotenv`, `tqdm`, `html2text`

**External Tool**:
- `xurls` - REQUIRED for URL extraction (Go-based tool)

## Installation & Setup

### Step 1: Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install Python Dependencies

**Option A: Using uv (Recommended)**

`uv` automatically creates and manages a virtual environment for you:

```bash
uv sync
uv pip install -e .
```

> **Note**: `uv sync` automatically creates a virtual environment in `.venv/` and installs all dependencies. The `uv pip install -e .` installs the project in editable mode for proper imports.

**Option B: Using traditional venv**

If you prefer the traditional approach:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

> **Note**: With traditional venv, you must activate the environment before running scripts. The `pip install -e .` step is required for proper imports.

### Step 3: Install xurls (REQUIRED)

xurls is **required** for URL extraction from model responses.

**Prerequisites**: Go 1.22+

```bash
# Install Go (if not installed)
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Install xurls
./setup_xurls.sh
```

The script will install xurls to `~/go/bin/xurls`. Verify installation:

```bash
~/go/bin/xurls --version
```

### Step 4: Configure API Keys

Create a `.env` file in the project root:

**Required - At least one model provider:**

```bash
GEMINI_API_KEY=your-gemini-key        # For Gemini models
OPENAI_API_KEY=your-openai-key        # For OpenAI models
ANTHROPIC_API_KEY=your-anthropic-key  # For Anthropic models
```

**Required - Web scraping:**

```bash
FIRECRAWL_API_KEY=your-firecrawl-key
```

```bash
SEARCHAPI_API_KEY=your-searchapi-key  # YouTube transcripts (falls back if missing)
```

**Optional - Supabase (see section below):**

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

## Result Storage Configuration and Supabase

ACE supports two modes for storing and viewing evaluation results:

### Option 1: Local Files Only (No Database)

- **Use Case**: Quick testing, development, no database setup needed
- **How**: Add `--no-supabase` flag to all commands
- **Result Storage**:
  - All results stored as JSON files in `results/` directory
  - No centralized database for querying or analysis
  - Must manually parse JSON files to view results
  - Test cases must be manually created in results directory

### Option 2: Supabase Database + Local Files (Recommended)

- **Use Case**: Production runs, centralized result tracking and analysis
- **Result Storage**:
  - Results stored in BOTH Supabase database AND local JSON files
  - Database enables easy querying, filtering, and analysis across runs
  - Supports variance testing across multiple runs (run 1-8)
  - Can track results over time and compare models

**Setting Up Supabase:**

1. **Create Supabase Project**
   - Go to https://supabase.com
   - Create new project
   - Note your project URL and anon key

2. **Add Credentials to `.env`**
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-public-key
   ```

3. **Create Database Tables**
   - Open Supabase SQL Editor
   - Run SQL from `create_tables.sql`
   - This creates **ALL 80 tables** (10 models × 4 domains × 2 table types)
   - **Table Schema**: Each model/domain combination requires 2 tables:
     - **Tasks Table** (`tasks_{domain}_{model}`):
       - `"Task ID"` INTEGER PRIMARY KEY - Unique task identifier  
       - `"Specified Prompt"` TEXT - The prompt sent to the model
       - `"Criteria List"` JSONB - List of criterion IDs
       - `"Shop vs. Product"` TEXT - (Shopping domain only)
       - `"Response Text - {1-8}"` TEXT - Model responses for runs 1-8
       - `"Product Source Map - {1-8}"` JSONB - Product-to-source mappings
       - `"Grounding Source Meta Data - {1-8}"` JSONB - Source metadata
       - `"Direct Grounding - {1-8}"` JSONB - Raw API grounding data
       - `"Scores - {1-8}"` JSONB - Criterion scores arrays
       - `"Total Score - {1-8}"` NUMERIC - Total scores
       - `"Total Hurdle Score - {1-8}"` NUMERIC - Hurdle scores
       - `"Score Overview - {1-8}"` JSONB - Detailed scoring breakdown
       - `"Failed Grounded Sites - {1-8}"` JSONB - Failed scraping attempts
     
     - **Criteria Table** (`criteria_{domain}_{model}`):
       - `"Criterion ID"` INTEGER PRIMARY KEY - Unique criterion identifier
       - `"Task ID"` INTEGER - Links to tasks table
       - `"Prompt"` TEXT - Original criterion prompt
       - `"Description"` TEXT - Criterion description
       - `"Criterion Grounding Check"` TEXT - Grounding verification details
       - `"Hurdle Tag"` TEXT - Whether criterion is a hurdle (Yes/Not)
       - `"Criterion Type ({Domain})"` TEXT - Type of criterion (domain-specific)
       - `"Specified Prompt"` TEXT - Full evaluation prompt
       - `"Workflow"` TEXT - Workflow type
       - `"Shop vs. Product"` TEXT - (Shopping domain only)
       - `"Score - {1-8}"` INTEGER - Scores for runs 1-8 (-1, 0, or 1)
       - `"Reasoning - {1-8}"` JSONB - Detailed reasoning
       - `"Failure Step - {1-8}"` TEXT - Where criterion failed

4. **Apply RLS Policies** (Optional - for access control)
   - Open Supabase SQL Editor  
   - Run SQL from `create_rls_policies.sql`
   - This applies Row Level Security policies to **ALL 80 tables**
   - Policies allow full access (SELECT, INSERT, UPDATE, DELETE) for all users

5. **Populate Tables with CSV Data** (See Pipeline Order below - MUST BE DONE FIRST)

## Pipeline Order (IMPORTANT)

The pipeline must be run in the following order:

### STEP 1: Initialize Database (REQUIRED FIRST)

Before running any tasks, you must initialize the database tables from CSV datasets.

**Usage:**
```bash
python3 pipeline/init_from_dataset.py <domain> <model> [--no-supabase] [--overwrite] [--dry-run]
```

**Arguments:**
- `domain`: Domain name - `Shopping`, `Food`, `Gaming`, `DIY`, or `all`
- `model`: Model name - specific model (e.g., `gemini-2.5-pro`) or `all`

**Options:**
- `--no-supabase`: Skip database operations (create local files only)
- `--overwrite`: Overwrite existing data
- `--dry-run`: Preview without making changes

**Examples - With Supabase:**
```bash
# Initialize for specific domain and model
python3 pipeline/init_from_dataset.py Shopping gemini-2.5-pro

# Initialize all domains for a model
python3 pipeline/init_from_dataset.py all gemini-2.5-pro

# Initialize all domains for all models
python3 pipeline/init_from_dataset.py all all
```

**Without Supabase (Local Files Only):**
```bash
# Creates test case files in results/{provider}/{model}/{domain}/run_N/task_ID/
python3 pipeline/init_from_dataset.py Shopping gemini-2.5-pro --no-supabase
python3 pipeline/init_from_dataset.py all gemini-2.5-pro --no-supabase
```

**What this does:**
- Reads CSV files from `dataset/` folder
- **With Supabase**: Creates/populates database tables and stores resuls locally in `results/`
- **Without Supabase**: Creates local test case JSON files in `results/` directory

### STEP 2: Run Tasks

After initialization, run evaluations:

#### Option A: Test Single Task

**Usage:**
```bash
python3 pipeline/test_single_task.py <task_id> <model> <run_number> [domain] [--no-supabase] [--skip-autograder]
```

**Arguments:**
- `task_id`: Task ID from CSV dataset
- `model`: Model name (e.g., `gemini-2.5-pro`, `gpt-5`, `sonnet-4.5`)
- `run_number`: Run number (1-8) for variance testing
- `domain`: (Optional) Domain name - `Shopping`, `Food`, `Gaming`, or `DIY` (default: Shopping)

**Options:**
- `--no-supabase`: Skip Supabase (use local files only)
- `--skip-autograder`: Skip autograder (only run grounding + scraping)

**Examples:**
```bash
# With Supabase (default)
python3 pipeline/test_single_task.py 312 gemini-2.5-pro 1 Shopping
python3 pipeline/test_single_task.py 161 sonnet-4.5 1 Food

# With local files only
python3 pipeline/test_single_task.py 312 gemini-2.5-pro 1 Shopping --no-supabase
python3 pipeline/test_single_task.py 276 gpt-5 1 Gaming --no-supabase --skip-autograder
```

#### Option B: Run all tasks for a specific run for a model in a domain

**With Supabase:**
```bash
python3 pipeline/runner.py <domain> --model <model> --run <run_number> --workers <worker_number>

# Examples:
python3 pipeline/runner.py Shopping --model gemini-2.5-pro --run 1
python3 pipeline/runner.py Food --model gpt-5 --run 1 --workers 10
python3 pipeline/runner.py Gaming --model sonnet-4.5 --run 1 --force
```

**Without Supabase (Local Files Only):**
```bash
# Must initialize with --no-supabase first!
python3 pipeline/runner.py Shopping --model gemini-2.5-pro --run 1 --no-supabase
python3 pipeline/runner.py Food --model gpt-5 --run 1 --workers 10 --no-supabase
```

**Parameters:**
- `domain`: Shopping, Food, Gaming, or DIY (required)
- `--model`: Model name (required)
- `--run`: Run number 1-8 (required)
- `--workers`: Parallel workers (default: 10)
- `--force`: Re-run all tasks (ignore existing responses)
- `--skip-autograder`: Only run grounding + scraping (skip evaluation)
- `--no-supabase`: Use local files only (no database)

### STEP 3: Clean/Manage Results (Optional)

After running tasks, you can clean or re-grade:

**Clear Specific Run:**

**Usage:**
```bash
python3 pipeline/clear_run.py <domain> <run_number> --model <model_name> [--no-supabase] [--yes]
```

**Arguments:**
- `domain`: Domain name - `Shopping`, `Food`, `Gaming`, or `DIY`
- `run_number`: Run number (1-8)
- `--model`: Model name (required)

**Options:**
- `--no-supabase`: Skip Supabase (clear local files only)
- `--yes` / `-y`: Skip confirmation prompt

**Examples:**
```bash
# With Supabase:
python3 pipeline/clear_run.py Shopping 1 --model gemini-2.5-pro

# Local files only:
python3 pipeline/clear_run.py Shopping 1 --model gemini-2.5-pro --no-supabase

# Skip confirmation:
python3 pipeline/clear_run.py Shopping 1 --model gemini-2.5-pro --yes
```

**Clear All Runs:**

**Usage:**
```bash
python3 pipeline/clear_all_runs.py <domain> --model <model_name> [--no-supabase] [--start-run N] [--end-run N] [--confirm]
```

**Arguments:**
- `domain`: Domain name - `Shopping`, `Food`, `Gaming`, or `DIY`
- `--model`: Model name (required)

**Options:**
- `--no-supabase`: Skip Supabase (clear local files only)
- `--start-run`: Start from this run number (default: 1)
- `--end-run`: End at this run number (default: 8)
- `--confirm`: Require confirmation before starting

**Examples:**
```bash
# With Supabase - clear all runs 1-8:
python3 pipeline/clear_all_runs.py Shopping --model gemini-2.5-pro

# Local files only:
python3 pipeline/clear_all_runs.py Shopping --model gemini-2.5-pro --no-supabase

# Clear runs 3-5 only:
python3 pipeline/clear_all_runs.py Shopping --model gemini-2.5-pro --start-run 3 --end-run 5
```

**Re-grade Specific Task:**

**Usage:**
```bash
python3 pipeline/regrade_task.py <task_id> <domain> <model> <run_number> [--no-supabase] [--dry-run]
```

**Arguments:**
- `task_id`: Task ID to re-grade
- `domain`: Domain name - `Shopping`, `Food`, `Gaming`, or `DIY`
- `model`: Model name (e.g., `gemini-2.5-pro`, `gpt-5`)
- `run_number`: Run number (1-8)

**Options:**
- `--no-supabase`: Skip Supabase (use local files only)
- `--dry-run`: Preview without updating database

**Examples:**
```bash
# With Supabase:
python3 pipeline/regrade_task.py 312 Shopping gemini-2.5-pro 1

# Local files only:
python3 pipeline/regrade_task.py 312 Shopping gemini-2.5-pro 1 --no-supabase

# Dry run (preview only):
python3 pipeline/regrade_task.py 312 Shopping gemini-2.5-pro 1 --dry-run
```

> **⚠️ IMPORTANT**: After clearing runs with `clear_run.py` or `clear_all_runs.py`, you **MUST** re-initialize from the dataset before running new tasks:
>
> ```bash
> # After clearing, re-initialize:
> python3 pipeline/init_from_dataset.py <domain> <model>
> # OR with local files:
> python3 pipeline/init_from_dataset.py <domain> <model> --no-supabase
> 
> # Then you can run tasks again:
> python3 pipeline/runner.py <domain> --model <model> --run <run_number>
> ```

## Output Structure

Results are saved in: `results/{provider}/{model}/{domain}/run_{N}/task_{ID}/`

**Example:** `results/gemini/gemini-2.5-pro/Shopping/run_1/task_715/`

**Each task folder contains:**
- `0_test_case.json` - Original test case from database/CSV
- `1_grounded_response.json` - Model response with grounding metadata
- `2_scraped_sources.json` - Scraped web sources and mappings
- `3_autograder_results.json` - Evaluation scores and reasoning

## How the Pipeline Works

### Stage 1: Grounded API Call
**Script**: `harness/make-grounded-call.py`

- Sends prompt to AI model with web search enabled
- Model generates response with citations to web sources
- Saves grounded response with metadata

### Stage 2: Scraping & Mapping
**Script**: `harness/grounding-pipeline.py`

- Scrapes all cited sources using Firecrawl API
- Extracts recommendations from response
- Maps recommendations to sources
- Handles YouTube transcripts and Reddit discussions

### Stage 3: Autograding
**Script**: `harness/autograder.py`

- Two-stage verification per criterion:
  - **Stage 1**: Verify claim in response text
  - **Stage 2**: Verify claim in grounded sources
- Generates scores and detailed reasoning

## Quick Start Example

### With Supabase (Database Mode)

Complete workflow for testing gemini-2.5-pro on Shopping domain:

```bash
# 1. Setup (one-time)
uv sync
./setup_xurls.sh
# Create .env with API keys (including SUPABASE_URL and SUPABASE_KEY)

# 2. Initialize database
python3 pipeline/init_from_dataset.py Shopping gemini-2.5-pro

# 3. Test single task
python3 pipeline/test_single_task.py 312 gemini-2.5-pro 1 Shopping

# 4. Run all tasks for run 1
python3 pipeline/runner.py Shopping --model gemini-2.5-pro --run 1 --workers 10

# 5. Check results
ls results/gemini/gemini-2.5-pro/Shopping/run_1/
```

### Without Supabase (Local Files Only)

```bash
# 1. Setup (one-time)
uv sync
./setup_xurls.sh
# Create .env with API keys (Supabase keys NOT required)

# 2. Initialize local test cases
python3 pipeline/init_from_dataset.py Shopping gemini-2.5-pro --no-supabase

# 3. Test single task
python3 pipeline/test_single_task.py 312 gemini-2.5-pro 1 Shopping --no-supabase

# 4. Run all tasks for run 1
python3 pipeline/runner.py Shopping --model gemini-2.5-pro --run 1 --workers 10 --no-supabase

# 5. Check results
ls results/gemini/gemini-2.5-pro/Shopping/run_1/
```

## Troubleshooting

### xurls not found

- Ensure Go is installed and in PATH
- Run `./setup_xurls.sh`
- Check `~/go/bin/xurls` exists

### Supabase connection errors

- Verify SUPABASE_URL and SUPABASE_KEY in `.env`
- Check tables exist in Supabase dashboard
- Run `python3 tests/check_supabase_connection.py`
- **Alternative**: Use `--no-supabase` flag to run without database

### Firecrawl errors

- Verify FIRECRAWL_API_KEY is valid
- Check Firecrawl API quota

## License

MIT License - see [LICENSE](LICENSE) file for details.
