#!/usr/bin/env python3
"""
Complete Grounding Pipeline
Stage 2 of Pipeline:
1. Extract recommendations and map citations (Logic moved from Stage 1)
2. Scrape sources (API chunks + Text links)
3. Generate productSourceMap for Autograder
"""

# Load environment variables from .env file first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Set, Optional, Any

# Add project root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.logging_config import setup_logging
from configs.config import config

logger = setup_logging(__name__)

# Third-party imports
from firecrawl import FirecrawlApp
from google import genai
from google.genai import types

from helpers.youtube_utils import (
    is_youtube_url,
    extract_video_id,
    get_youtube_transcript,
    format_transcript_markdown
)
from helpers.reddit_utils import (
    is_reddit_url,
    get_reddit_content,
    format_reddit_markdown
)

# Configuration
config.validate_firecrawl()
FIRECRAWL_API_KEY = config.FIRECRAWL_API_KEY

# Pipeline Configuration Constants
# DO NOT CHANGE without extensive testing - these are tuned values

# URL Extraction
URL_EXTRACTION_TIMEOUT = 60  # seconds for xurls command

# Gemini Helper Functions (for mapping/extraction only, not main API calls)
GEMINI_HELPER_MODEL = 'gemini-2.5-flash'  # Model for extraction/mapping tasks
GEMINI_HELPER_TEMPERATURE = 0  # Deterministic for consistent results

# Citation Mapping Performance
MAX_CITATION_WORKERS = 100  # Parallel workers for citation mapping

# Web Scraping Configuration (Firecrawl)
SCRAPE_MAX_RETRIES = 3  # Number of retry attempts per URL
SCRAPE_RETRY_DELAY = 3  # Seconds between retry attempts

# URL Redirect Resolution (for Google vertexaisearch redirects)
REDIRECT_MAX_HOPS = 10
REDIRECT_CONNECT_TIMEOUT = 20.0  # seconds
REDIRECT_READ_TIMEOUT = 20.0  # seconds
REDIRECT_RETRY_TOTAL = 3
REDIRECT_RETRY_CONNECT = 3
REDIRECT_RETRY_READ = 2
REDIRECT_BACKOFF_FACTOR = 0.4

# Gemini client for helper functions - initialize once when first needed
_gemini_client = None


# --- Helper Functions ---

def clean_url(url: str) -> str:
    """Remove trailing markdown artifacts from URL"""
    while url and url[-1] in ')]},;:.':
        url = url[:-1]
    return url


def extract_urls_regex(text: str) -> List[str]:
    """Fallback URL extraction using regex when xurls is not available"""
    url_pattern = r'https?://[^\s\)<>\]"]+'
    urls = re.findall(url_pattern, text)
    cleaned = [clean_url(url) for url in urls if clean_url(url)]
    return list(dict.fromkeys(cleaned))  # Deduplicate preserving order


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing tracking parameters for duplicate detection.

    ONLY removes: utm_source, utm_medium, utm_campaign
    PRESERVES: All other params (product ID, category, color, etc.)

    Examples:
        https://ebay.com/itm/123?utm_source=openai → https://ebay.com/itm/123
        https://ebay.com/itm/122?utm_source=openai → https://ebay.com/itm/122
        https://site.com?id=5&utm_source=x → https://site.com?id=5

    Returns original URL if parsing fails.
    """
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        # Remove ONLY tracking parameters
        query_params.pop('utm_source', None)
        query_params.pop('utm_medium', None)
        query_params.pop('utm_campaign', None)
        clean_query = urlencode(query_params, doseq=True)
        normalized_parts = list(parsed)
        normalized_parts[4] = clean_query
        return urlunparse(normalized_parts)
    except:
        return url


def resolve_redirect_url(url: str) -> Optional[str]:
    """
    Resolve redirect URL to get final destination.
    Used for Google vertexaisearch redirect URLs from Gemini grounding.

    Manually follows redirects without downloading page content.

    Args:
        url: URL that may be a redirect (e.g., vertexaisearch.cloud.google.com/...)

    Returns:
        str: Final URL after following redirects, or None on error
    """
    try:
        import time
        import requests
        from urllib.parse import urljoin

        session = requests.Session()

        # Configure retry adapter for transient errors
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=requests.packages.urllib3.util.retry.Retry(
                total=REDIRECT_RETRY_TOTAL,
                connect=REDIRECT_RETRY_CONNECT,
                read=REDIRECT_RETRY_READ,
                backoff_factor=REDIRECT_BACKOFF_FACTOR,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET"]
            )
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8"
        }

        current_url = url

        for hop in range(REDIRECT_MAX_HOPS):
            # Try HEAD first (faster, no body download)
            try:
                response = session.head(
                    current_url,
                    headers=headers,
                    allow_redirects=False,
                    timeout=(REDIRECT_CONNECT_TIMEOUT, REDIRECT_READ_TIMEOUT)
                )
            except requests.RequestException:
                # HEAD failed, try GET with stream
                response = None

            # Fallback to GET if HEAD not supported (405) or failed
            if response is None or response.status_code == 405:
                try:
                    response = session.get(
                        current_url,
                        headers=headers,
                        allow_redirects=False,
                        stream=True,
                        timeout=(REDIRECT_CONNECT_TIMEOUT, REDIRECT_READ_TIMEOUT)
                    )
                    response.close()  # Close immediately to avoid downloading body
                except requests.RequestException:
                    # Both HEAD and GET failed
                    return None

            # Check if this is a redirect (3xx status code)
            if response.status_code < 300 or response.status_code >= 400:
                # Not a redirect, this is the final URL
                return response.url if hasattr(response, 'url') else current_url

            # Get redirect location
            location = response.headers.get('Location') or response.headers.get('location')
            if not location:
                # Redirect without Location header
                return current_url

            # Handle relative redirects
            current_url = urljoin(current_url, location)

            # Small delay to be polite
            time.sleep(0.1)

        # Max redirects reached, return last URL
        return current_url

    except Exception:
        return None


def get_gemini_client():
    """Lazy initialization of Gemini client"""
    global _gemini_client
    if _gemini_client is None:
        try:
            # Load Gemini API key from environment
            config.validate_model_key('gemini')
            _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
            print("   Initialized Gemini for helper functions")
        except Exception as e:
            print(f"   ⚠️  Gemini initialization failed: {e}")
            raise
    return _gemini_client


# --- Helper Functions (Moved from make-grounded-call.py) ---

def extract_and_map_response_links(response_text, product_names):
    """
    Extract URLs using xurls CLI and map them to products using Gemini.
    Returns: {url: [product_names]} - only URLs mapped to at least one product
    """
    import subprocess

    if not product_names:
        return {}

    # Use xurls CLI to extract URLs
    import os
    xurls_path = os.path.expanduser('~/go/bin/xurls')

    # Fallback to 'xurls' if not in default Go path
    if not os.path.exists(xurls_path):
        xurls_path = 'xurls'

    try:
        result = subprocess.run(
            [xurls_path],
            input=response_text,
            capture_output=True,
            text=True,
            timeout=60
        )
        urls = [url.strip() for url in result.stdout.strip().split('\n') if url.strip()]

        # Clean URLs and deduplicate
        cleaned_urls = [clean_url(url) for url in urls if clean_url(url)]
        urls = list(dict.fromkeys(cleaned_urls))

        if not urls:
            return {}

        print(f"   Found {len(urls)} URLs via xurls")

    except FileNotFoundError:
        print(f"   ⚠️  xurls not installed at {xurls_path}, using regex fallback")
        urls = extract_urls_regex(response_text)

        if not urls:
            return {}

        print(f"   Found {len(urls)} URLs via regex")

    except Exception as e:
        print(f"   ⚠️  Error running xurls: {e}, using regex fallback")
        urls = extract_urls_regex(response_text)
        if not urls:
            return {}

    # Map to products using Gemini (same as citation mapping)
    prompt = f"""Map these URLs to products based on proximity and context.

Products: {json.dumps(product_names, indent=2)}
URLs: {json.dumps(urls, indent=2)}

Response:
{response_text}

IMPORTANT: Do NOT map image URLs (e.g., imgur.com, i.imgur.com, images, .png, .jpg, .jpeg) to products.
Only map shopping links, product pages, or relevant content URLs to products.

Return JSON mapping each URL to product name(s) it relates to:
{{"url": ["Product1", "Product2"], ...}}

If URL doesn't relate to any product or is an image URL, map to empty array."""

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        text = response.text.strip()

        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        mapping = json.loads(text)

        # Debug: Show what Gemini returned
        total_urls = len(mapping)
        mapped_urls = {url: prods for url, prods in mapping.items() if prods}
        unmapped_count = total_urls - len(mapped_urls)

        if unmapped_count > 0:
            print(f"   Gemini filtered out {unmapped_count} URLs (no product match)")

        # Return only URLs with at least one product
        return mapped_urls

    except Exception as e:
        error_str = str(e)
        if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
            print(f"   ⚠️  Rate limit hit - failing task for retry: {e}")
            raise  # Re-raise to fail the task
        print(f"   ⚠️  Error mapping links: {e}")
        return {}

#Only impacts Shopping and Food domains when grading
def extract_recommendations(response_text, query):
    """Extract all specific recommendations from response (works for any domain)"""

    prompt = f"""Extract the MAIN recommendations from this response. These are the specific recommendations/items/solutions/products/services/etc. that directly answer the user's query.

    USER QUERY:
    {query}

    RESPONSE TEXT:
    {response_text}

    IMPORTANT - READ THE QUERY CAREFULLY:
    First, understand what the user is ACTUALLY asking to be recommended. Look at the query to distinguish:
    - Meta-information requests: "give me the link to X", "find the post about Y", "what article discusses Z"
    - Actual recommendation requests: "recommend a product", "suggest a restaurant", "what should I buy"

    ONLY extract items that are being RECOMMENDED as solutions/answers, NOT items that are:
    - Background information or context

    WHAT TO EXTRACT:
    Main recommendations are usually clearly identified by:
    - Headers or Numbered formatting: "1. [Item]", "2. [Item]", "Option 1:", "Recommendation 1:"
    - Direct answer sections: "I recommend...", "Here are...", "Top picks:"
    - Clear labels that indicate primary suggestions like "Product that meets your specs:", "Recommended [X]:"

    Extract the SPECIFIC NAMES of what's being recommended (could be products, recipes, strategies, games, tools, approaches, etc. or any other specific items being recommended based on the user's query.).

INCLUDE:
    - Main numbered/labeled recommendations from primary sections that directly answer the recommendation request
    - Specific names that are being RECOMMENDED as solutions
    - Items in "Recommended [X]", "Option 1/2/3", or "Top [X]" sections when they are recommendations
    - **"Closest Matches" or "Best Available Options"** - These ARE recommendations when:
      * They are the primary/only suggestions offered
      * They are presented in a structured format with details
      * The response is genuinely trying to help even if caveats are mentioned
    - Items presented with caveats like "while not perfect" or "with some compromises" - these are still recommendations if they're what's being suggested

**CRITICAL DISTINCTION - "Closest Match" vs "Alternative":**
    - "Closest Matches (with caveats)" followed by specific products = EXTRACT THESE (they ARE the recommendations)
    - "I recommend X. Alternatively, you could try Y" = Extract X only (Y is truly an alternative)
    - "I couldn't find exactly what you want, but here are options" = EXTRACT the options (they're the best-effort recommendations)
    - "If X doesn't work, try Y" = Extract X only (Y is a backup)

DO NOT include:
- Sub-items, examples, or details mentioned WITHIN a recommendation
- TRUE alternatives that come AFTER a primary recommendation (e.g., "Alternatively...", "Or you could try...")
- Items explicitly labeled as "Other options if the above don't work"
- If the recommendation is a product or service, ignore action phrases, and dont treat them as separate recommendation if you have already extracted the product or service exactly describing that action.
- Generic categories without specific names
- Items explicitly rejected or said NOT to buy
- Games or products mentioned only as comparisons or references (e.g., "unlike X which has..." or "compared to Y...")


CRITICAL RULES:
    1. If the response uses numbered formatting (1., 2., 3.), READ THE CONTEXT to determine if each item is a RECOMMENDATION or just INFORMATION. Don't blindly extract all numbered items.
    2. **"Closest Matches" ARE recommendations** - If the response says "I couldn't find a perfect match but here are the closest options" and then lists specific items, EXTRACT THOSE ITEMS. They are the best-effort recommendations.
    3. Only ignore TRUE alternatives that come AFTER a primary recommendation is given. If the "alternative" section is the ONLY substantive suggestion, it IS the recommendation.
    4. IGNORE action phrases if the recommendation is a product or service: If text says "contact ABC directly" or "visit XYZ", extract only "ABC" or "XYZ", not the entire action phrase.
    5. Remove duplicate variations: If "ABC Shop" and "contact ABC Shop directly" both appear, only include "ABC Shop" once.
    6. If the query asks "give me the link to X and recommend Y", only extract Y (the recommendation), not X (the link/meta-information).
    7. **Caveats don't disqualify recommendations**: If the response says "Product X (though it doesn't meet all your criteria)" and presents it as an option to consider, it IS a recommendation.

    FORMATTING RULES:
    - "Option 1 - Product A" → extract "Product A" (remove the label)
    - "1. Restaurant Name" → extract "Restaurant Name" (remove the number)

    Return ONLY a JSON array of the main recommendation names:
    ["Recommendation 1", "Recommendation 2", ...]

**When to return empty []:**
- Response explicitly refuses to recommend anything (e.g., "I cannot recommend any products")
- Response only provides general advice with NO specific items mentioned
- Response only asks clarifying questions without suggestions

**When to extract (NOT empty):**
- Response says "here are the closest matches" and lists items → EXTRACT them
- Response says "I couldn't find exactly what you want, but consider X" → EXTRACT X
- Response hedges but still provides specific named options → EXTRACT them"""

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        text = response.text.strip()

        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        return json.loads(text)
    except Exception as e:
        error_str = str(e)
        if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
            print(f"   ⚠️  Rate limit hit - failing task for retry: {e}")
            raise  # Re-raise to fail the task
        print(f"   ⚠️  Recommendation extraction failed: {e}")
        return []


def match_citation_to_recommendations(citation_text, recommendation_names, full_response_text, start_index, end_index, chunk_url):
    """Step 2: Determine which recommendations a citation supports using full context"""
    prompt = f"""Given a citation from an AI response, determine which recommendation(s) it supports.

FULL RESPONSE TEXT (for context):
{full_response_text}

CITATION DETAILS:
- Citation Text: "{citation_text}"
- Position in Response: characters {start_index} to {end_index}
- Source URL: {chunk_url}

RECOMMENDATIONS (0-indexed):
{json.dumps(recommendation_names, indent=2)}

TASK: Determine which recommendation(s) this citation supports based on:
    1. The citation text content
    2. The citation's position in the response (which recommendation section is it in?) (if applicable)
3. Context from the full response

IMPORTANT:
- Citations that say "the game", "it", "this" refer to the recommendation they appear under or relative contextual location
- Price/purchase citations belong to whichever product section they're in
- Use the start/end indices to determine which product section the citation is in

Return the indices (0-based) of recommendations this citation supports.
Return ONLY a JSON array of indices: [0, 1, 2]

If citation doesn't support any specific recommendation, return: []"""

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        text = response.text.strip()

        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        return json.loads(text)
    except Exception as e:
        error_str = str(e)
        if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
            raise  # Re-raise rate limit errors to fail the task
        return []


def create_recommendation_source_map(response_text, grounding_json, query):
    """Two-step deterministic mapping: extract recommendations, then map citations in parallel"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    supports = grounding_json.get('groundingSupports', [])
    chunks = grounding_json.get('groundingChunks', [])

    # Step 1: Extract recommendations (always do this, even without API grounding)
    print(f"   Step 1: Extracting recommendations...")
    recommendation_names = extract_recommendations(response_text, query)

    if not recommendation_names:
        print(f"   ⚠️  No recommendations extracted from response")
        return []

    print(f"   ✅ Found {len(recommendation_names)} recommendations")

    # Initialize recommendations with empty source sets
    recommendations = [
        {'product_name': name, 'source_indices': set()}  # Keep 'product_name' key for backward compatibility
        for name in recommendation_names
    ]

    # Step 2: Map citations to recommendations (skip if no API grounding)
    if not chunks:
        print(f"   ⚠️  No API grounding chunks - will rely on response text links only")
    else:
        print(f"   Step 2: Mapping {len(supports)} citations to recommendations in parallel...")

        # Prepare all citation mapping tasks
        citation_tasks = []
        for i, support in enumerate(supports):
            citation_text = support['segment']['text']
            start_index = support['segment'].get('startIndex', 0)
            end_index = support['segment'].get('endIndex', 0)
            chunk_indices = support['groundingChunkIndices']

            # Get URL from first chunk
            chunk_url = ''
            if chunk_indices and chunk_indices[0] < len(chunks):
                chunk_url = chunks[chunk_indices[0]].get('web', {}).get('uri', '')

            citation_tasks.append({
                'index': i,
                'citation_text': citation_text,
                'start_index': start_index,
                'end_index': end_index,
                'chunk_indices': chunk_indices,
                'chunk_url': chunk_url
            })

        # Process all citations in parallel
        with ThreadPoolExecutor(max_workers=MAX_CITATION_WORKERS) as executor:
            futures = {
                executor.submit(
                    match_citation_to_recommendations,
                    task['citation_text'],
                    recommendation_names,
                    response_text,
                    task['start_index'],
                    task['end_index'],
                    task['chunk_url']
                ): task
                for task in citation_tasks
            }

            # Collect results as they complete
            for future in as_completed(futures):
                task = futures[future]
                try:
                    matching_rec_indices = future.result()
                    chunk_indices = task['chunk_indices']

                    # Add the chunk_indices to the matched recommendations
                    for rec_idx in matching_rec_indices:
                        if 0 <= rec_idx < len(recommendations):
                            recommendations[rec_idx]['source_indices'].update(chunk_indices)
                except Exception as e:
                    error_str = str(e)
                    if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                        print(f"   ⚠️  Rate limit hit - failing task for retry: {e}")
                        raise  # Re-raise to fail the task
                    print(f"   ⚠️  Citation {task['index']} mapping failed: {e}")

    # Convert sets to sorted lists
    final_map = [
        {
            'product_name': rec['product_name'],  # Keep 'product_name' key for backward compatibility
            'source_indices': sorted(list(rec['source_indices']))
        }
        for rec in recommendations
    ]

    # Diagnostic: Check for unmapped sources (only if we have API chunks)
    if chunks:
        all_mapped_indices = set()
        for rec in recommendations:
            all_mapped_indices.update(rec['source_indices'])

        all_source_indices = set(range(len(chunks)))
        unmapped_indices = all_source_indices - all_mapped_indices

        if unmapped_indices:
            print(f"   ⚠️  Warning: {len(unmapped_indices)} source(s) not mapped to any recommendation")
            print(f"      Unmapped source indices: {sorted(unmapped_indices)}")
            print(f"      This happens when Gemini provides sources but doesn't cite them in the response")

        print(f"   ✅ Mapped recommendations to grounding sources")
        print(f"      Total sources: {len(chunks)} | Mapped: {len(all_mapped_indices)} | Unmapped: {len(unmapped_indices)}")
    else:
        print(f"   No API grounding - products will use response text links as sources")

    return final_map


# --- End Helper Functions ---


class GroundingProcessor:
    """Process Gemini grounding metadata and scrape sources"""

    def __init__(self):
        # Always use Firecrawl for scraping
            print('[*] Using Firecrawl for scraping\n')
            self.firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    def parse_json_file(self, file_path):
        """Parse JSON grounding file"""
        print(f'Reading file: {file_path}\n')

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract chunks
        chunks = [
            {
                'index': i,
                'title': chunk.get('web', {}).get('title', 'Unknown'),
                'uri': chunk.get('web', {}).get('uri', '')
            }
            for i, chunk in enumerate(data.get('groundingChunks', []))
        ]

        # Extract supports
        supports = [
            {
                'chunk_indices': s.get('groundingChunkIndices', []),
                'text': s.get('segment', {}).get('text', '')
            }
            for s in data.get('groundingSupports', [])
        ]

        print(f'✅ Found {len(chunks)} sources')
        print(f'✅ Found {len(supports)} citations\n')

        return chunks, supports

    def scrape_sources(self, chunks):
        """Scrape all source URLs using Firecrawl"""
        import time
        scraping_start = time.time()

        print(f'Scraping {len(chunks)} sources...\n')

        scraped_data = []
        failed_sites = []  # Track failed scrapes

        # Use Firecrawl API for all scraping
        for i, chunk in enumerate(chunks):
                print(f'[{i + 1}/{len(chunks)}] {chunk["title"]}')

                # Resolve redirects if needed (for vertexaisearch URLs)
                final_url = chunk['uri']
                if 'vertexaisearch.cloud.google.com' in chunk['uri']:
                    print(f'   Resolving redirect...')
                    resolved = resolve_redirect_url(chunk['uri'])
                    if resolved:
                        final_url = resolved
                        print(f'   ✅ Resolved to: {final_url[:80]}...')
                    else:
                        print(f'   ⚠️  Could not resolve redirect, using original URL')

                # Check if this is a YouTube URL
                is_youtube = is_youtube_url(final_url)
                youtube_transcript = None

                if is_youtube:
                    print(f'   YouTube video detected')
                    video_id = extract_video_id(final_url)

                    if video_id:
                        print(f'   Fetching transcript for video: {video_id}')
                        transcript_result = get_youtube_transcript(video_id)

                        if transcript_result['success']:
                            youtube_transcript = transcript_result['transcript']
                            print(f'   ✅ Transcript retrieved ({len(youtube_transcript):,} chars)')
                        else:
                            print(f'   ⚠️  Transcript unavailable: {transcript_result["error"]}')
                    else:
                        print(f'   ⚠️  Could not extract video ID')

                # Check if this is a Reddit URL
                is_reddit = is_reddit_url(final_url)
                reddit_content = None

                if is_reddit:
                    print(f'   Reddit post detected')
                    print(f'   Fetching Reddit discussion...')
                    reddit_result = get_reddit_content(final_url)

                    if reddit_result['success']:
                        reddit_content = reddit_result['markdown']
                        print(f'   ✅ Reddit content retrieved ({len(reddit_content):,} chars)')
                    else:
                        print(f'   ⚠️  Reddit fetch failed: {reddit_result["error"]}')

                # Retry logic for transient failures
                max_retries = 3
                retry_delay = 3  # seconds

                # Skip Firecrawl for Reddit URLs - use custom scraper only
                if is_reddit:
                    if reddit_content:
                        # Format Reddit content
                        reddit_title = chunk['title']
                        if reddit_result.get('title'):
                            reddit_title = reddit_result['title']

                        markdown = format_reddit_markdown(
                            reddit_content,
                            final_url,
                            reddit_title
                        )

                        scraped_data.append({
                            'index': chunk['index'],
                            'title': chunk['title'],
                            'uri': chunk['uri'],
                            'content': {
                                'html': '',  # Not using HTML
                                'text': markdown,
                                'title': reddit_title,
                                'url': final_url,
                                'source_type': 'reddit_discussion'
                            },
                            'error': None
                        })

                        print(f'   ✅ Reddit scraping success ({len(markdown):,} chars)')
                    else:
                        # Reddit fetch failed
                        error_msg = reddit_result.get('error', 'Unknown error')
                        scraped_data.append({
                            'index': chunk['index'],
                            'title': chunk['title'],
                            'uri': chunk['uri'],
                            'error': error_msg,
                            'content': None
                        })

                        failed_sites.append({
                            'source_number': chunk['index'] + 1,
                            'source_title': chunk['title'],
                            'source_link': chunk['uri'],
                            'error': error_msg,
                            'attempts': 1
                        })

                    continue  # Skip Firecrawl for Reddit URLs

                for attempt in range(max_retries + 1):
                    try:
                        result = self.firecrawl.scrape(
                            final_url,  # Use resolved URL (not redirect)
                            formats=['markdown'],
                            proxy='auto',
                            only_main_content=False  # Get full page, not just main content
                        )

                        markdown = result.markdown if hasattr(result, 'markdown') else ''
                        title = ''
                        url = chunk['uri']

                        if hasattr(result, 'metadata'):
                            if hasattr(result.metadata, 'title'):
                                title = result.metadata.title
                            if hasattr(result.metadata, 'url'):
                                url = result.metadata.url

                        # If YouTube video with transcript, prepend transcript to markdown
                        if youtube_transcript:
                            transcript_section = format_transcript_markdown(
                                youtube_transcript,
                                final_url,  # Use resolved URL
                                title or chunk['title']
                            )
                            markdown = transcript_section + markdown
                            print(f'   ✅ Combined transcript + page content')

                        scraped_data.append({
                            'index': chunk['index'],
                            'title': chunk['title'],
                            'uri': chunk['uri'],
                            'content': {
                                'html': '',  # Not using HTML
                                'text': markdown,
                                'title': title,
                                'url': url,
                                'source_type': 'youtube_video' if is_youtube else 'webpage'
                            },
                            'error': None
                        })

                        retry_msg = f" (retry {attempt})" if attempt > 0 else ""
                        print(f'   ✅ Firecrawl success{retry_msg} ({len(markdown):,} chars)')
                        break  # Success - exit retry loop

                    except Exception as error:
                        if attempt < max_retries:
                            print(f'   ⚠️  Attempt {attempt + 1} failed, retrying in {retry_delay}s...')
                            time.sleep(retry_delay)
                        else:
                            # All retries failed
                            error_msg = str(error)[:200]

                            # If YouTube with transcript, save transcript even if Firecrawl failed
                            if youtube_transcript:
                                print(f'   ⚠️  Firecrawl failed but have transcript - saving transcript only')
                                markdown = format_transcript_markdown(
                                    youtube_transcript,
                                    final_url,  # Use resolved URL
                                    chunk['title']
                                )

                                scraped_data.append({
                                    'index': chunk['index'],
                                    'title': chunk['title'],
                                    'uri': chunk['uri'],
                                    'content': {
                                        'html': '',
                                        'text': markdown,
                                        'title': chunk['title'],
                                        'url': chunk['uri'],
                                        'source_type': 'youtube_transcript_only'
                                    },
                                    'error': f'Firecrawl failed but transcript available: {error_msg}'
                                })
                            else:
                                # Complete failure
                                print(f'   ❌ Error after {max_retries + 1} attempts: {error_msg}')

                                scraped_data.append({
                                    'index': chunk['index'],
                                    'title': chunk['title'],
                                    'uri': chunk['uri'],
                                    'error': error_msg,
                                    'content': None
                                })

                                # Track failed site with details
                                failed_sites.append({
                                    'source_number': chunk['index'] + 1,
                                    'source_title': chunk['title'],
                                    'source_link': chunk['uri'],
                                    'error': error_msg,
                                    'attempts': max_retries + 1
                                })

        scraping_time = time.time() - scraping_start
        return scraped_data, scraping_time, failed_sites

    def create_output(self, chunks, supports, scraped_data, failed_sites=None, product_map=None, response_text='', query='', criteria=None, task_id=None, test_id=None, direct_grounding=None, provider='gemini', shop_vs_product=None):
        """Map citations to sources"""
        print('\nCreating structured output...\n')

        sources = []

        for chunk in chunks:
            # Find citations for this source
            relevant_texts = [
                s['text'] for s in supports
                if chunk['index'] in s['chunk_indices']
            ]

            # Get scraped content
            scraped = next((s for s in scraped_data if s['index'] == chunk['index']), None)

            # Get content from scraped data (Firecrawl provides markdown directly)
            text_content = scraped.get('content', {}).get('text') if scraped and scraped.get('content') else None

            webpage_content = {
                'title': scraped.get('content', {}).get('title') if scraped and scraped.get('content') else None,
                'url': scraped.get('content', {}).get('url') if scraped and scraped.get('content') else None,
                'markdown': text_content,  # Firecrawl returns markdown as text
                'text': text_content,
                'error': scraped.get('error') if scraped else None,
                'source_type': scraped.get('content', {}).get('source_type', 'webpage') if scraped and scraped.get('content') else 'webpage'
            }

            sources.append({
                'source_number': chunk['index'] + 1,
                'source_title': chunk['title'],
                'source_link': chunk['uri'],
                'relevant_text': relevant_texts,
                'webpage_content': webpage_content
            })

        # Default to empty list if not provided
        if failed_sites is None:
            failed_sites = []

        output = {
            'metadata': {
                'total_sources': len(sources),
                'scraped_at': datetime.now().isoformat(),
                'failed_scrapes': len(failed_sites)
            },
            'sources': sources,
            'failed_grounded_sites': failed_sites  # Track failed scrapes
        }

        # Build final output with items at top
        final_output = {}

        # Add task_id and test_id at very top if available
        if task_id is not None:
            final_output['task_id'] = task_id
        if test_id:
            final_output['test_id'] = test_id

        # Add query
        if query:
            final_output['query'] = query

        # Add response text if available
        if response_text:
            final_output['responseText'] = response_text

        # Add provider
        final_output['provider'] = provider

        # Always add product map (even if empty, to show that extraction was attempted)
        # An empty list means recommendations were extracted but have no grounded sources
        final_output['productSourceMap'] = product_map if product_map is not None else []

        # Add criteria if available
        if criteria:
            final_output['criteria'] = criteria

        # Add shop_vs_product if available (Shopping domain only)
        if shop_vs_product:
            final_output['shop_vs_product'] = shop_vs_product

        # Add direct grounding if available (raw grounding metadata - provider-agnostic)
        if direct_grounding:
            final_output['direct_grounding'] = direct_grounding

        # Add the rest
        final_output.update(output)

        return final_output

    def process(self, input_file, output_file):
        """Main processing pipeline using Firecrawl for scraping"""
        import time
        pipeline_start = time.time()

        print('[*] Starting Grounding Pipeline\n')
        print('=' * 60 + '\n')

        # Parse grounding file and extract metadata
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        response_text = input_data.get('responseText', '')
        query = input_data.get('query', '')
        criteria = input_data.get('criteria', [])
        task_id = input_data.get('task_id')
        test_id = input_data.get('test_id')
        provider = input_data.get('provider', 'gemini')  # Detect provider
        shop_vs_product = input_data.get('shop_vs_product')  # Shopping domain only

        # Read direct grounding with backward compatibility
        direct_grounding = (input_data.get('direct_grounding') or
                           input_data.get('gemini_direct_grounding') or
                           input_data.get('openai_direct_grounding') or
                           input_data.get('claude_direct_grounding'))

        print(f'Provider: {provider.upper()}\n')

        # Parse chunks and supports from input file
        # chunks = groundingChunks (API-provided)
        # supports = groundingSupports
        chunks, supports = self.parse_json_file(input_file)

        # --- LOGIC MOVED FROM STAGE 1 ---
        # Create recommendation-to-source mapping
        print(f"\nCreating recommendation-to-source mapping...")
        product_map = create_recommendation_source_map(response_text, input_data, query)

        # Extract and map response links
        print(f"\nExtracting response text links...")
        product_names = [p['product_name'] for p in product_map] if product_map else []

        if product_names:
            print(f"   Attempting to extract and map links for {len(product_names)} products...")
            link_to_products = extract_and_map_response_links(response_text, product_names)

            if link_to_products:
                # Get normalized URLs from existing API chunks and build index map
                # Uses top-level normalize_url() function defined at module level
                existing_normalized_urls = {}  # normalized_url -> chunk_index
                # Use 'chunks' list (parsed above) which contains {index, title, uri}
                for idx, chunk in enumerate(chunks):
                    url = chunk.get('uri', '')
                    if url:
                        existing_normalized_urls[normalize_url(url)] = idx

                # Separate new links from existing links
                new_links = {}
                existing_links_to_map = {}  # Links that exist in chunks but need to be mapped to products

                for link, products in link_to_products.items():
                    normalized = normalize_url(link)
                    if normalized not in existing_normalized_urls:
                        # Truly new link
                        new_links[link] = products
                    else:
                        # Link already exists in chunks - need to map it to products
                        existing_chunk_idx = existing_normalized_urls[normalized]
                        existing_links_to_map[existing_chunk_idx] = products

                # First convert lists back to sets for easy updating
                for product in product_map:
                    if isinstance(product['source_indices'], list):
                        product['source_indices'] = set(product['source_indices'])

                # Map existing links (that are already in chunks) to products
                if existing_links_to_map:
                    for chunk_idx, product_names_for_link in existing_links_to_map.items():
                        for product in product_map:
                            if product['product_name'] in product_names_for_link:
                                product['source_indices'].add(chunk_idx)
                    print(f"   ✅ Mapped {len(existing_links_to_map)} existing API chunks to products")

                # Add new links as chunks and map to products
                if new_links:
                    print(f"   ✅ Found {len(new_links)} new links (not in API chunks)")

                    # Add only new links as chunks
                    current_chunk_count = len(chunks)
                    link_to_index = {}

                    for link in new_links.keys():
                        new_idx = len(chunks)
                        link_to_index[link] = new_idx

                        # Add to local chunks list for scraping
                        chunks.append({
                            'index': new_idx,
                            'title': f"Link from response: {link}",
                            'uri': link
                        })

                    # Now add the new link indices to products
                    for product in product_map:
                        for link, product_names_for_link in new_links.items():
                            if product['product_name'] in product_names_for_link:
                                product['source_indices'].add(link_to_index[link])

                    print(f"   Total: {current_chunk_count} API + {len(new_links)} new links = {len(chunks)}")

                # Summary
                if not new_links and not existing_links_to_map:
                    print(f"   No response text links to add")
            else:
                print(f"   No links extracted from response text")
        else:
            print(f"   ⚠️  No products to map links to")

        # Convert sets to lists in productSourceMap before saving
        for product in product_map:
            if isinstance(product.get('source_indices'), set):
                product['source_indices'] = sorted(list(product['source_indices']))

        # --- END LOGIC MOVED FROM STAGE 1 ---

        # Scrape all sources (now includes new links)
        scraped_data, scraping_time, failed_sites = self.scrape_sources(chunks)

        # Create structured output with product map, response text, criteria, and IDs
        processing_start = time.time()
        output = self.create_output(chunks, supports, scraped_data, failed_sites, product_map, response_text, query, criteria, task_id, test_id, direct_grounding, provider, shop_vs_product)
        processing_time = time.time() - processing_start

        # Add timing information
        total_pipeline_time = time.time() - pipeline_start
        output['pipeline_timing'] = {
            'total_seconds': round(total_pipeline_time, 2),
            'scraping_seconds': round(scraping_time, 2),
            'processing_seconds': round(processing_time, 2)
        }

        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f'Saving to: {output_file}')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print('\n' + '=' * 60)
        print('✅ Complete!\n')

        # Summary
        print('Summary:')
        print(f'   Sources: {len(output["sources"])}')
        print(f'   Scraped: {sum(1 for s in output["sources"] if not s["webpage_content"]["error"])}')
        print(f'   Failed: {sum(1 for s in output["sources"] if s["webpage_content"]["error"])}')
        print(f'\nTiming:')
        print(f'   Scraping: {output["pipeline_timing"]["scraping_seconds"]}s')
        print(f'   Processing: {output["pipeline_timing"]["processing_seconds"]}s')
        print(f'   Total: {output["pipeline_timing"]["total_seconds"]}s')

        return output


def main():
    """Main execution"""
    if len(sys.argv) < 3:
        print('❌ Error: Missing required arguments')
        print('\nUsage:')
        print('  python harness/grounding-pipeline.py <input.json> <output.json>')
        print('\nExample:')
        print('  python harness/grounding-pipeline.py 1_grounded_response.json 2_scraped_sources.json')
        print('\nRequired arguments:')
        print('  input.json: Grounded response file (from make-grounded-call.py)')
        print('  output.json: Where to save scraped sources')
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f'❌ Error: Input file not found: {input_file}')
        sys.exit(1)

    processor = GroundingProcessor()
    processor.process(input_file, output_file)


if __name__ == '__main__':
    main()
