#!/usr/bin/env python3
"""
Reddit utilities for grounding pipeline
Handles Reddit post scraping via Reddit JSON API
"""

import json
import os
import re
import ssl
import sys
import urllib.request
from html import unescape
from typing import Optional, Dict

# Add project root to path FIRST (2 levels up from helpers/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from configs.logging_config import setup_logging
logger = setup_logging(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, rely on system env vars
    pass

# Reddit Configuration Constants
REDDIT_API_TIMEOUT = 30  # seconds for API requests


def is_reddit_url(url: str) -> bool:
    """
    Check if URL is a Reddit post

    Args:
        url: URL to check

    Returns:
        bool: True if Reddit post URL
    """
    if not url:
        return False

    reddit_patterns = [
        r'(?:https?://)?(?:www\.)?reddit\.com/',
        r'(?:https?://)?(?:old\.)?reddit\.com/',
        r'(?:https?://)?(?:new\.)?reddit\.com/',
        r'(?:https?://)?(?:m\.)?reddit\.com/',
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in reddit_patterns)


def fetch_reddit_json(url: str) -> Dict:
    """
    Fetch Reddit post data as JSON by appending .json to URL

    Args:
        url: Reddit post URL (may include query params like ?utm_source=openai)

    Returns:
        dict: {
            'success': bool,
            'data': dict or None,  # Raw JSON data
            'error': str or None
        }
    """
    try:
        # Add .json if not already present
        if not url.endswith('.json'):
            # Remove query parameters before adding .json
            # Reddit's JSON API doesn't use query params, and tracking params (utm_source, etc.)
            # from providers like OpenAI will break the .json endpoint if not removed
            if '?' in url:
                base_url = url.split('?')[0]
                json_url = base_url.rstrip('/') + '/.json'
            else:
                json_url = url.rstrip('/') + '/.json'
        else:
            json_url = url

        # Create SSL context
        # NOTE: SSL verification must be disabled for Reddit JSON API
        # Reddit returns 403 Forbidden when SSL verification is enabled
        # This is a known Reddit API requirement, not a security flaw
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create request with User-Agent header (Reddit requires this)
        req = urllib.request.Request(
            json_url,
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        )

        # Fetch the JSON data
        with urllib.request.urlopen(req, context=ssl_context, timeout=REDDIT_API_TIMEOUT) as response:
            json_data = json.loads(response.read())

        return {
            'success': True,
            'data': json_data,
            'error': None
        }

    except urllib.error.HTTPError as e:
        return {
            'success': False,
            'data': None,
            'error': f'HTTP error {e.code}: {e.reason}'
        }
    except urllib.error.URLError as e:
        return {
            'success': False,
            'data': None,
            'error': f'URL error: {str(e.reason)}'
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'data': None,
            'error': f'JSON decode error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': f'Unexpected error: {str(e)}'
        }


def clean_reddit_json_to_markdown(json_data):
    """
    Cleans Reddit JSON (from link.json) and formats it into Markdown.

    Args:
        json_data: Raw JSON data from Reddit API (dict or string)

    Returns:
        str: Markdown formatted Reddit post and comments
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    markdown_output = []

    def process_comment(item, depth=0):
        """Recursively process a comment and its replies."""
        author = item.get("author", "[unknown]")
        body = unescape(re.sub(r'<.*?>', '', item.get("body", "")))
        indent = "    " * depth
        markdown_output.append(f"{indent}- **u/{author}:** {body}")

        # Process replies recursively
        replies = item.get("replies", "")
        if replies and isinstance(replies, dict):
            if replies.get("kind") == "Listing":
                for reply_child in replies["data"].get("children", []):
                    if reply_child.get("kind") == "t1":
                        process_comment(reply_child.get("data", {}), depth + 1)

    for listing in data:
        if listing.get("kind") != "Listing":
            continue
        for child in listing["data"].get("children", []):
            kind = child.get("kind")
            item = child.get("data", {})

            # --- POST ---
            if kind == "t3":  # Submission
                title = item.get("title", "")
                author = item.get("author", "[unknown]")
                selftext = unescape(re.sub(r'<.*?>', '', item.get("selftext", "")))
                url = item.get("url", "")
                permalink = f"https://www.reddit.com{item.get('permalink', '')}"
                subreddit = item.get("subreddit_name_prefixed", "")

                markdown_output.append(f"# {title}\n")
                markdown_output.append(f"- **Author:** u/{author}")
                markdown_output.append(f"- **Subreddit:** {subreddit}")
                markdown_output.append(f"- **URL:** [{url}]({url})")
                markdown_output.append(f"- **Discussion:** [{permalink}]({permalink})\n")
                if selftext:
                    markdown_output.append(f"**Post:**\n\n{selftext}\n")

            # --- COMMENTS ---
            elif kind == "t1":  # Comment
                process_comment(item)

    return "\n".join(markdown_output)


def format_reddit_markdown(markdown: str, url: str, title: Optional[str] = None) -> str:
    """
    Format Reddit content as markdown section (similar to YouTube format)

    Args:
        markdown: Markdown content from clean_reddit_json_to_markdown
        url: Original Reddit URL
        title: Post title (optional)

    Returns:
        str: Formatted markdown with header
    """
    formatted = "## Reddit Discussion\n\n"

    if title:
        formatted += f"**Post:** {title}\n\n"

    formatted += f"**URL:** {url}\n\n"
    formatted += "### Discussion Content\n\n"
    formatted += markdown
    formatted += "\n\n---\n\n"

    return formatted


def get_reddit_content(url: str) -> Dict:
    """
    Main function to fetch and convert Reddit post to markdown

    Args:
        url: Reddit post URL

    Returns:
        dict: {
            'success': bool,
            'markdown': str,  # Formatted markdown
            'title': str or None,
            'error': str or None
        }
    """
    # Fetch JSON data
    result = fetch_reddit_json(url)

    if not result['success']:
        return {
            'success': False,
            'markdown': '',
            'title': None,
            'error': result['error']
        }

    try:
        # Convert to markdown
        markdown = clean_reddit_json_to_markdown(result['data'])

        # Extract title from first listing
        title = None
        if result['data'] and len(result['data']) > 0:
            first_listing = result['data'][0]
            if first_listing.get('kind') == 'Listing':
                children = first_listing.get('data', {}).get('children', [])
                if children and children[0].get('kind') == 't3':
                    title = children[0].get('data', {}).get('title', '')

        return {
            'success': True,
            'markdown': markdown,
            'title': title,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'markdown': '',
            'title': None,
            'error': f'Error converting to markdown: {str(e)}'
        }


# Quick test
if __name__ == '__main__':
    import sys

    # Test with Reddit URL (or provide as argument)
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.reddit.com/r/singing/comments/173qi20/how_do_you_easily_develop_vibrato/"

    print(f"Testing Reddit utilities...\n")
    print(f"URL: {test_url}")
    print(f"Is Reddit URL: {is_reddit_url(test_url)}\n")

    if is_reddit_url(test_url):
        print("Fetching Reddit content...")
        result = get_reddit_content(test_url)

        if result['success']:
            print(f"✅ Success!")
            print(f"Title: {result['title']}")
            print(f"Markdown length: {len(result['markdown'])} chars")
            print(f"\nFirst 500 chars:\n{result['markdown'][:500]}...")

            # Format and display
            formatted = format_reddit_markdown(result['markdown'], test_url, result['title'])
            print(f"\nFormatted markdown length: {len(formatted)} chars")

            # Optionally save to file
            with open("reddit_output.md", "w", encoding="utf-8") as f:
                f.write(formatted)
            print(f"\n✅ Output saved to reddit_output.md")
        else:
            print(f"❌ Error: {result['error']}")

