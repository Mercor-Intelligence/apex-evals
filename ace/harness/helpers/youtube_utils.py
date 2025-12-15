#!/usr/bin/env python3
"""
YouTube transcript utilities for grounding pipeline
Handles YouTube video transcript extraction via SearchAPI
"""

import re
import os
import sys
import requests
from typing import Optional, Dict, List
from urllib.parse import urlparse, parse_qs

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, rely on system env vars
    pass

# YouTube Configuration Constants
YOUTUBE_VIDEO_ID_LENGTH = 11
YOUTUBE_API_TIMEOUT = 30  # seconds for SearchAPI requests


def is_youtube_url(url: str) -> bool:
    """
    Check if URL is a YouTube video

    Args:
        url: URL to check

    Returns:
        bool: True if YouTube video URL
    """
    if not url:
        return False

    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/',
        r'(?:https?://)?youtu\.be/',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=',
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats

    Handles:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID

    Args:
        url: YouTube URL

    Returns:
        str: Video ID or None if not found
    """
    if not url:
        return None

    # Pattern 1: youtube.com/watch?v=VIDEO_ID
    match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)

    # Pattern 2: youtu.be/VIDEO_ID
    match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)

    # Pattern 3: youtube.com/embed/VIDEO_ID
    match = re.search(r'youtube\.com/embed/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)

    # Pattern 4: youtube.com/v/VIDEO_ID
    match = re.search(r'youtube\.com/v/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)

    return None


def get_youtube_transcript(video_id: str, api_key: Optional[str] = None) -> Dict:
    """
    Fetch YouTube transcript using SearchAPI

    Args:
        video_id: YouTube video ID
        api_key: SearchAPI API key (defaults to env variable)

    Returns:
        dict: {
            'success': bool,
            'transcript': str,  # Full transcript text
            'segments': list,   # Raw segments with timing
            'language': str,    # Detected language
            'error': str or None
        }
    """
    if not api_key:
        # Load from environment variable
        api_key = os.getenv('SEARCHAPI_API_KEY')

    if not api_key:
        return {
            'success': False,
            'transcript': '',
            'segments': [],
            'language': None,
            'error': 'No SearchAPI API key provided'
        }

    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "youtube_transcripts",
            "video_id": video_id,
            "api_key": api_key
        }

        response = requests.get(url, params=params, timeout=YOUTUBE_API_TIMEOUT)
        response.raise_for_status()

        data = response.json()

        # Check if transcripts exist
        if "transcripts" not in data or not data["transcripts"]:
            return {
                'success': False,
                'transcript': '',
                'segments': [],
                'language': None,
                'error': 'No transcripts available for this video'
            }

        # Extract text from segments
        segments = data["transcripts"]
        transcript_text = " ".join([segment["text"] for segment in segments])

        # Get language if available
        language = None
        if "available_languages" in data and data["available_languages"]:
            language = data["available_languages"][0].get("lang", "en")

        return {
            'success': True,
            'transcript': transcript_text,
            'segments': segments,
            'language': language,
            'error': None
        }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'transcript': '',
            'segments': [],
            'language': None,
            'error': f'API request failed: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'transcript': '',
            'segments': [],
            'language': None,
            'error': f'Unexpected error: {str(e)}'
        }


def format_transcript_markdown(transcript: str, video_url: str, title: Optional[str] = None) -> str:
    """
    Format transcript as markdown section

    Args:
        transcript: Full transcript text
        video_url: Original YouTube URL
        title: Video title (optional)

    Returns:
        str: Formatted markdown
    """
    markdown = "## Video Transcript\n\n"

    if title:
        markdown += f"**Video:** {title}\n\n"

    markdown += f"**URL:** {video_url}\n\n"
    markdown += "### Transcript Content\n\n"
    markdown += transcript
    markdown += "\n\n---\n\n"

    return markdown


# Quick test
if __name__ == '__main__':
    # Test with video URL (or provide as argument)
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=NiSIYq06nt8"

    print("Testing YouTube utilities...\n")

    print(f"URL: {test_url}")
    print(f"Is YouTube URL: {is_youtube_url(test_url)}")

    video_id = extract_video_id(test_url)
    print(f"Video ID: {video_id}")

    if video_id:
        print("\nFetching transcript...")
        result = get_youtube_transcript(video_id)

        if result['success']:
            print(f"✅ Success!")
            print(f"Transcript length: {len(result['transcript'])} chars")
            print(f"Language: {result['language']}")
            print(f"\nFirst 200 chars: {result['transcript'][:200]}...")
        else:
            print(f"❌ Error: {result['error']}")

