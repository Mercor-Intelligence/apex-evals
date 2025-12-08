#!/usr/bin/env python3
"""Unit tests for pipeline/runner.py"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

import pytest
from runner import (
    deduplicate_urls,
    enhance_product_map,
    extract_direct_grounding,
    determine_failure_step,
    retry_with_backoff,
    build_common_task_data,
)


class TestDeduplicateUrls:
    def test_removes_duplicate_urls(self):
        """Removes duplicate URLs from list"""
        urls = [
            'https://example.com/page',
            'https://example.com/page',
            'https://other.com/page',
        ]
        result = deduplicate_urls(urls)
        assert len(result) == 2

    def test_removes_utm_params_for_deduplication(self):
        """URLs differing only by UTM params are considered duplicates"""
        urls = [
            'https://example.com/page',
            'https://example.com/page?utm_source=google',
        ]
        result = deduplicate_urls(urls)
        assert len(result) == 1

    def test_removes_empty_urls(self):
        """Empty URLs are removed"""
        urls = ['https://example.com', '', 'https://other.com']
        result = deduplicate_urls(urls)
        assert '' not in result

    def test_handles_malformed_urls(self):
        """Malformed URLs are kept as-is"""
        urls = ['not-a-url', 'https://example.com']
        result = deduplicate_urls(urls)
        assert 'not-a-url' in result

    def test_preserves_different_query_params(self):
        """URLs with different non-UTM query params are not deduplicated"""
        urls = [
            'https://example.com/page?id=1',
            'https://example.com/page?id=2',
        ]
        result = deduplicate_urls(urls)
        assert len(result) == 2


class TestEnhanceProductMap:
    def test_adds_urls_from_source_metadata(self):
        """Enhances product map with URLs from source metadata"""
        product_map = [
            {'product_name': 'Product A', 'source_indices': [0, 1]}
        ]
        source_metadata = [
            {'source_number': 1, 'source_link': 'https://example.com/1'},
            {'source_number': 2, 'source_link': 'https://example.com/2'},
        ]

        result = enhance_product_map(product_map, source_metadata)

        assert len(result) == 1
        assert result[0]['product_name'] == 'Product A'
        assert 'https://example.com/1' in result[0]['source_urls']
        assert 'https://example.com/2' in result[0]['source_urls']

    def test_handles_missing_source(self):
        """Handles source indices that don't exist in metadata"""
        product_map = [
            {'product_name': 'Product A', 'source_indices': [0, 99]}
        ]
        source_metadata = [
            {'source_number': 1, 'source_link': 'https://example.com/1'},
        ]

        result = enhance_product_map(product_map, source_metadata)

        assert len(result[0]['source_urls']) == 1

    def test_handles_empty_product_map(self):
        """Returns empty list for empty product map"""
        result = enhance_product_map([], [])
        assert result == []

    def test_preserves_source_indices(self):
        """Preserves original source indices in output"""
        product_map = [
            {'product_name': 'Product A', 'source_indices': [0, 1, 2]}
        ]

        result = enhance_product_map(product_map, [])

        assert result[0]['source_indices'] == [0, 1, 2]


class TestExtractDirectGrounding:
    def test_extracts_direct_grounding(self):
        """Extracts direct_grounding key"""
        data = {'direct_grounding': {'chunks': []}}
        assert extract_direct_grounding(data) == {'chunks': []}

    def test_extracts_gemini_direct_grounding(self):
        """Falls back to gemini_direct_grounding"""
        data = {'gemini_direct_grounding': {'chunks': []}}
        assert extract_direct_grounding(data) == {'chunks': []}

    def test_extracts_openai_direct_grounding(self):
        """Falls back to openai_direct_grounding"""
        data = {'openai_direct_grounding': {'chunks': []}}
        assert extract_direct_grounding(data) == {'chunks': []}

    def test_extracts_claude_direct_grounding(self):
        """Falls back to claude_direct_grounding"""
        data = {'claude_direct_grounding': {'chunks': []}}
        assert extract_direct_grounding(data) == {'chunks': []}

    def test_returns_none_when_missing(self):
        """Returns None when no grounding data present"""
        data = {'other_key': 'value'}
        assert extract_direct_grounding(data) is None

    def test_prefers_direct_grounding_over_provider_specific(self):
        """Prefers direct_grounding over provider-specific keys"""
        data = {
            'direct_grounding': {'preferred': True},
            'gemini_direct_grounding': {'preferred': False},
        }
        assert extract_direct_grounding(data) == {'preferred': True}


class TestDetermineFailureStep:
    def test_score_1_returns_passed(self):
        """Score 1 returns 'Passed'"""
        assert determine_failure_step(1) == "Passed"

    def test_score_0_returns_stage_1(self):
        """Score 0 returns 'Stage 1'"""
        assert determine_failure_step(0) == "Stage 1"

    def test_score_neg1_returns_stage_2(self):
        """Score -1 returns 'Stage 2'"""
        assert determine_failure_step(-1) == "Stage 2"

    def test_unknown_score_returns_unknown(self):
        """Unknown scores return 'Unknown'"""
        assert determine_failure_step(5) == "Unknown"
        assert determine_failure_step(-5) == "Unknown"
        assert determine_failure_step(None) == "Unknown"


class TestRetryWithBackoff:
    def test_returns_result_on_success(self):
        """Returns function result on success"""
        result = retry_with_backoff(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self):
        """Retries function on failure"""
        call_count = [0]

        def failing_then_succeeding():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("fail")
            return "success"

        result = retry_with_backoff(failing_then_succeeding, max_attempts=3)
        assert result == "success"
        assert call_count[0] == 3

    def test_raises_after_max_attempts(self):
        """Raises exception after max attempts"""
        def always_fails():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(always_fails, max_attempts=2)

    def test_uses_exponential_backoff(self):
        """Uses exponential backoff between retries"""
        call_times = []

        def record_time_and_fail():
            call_times.append(time.time())
            raise Exception("fail")

        import time
        with pytest.raises(Exception):
            retry_with_backoff(record_time_and_fail, max_attempts=3)

        # Check that delays increase (exponential)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1


class TestBuildCommonTaskData:
    def test_builds_task_data_with_suffix(self):
        """Builds task data with numbered suffix"""
        scraped_data = {
            'task_id': 123,
            'query': 'test query',
            'responseText': 'response',
            'criteria': [{'id': 1}],
            'failed_grounded_sites': [],
        }

        result = build_common_task_data(
            scraped_data,
            enhanced_product_map=[],
            grounding_source_meta_data=[],
            suffix=' - 1'
        )

        assert result['Task ID'] == 123
        assert result['Specified Prompt'] == 'test query'
        assert result['Response Text - 1'] == 'response'
        assert result['Criteria List'] == [{'id': 1}]

    def test_includes_direct_grounding_when_present(self):
        """Includes direct grounding in output when present"""
        scraped_data = {
            'task_id': 123,
            'direct_grounding': {'chunks': [1, 2, 3]},
        }

        result = build_common_task_data(scraped_data, [], [], ' - 1')

        assert result['Direct Grounding - 1'] == {'chunks': [1, 2, 3]}

    def test_includes_shop_vs_product_for_shopping(self):
        """Includes shop_vs_product when present (Shopping domain)"""
        scraped_data = {
            'task_id': 123,
            'shop_vs_product': 'Product',
        }

        result = build_common_task_data(scraped_data, [], [], ' - 1')

        assert result['Shop vs. Product'] == 'Product'

    def test_omits_shop_vs_product_when_missing(self):
        """Omits shop_vs_product when not in scraped data"""
        scraped_data = {'task_id': 123}

        result = build_common_task_data(scraped_data, [], [], ' - 1')

        assert 'Shop vs. Product' not in result
