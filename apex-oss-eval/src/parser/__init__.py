import logging
from typing import Optional

from .base_parser import BaseParser, ParseResult, ParserCapabilities
from .document_parsing_service import parse_documents, parse_single_document
from .parser_registry import ParserRegistry, parser_registry
from .parsing_cache import ParsingCacheService

logger = logging.getLogger(__name__)


def initialize_parsers(registry: Optional[ParserRegistry] = None) -> None:
    """Register the built-in parsers (currently just Reducto)."""
    registry = registry or parser_registry
    try:
        from .builtin.reducto_parser import ReductoParser

        registry.register(ReductoParser())
    except Exception as exc:
        logger.error("Failed to initialize parsers: %s", exc)


initialize_parsers()

__all__ = [
    "BaseParser",
    "ParserCapabilities",
    "ParseResult",
    "ParserRegistry",
    "parser_registry",
    "ParsingCacheService",
    "parse_documents",
    "parse_single_document",
    "initialize_parsers",
]

