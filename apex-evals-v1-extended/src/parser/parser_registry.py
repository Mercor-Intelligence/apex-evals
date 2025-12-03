"""
Central parser registry.
"""

import logging
from typing import Dict, List, Optional
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for parsers."""
    
    def __init__(self):
        """Initializes registry."""
        self._parsers: Dict[str, BaseParser] = {}
    
    def register(self, parser: BaseParser) -> None:
        """Registers parser."""
        self._parsers[parser.name] = parser
        logger.info(f"Registered parser: {parser.name}")
    
    def unregister(self, parser_name: str) -> bool:
        """Removes parser."""
        if parser_name in self._parsers:
            del self._parsers[parser_name]
            logger.info(f"Unregistered parser: {parser_name}")
            return True
        return False
    
    def get_parser(self, parser_name: str) -> Optional[BaseParser]:
        """Gets parser by name."""
        return self._parsers.get(parser_name)
    
    def list_parsers(self) -> List[Dict[str, any]]:
        """Lists parsers and capabilities."""
        parsers = []
        
        for parser in self._parsers.values():
            parsers.append({
                'name': parser.name,
                'capabilities': {
                    'supported_extensions': parser.capabilities.supported_extensions,
                    'supported_mime_types': parser.capabilities.supported_mime_types,
                    'max_file_size': parser.capabilities.max_file_size,
                    'supports_ocr': parser.capabilities.supports_ocr,
                    'supports_tables': parser.capabilities.supports_tables,
                    'supports_images': parser.capabilities.supports_images,
                    'requires_api_key': parser.capabilities.requires_api_key,
                }
            })
        
        return parsers
    
    def get_parser_names(self) -> List[str]:
        """Gets registered parser names."""
        return list(self._parsers.keys())
    
    def find_parser_for_file(
        self, 
        filename: str, 
        mime_type: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> Optional[BaseParser]:
        """Finds suitable parser."""
        for parser in self._parsers.values():
            if parser.can_parse(filename, mime_type, file_size):
                return parser
        return None


# Global registry instance
parser_registry = ParserRegistry()

