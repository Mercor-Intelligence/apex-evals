"""
Base parser abstractions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParserCapabilities:
    """Parser capabilities."""
    supported_extensions: List[str]
    supported_mime_types: List[str] = field(default_factory=list)
    max_file_size: Optional[int] = None
    supports_ocr: bool = False
    supports_tables: bool = False
    supports_images: bool = False
    requires_api_key: bool = False


@dataclass
class ParseResult:
    """Parsing result."""
    success: bool
    content: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    parser_used: str = ""
    processing_time: float = 0.0


class BaseParser(ABC):
    """Abstract base parser."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes parser."""
        self.config = config or {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Parser identifier."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> ParserCapabilities:
        """Parser capabilities."""
        pass
    
    @abstractmethod
    async def parse(
        self, 
        content: bytes, 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParseResult:
        """Parses content."""
        pass
    
    def can_parse(
        self, 
        filename: str, 
        mime_type: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> bool:
        """Checks if file is supported."""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if ext and ext not in self.capabilities.supported_extensions:
            return False
        
        if mime_type and self.capabilities.supported_mime_types:
            if mime_type not in self.capabilities.supported_mime_types:
                return False
        
        if file_size and self.capabilities.max_file_size:
            if file_size > self.capabilities.max_file_size:
                return False
        
        return True
    
    async def validate(self) -> bool:
        """Validates parser readiness."""
        return True

