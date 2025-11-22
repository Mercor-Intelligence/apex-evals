"""
Built-in document parsers.

This package contains the default parser implementations that come
with the SDK. Additional parsers can be created by inheriting from
BaseParser and registered with the parser registry.
"""

from .reducto_parser import ReductoParser

__all__ = ["ReductoParser"]

