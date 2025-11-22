import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .file_hash_utils import calculate_file_hash

logger = logging.getLogger(__name__)


class ParsingCacheService:
    """File-based cache for parsed attachments."""

    def __init__(self, base_path: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.base_path = Path(base_path or Path.home() / ".oss_eval_parser_cache")
        if self.enabled:
            self.base_path.mkdir(parents=True, exist_ok=True)

    def _parsed_path(self, parser_name: str, file_hash: str) -> Path:
        return self.base_path / parser_name / "parsed" / f"{file_hash}.txt"

    def _original_path(self, parser_name: str, file_hash: str, filename: str) -> Path:
        suffix = Path(filename).suffix or ".bin"
        return self.base_path / parser_name / "original" / f"{file_hash}{suffix}"

    def _meta_path(self, parser_name: str, file_hash: str) -> Path:
        return self.base_path / parser_name / f"{file_hash}.json"

    async def get_cached(
        self,
        content: bytes,
        filename: str,
        parser_name: str,
    ) -> Optional[str]:
        """Returns cached text."""
        if not self.enabled:
            return None

        file_hash = calculate_file_hash(content)
        parsed_path = self._parsed_path(parser_name, file_hash)

        if not parsed_path.exists():
            return None

        text = await asyncio.to_thread(parsed_path.read_text, encoding="utf-8")
        logger.debug("Cache hit for %s via %s", filename, parser_name)
        return text

    async def cache_result(
        self,
        content: bytes,
        filename: str,
        parsed_text: str,
        parser_name: str,
        url: str = "",
    ) -> bool:
        """Persists parsed text."""
        if not self.enabled or not parsed_text:
            return False

        file_hash = calculate_file_hash(content)
        parsed_path = self._parsed_path(parser_name, file_hash)
        original_path = self._original_path(parser_name, file_hash, filename)
        meta_path = self._meta_path(parser_name, file_hash)

        def _write_files():
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.parent.mkdir(parents=True, exist_ok=True)

            parsed_path.write_text(parsed_text, encoding="utf-8")
            original_path.write_bytes(content)

            metadata = {
                "filename": filename,
                "file_hash": file_hash,
                "parser": parser_name,
                "url": url,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        try:
            await asyncio.to_thread(_write_files)
            logger.debug("Cached %s via %s", filename, parser_name)
            return True
        except Exception as exc:
            logger.warning("Failed to cache %s: %s", filename, exc)
            return False

    def clear_cache(self) -> None:
        """Clears all cached files."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            logger.info("Cleared parsing cache at %s", self.base_path)
        else:
            logger.info("Cache directory does not exist: %s", self.base_path)
        
        if self.enabled:
            self.base_path.mkdir(parents=True, exist_ok=True)


