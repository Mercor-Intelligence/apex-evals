import asyncio
import json
import logging
import os
import tempfile
import time
from importlib import import_module
from typing import Any, Dict, Optional, Sequence
from urllib import request as urllib_request

from ..base_parser import BaseParser, ParseResult, ParserCapabilities

logger = logging.getLogger(__name__)


class ReductoParser(BaseParser):
    """Reducto AI parser."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client: Optional[object] = None
        self._client_initialized = False

    @property
    def name(self) -> str:
        return "reducto"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supported_extensions=[
                "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx",
                "csv", "txt", "html", "htm", "png", "jpeg", "jpg",
                "gif", "webp",
            ],
            supported_mime_types=[],
            supports_ocr=True,
            supports_tables=True,
            supports_images=True,
            requires_api_key=True,
        )

    @property
    def client(self) -> Optional[object]:
        if not self._client_initialized:
            self._client = self._create_client()
            self._client_initialized = True
        return self._client

    def _create_client(self) -> Optional[object]:
        try:
            sdk = import_module("reducto")
        except ImportError:
            logger.warning("reductoai package is not installed")
            return None

        api_key = os.getenv("REDUCTO_API_KEY")
        if not api_key:
            logger.warning("REDUCTO_API_KEY not provided")
            return None

        try:
            client = sdk.Reducto(api_key=api_key)
            logger.info("Reducto client created successfully")
            return client
        except Exception as exc:  # pragma: no cover - network dependency
            logger.error("Failed to create Reducto client: %s", exc)
            return None

    async def validate(self) -> bool:
        """Checks service availability."""
        return self.client is not None

    def _detect_file_type(self, content: bytes, filename: str = "") -> str:
        """Detects file type."""
        if content.startswith(b"%PDF"):
            return "pdf"
        if content.startswith(b"\x89PNG"):
            return "png"
        if content.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if content.startswith(b"GIF8"):
            return "gif"
        if content.startswith(b"RIFF") and b"WEBP" in content[:20]:
            return "webp"

        if filename:
            ext = filename.lower().split(".")[-1]
            if ext in [
                "pdf", "png", "jpg", "jpeg", "gif", "webp", "doc", "docx",
                "ppt", "pptx", "xls", "xlsx", "csv", "txt", "json", "xml",
                "html", "htm", "eml",
            ]:
                return ext

        return "unknown"

    def _supports_file_type(self, file_type: str) -> bool:
        """Checks type support."""
        supported_types = {
            "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx",
            "txt", "csv", "xml", "html", "htm", "png", "jpeg",
            "jpg", "gif", "webp",
        }
        return file_type.lower() in supported_types

    def _extract_text_from_result(self, result: object) -> str:
        """Extracts text from result."""
        try:
            if hasattr(result, "result") and result.result:
                parse_result = result.result

                if hasattr(parse_result, "url") and parse_result.url:
                    return self._fetch_text_from_url(parse_result.url)

                if hasattr(parse_result, "chunks") and parse_result.chunks:
                    texts = [
                        str(getattr(chunk, "content", "")).strip()
                        for chunk in parse_result.chunks
                        if getattr(chunk, "content", None)
                    ]
                    if texts:
                        return "\n\n".join(texts)

            if hasattr(result, "chunks") and result.chunks:
                texts = [
                    str(getattr(chunk, "content", "")).strip()
                    for chunk in result.chunks
                    if getattr(chunk, "content", None)
                ]
                if texts:
                    return "\n\n".join(texts)

            if hasattr(result, "content") and result.content:
                return str(result.content)

            logger.debug(
                "Reducto result structure: %s with attributes: %s",
                type(result),
                [attr for attr in dir(result) if not attr.startswith("_")],
            )
            return ""
        except Exception as exc:
            logger.error("Reducto error - Failed to extract text from result: %s", exc)
            return ""

    def _text_from_chunks(self, chunks: Optional[Sequence[Dict[str, Any]]]) -> str:
        """Joins chunk content."""
        if not isinstance(chunks, list):
            return ""
        texts = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content")
            if not content:
                continue
            content_str = str(content).strip()
            if content_str:
                texts.append(content_str)
        return "\n\n".join(texts) if texts else ""

    def _fetch_text_from_url(self, url: str) -> str:
        """Fetches parsed text from URL."""
        logger.debug("Fetching Reducto parsed content from URL: %s", url[:100])
        try:
            with urllib_request.urlopen(url, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            logger.error("Reducto error - Failed to download output from URL: %s", exc)
            return ""

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Reducto error - Invalid JSON in response: %s", exc)
            return ""

        if not isinstance(payload, dict):
            logger.error("Reducto error - Unexpected payload type (expected dict, got %s)", type(payload).__name__)
            return ""

        chunk_text = self._text_from_chunks(payload.get("chunks"))
        if chunk_text:
            logger.debug("Successfully fetched %d characters from Reducto URL", len(chunk_text))
            return chunk_text

        for key in ["text", "content", "parsed_text", "output"]:
            value = payload.get(key)
            if value:
                text_value = str(value).strip()
                if text_value:
                    logger.debug(
                        "Successfully fetched %d characters from Reducto URL (key: %s)",
                        len(text_value),
                        key,
                    )
                    return text_value

        logger.error("Reducto error - No extractable content found in response. Keys: %s", list(payload.keys()))
        return ""

    async def parse(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        """Parses document."""
        start_time = time.perf_counter()

        if not self.client:
            return ParseResult(
                success=False,
                content="",
                metadata={},
                error="Reducto client not available (check REDUCTO_API_KEY)",
                parser_used=self.name,
                processing_time=time.perf_counter() - start_time,
            )

        file_type = self._detect_file_type(content, filename)
        if not self._supports_file_type(file_type):
            return ParseResult(
                success=False,
                content="",
                metadata={"file_type": file_type},
                error=f"Reducto does not support file type: {file_type}",
                parser_used=self.name,
                processing_time=time.perf_counter() - start_time,
            )

        logger.info("Parsing %s (type: %s) with Reducto", filename, file_type)

        suffix = ""
        if filename and "." in filename:
            suffix = "." + filename.split(".")[-1]

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            temp_file.write(content)
            temp_file.close()

            parse_config = self.config.get("parse_config") if self.config else None
            parse_kwargs: Dict[str, Any] = {}
            if isinstance(parse_config, dict):
                parse_kwargs.update(parse_config)

            def _sync_reducto_parse() -> Optional[str]:
                try:
                    logger.info("Reducto: Uploading %s (%d bytes)...", filename, len(content))
                    try:
                        with open(temp_file.name, "rb") as handle:
                            upload_result = self.client.upload(file=handle)
                            upload_id = getattr(upload_result, "file_id", None)
                    except Exception as upload_exc:
                        logger.error("Reducto error - Upload failed for %s: %s", filename, upload_exc)
                        raise
                    
                    logger.info("Reducto: Upload complete for %s (file_id: %s)", filename, upload_id)

                    if not upload_id:
                        raise RuntimeError("Reducto upload did not return file_id")

                    logger.info("Reducto: Parsing %s...", filename)
                    try:
                        result = self.client.parse.run(input=upload_id, **parse_kwargs)
                    except Exception as parse_exc:
                        logger.error("Reducto error - Parse API call failed for %s: %s", filename, parse_exc)
                        raise
                    
                    logger.info("Reducto: Parse complete for %s, extracting text...", filename)
                    extracted_text = self._extract_text_from_result(result)

                    if not extracted_text or not extracted_text.strip():
                        logger.error("Reducto error - Empty result returned for %s", filename)

                    return extracted_text
                except Exception as exc:
                    error_str = str(exc)
                    if "Invalid access token" in error_str or "401" in error_str:
                        logger.error("Reducto error - Authentication failed: %s", exc)
                        raise exc
                    
                    logger.error("Reducto error - Parsing failed for %s: %s", filename, exc)
                    return None

            text = await asyncio.to_thread(_sync_reducto_parse)
        finally:
            try:
                os.unlink(temp_file.name)
            except OSError as exc:
                logger.debug("Failed to clean up temp file: %s", exc)

        if text and text.strip():
            return ParseResult(
                success=True,
                content=text,
                metadata={
                    "parser": self.name,
                    "file_type": file_type,
                    "file_size": len(content),
                    "output_length": len(text),
                },
                parser_used=self.name,
                processing_time=time.perf_counter() - start_time,
            )

        return ParseResult(
            success=False,
            content="",
            metadata={"file_type": file_type},
            error="Reducto returned empty result",
            parser_used=self.name,
            processing_time=time.perf_counter() - start_time,
        )

