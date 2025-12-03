import asyncio
import logging
from importlib import import_module
from typing import Any, Dict, List, Optional

import aiohttp
ClientSession = object  # pragma: no cover - fallback for type hints
from errors import UserInputError
from .base_parser import ParseResult
from .parser_registry import ParserRegistry, parser_registry
from .parsing_cache import ParsingCacheService

logger = logging.getLogger(__name__)

MAX_PARALLEL_DOWNLOADS = 5


async def parse_documents(
    attachments: List[Dict[str, str]],
    parser_name: str = "reducto",
    use_cache: bool = True,
    registry: Optional[ParserRegistry] = None,
    cache_service: Optional[ParsingCacheService] = None,
) -> str:
    """Download, parse, cache, and format attachment text."""
    if not attachments:
        return ""

    if aiohttp is None:
        raise UserInputError(
            title="Missing dependency",
            summary="Install aiohttp to enable attachment downloading.",
            context={"dependency": "aiohttp"},
        )

    registry = registry or parser_registry
    parser = _get_parser_or_raise(registry, parser_name)

    if not await parser.validate():
        raise UserInputError(
            title=f"Parser '{parser.name}' failed validation",
            summary="Check API keys and parser dependencies before retrying.",
            context={"parser": parser.name},
        )

    cache_service = cache_service or ParsingCacheService(enabled=use_cache)
    cache_enabled = use_cache and cache_service.enabled

    logger.info("Downloading %d attachment(s)...", len(attachments))
    downloads = await _download_attachments(attachments, MAX_PARALLEL_DOWNLOADS)
    
    if not downloads:
        logger.warning("No attachments were successfully downloaded")
        return ""
    
    if len(downloads) < len(attachments):
        logger.warning("Downloaded %d/%d attachments (some failed)", len(downloads), len(attachments))
    else:
        logger.info("Downloaded all %d attachment(s) successfully", len(downloads))

    downloads.sort(key=lambda item: item["size"])
    sections: List[str] = []

    for idx, file_data in enumerate(downloads, 1):
        filename = file_data["filename"]
        url = file_data["url"]
        content = file_data["content"]

        logger.info("Processing attachment %d/%d: %s", idx, len(downloads), filename)

        cached_text = None
        if cache_enabled:
            cached_text = await cache_service.get_cached(content, filename, parser.name)

        if cached_text:
            logger.info("  ✓ Loaded from cache (%d chars)", len(cached_text))
            sections.append(_format_section(filename, cached_text))
            continue

        try:
            result = await parser.parse(content=content, filename=filename, metadata={"url": url})
        except Exception as exc:
            logger.error("Parsing failed for %s: %s", filename, exc)
            continue

        text = (result.content or "").strip()
        if not result.success or not text:
            error_msg = f"No text extracted for {filename}"
            if result.error:
                error_msg += f" - Error: {result.error}"
            logger.warning(error_msg)
            continue

        logger.info("  ✓ Parsed successfully (%d chars)", len(text))
        sections.append(_format_section(filename, text))

        if cache_enabled:
            await cache_service.cache_result(
                content=content,
                filename=filename,
                parsed_text=result.content,
                parser_name=parser.name,
                url=url,
            )

    if sections:
        logger.info("Successfully parsed %d/%d attachment(s)", len(sections), len(downloads))
    else:
        logger.warning("No attachments were successfully parsed")
    
    return "\n\n".join(sections)


async def parse_single_document(
    content: bytes,
    filename: str,
    parser_name: str = "reducto",
    use_cache: bool = True,
    registry: Optional[ParserRegistry] = None,
    cache_service: Optional[ParsingCacheService] = None,
    url: str = "",
) -> ParseResult:
    """Parses single document."""
    registry = registry or parser_registry
    parser = _get_parser_or_raise(registry, parser_name)

    if not await parser.validate():
        raise UserInputError(
            title=f"Parser '{parser.name}' failed validation",
            summary="Check API keys and parser dependencies before retrying.",
            context={"parser": parser.name},
        )

    cache_service = cache_service or ParsingCacheService(enabled=use_cache)
    cache_enabled = use_cache and cache_service.enabled

    if cache_enabled:
        cached_text = await cache_service.get_cached(content, filename, parser.name)
        if cached_text:
            return ParseResult(
                success=True,
                content=cached_text,
                metadata={"cached": True},
                parser_used=parser.name,
                processing_time=0.0,
            )

    result = await parser.parse(content=content, filename=filename, metadata={"url": url})

    if cache_enabled and result.success and (result.content or "").strip():
        await cache_service.cache_result(
            content=content,
            filename=filename,
            parsed_text=result.content,
            parser_name=parser.name,
            url=url,
        )

    return result


def _format_section(filename: str, text: str) -> str:
    return f"=== {filename} ===\n{text.strip()}"


def _get_parser_or_raise(registry: ParserRegistry, parser_name: str):
    parser = registry.get_parser(parser_name)
    if parser:
        return parser

    available = ", ".join(registry.get_parser_names()) or "none"
    raise UserInputError(
        title="Unknown parser selected",
        summary=f"Parser '{parser_name}' is not registered. Available: {available}.",
        context={"requested_parser": parser_name},
    )


async def _download_attachments(
    attachments: List[Dict[str, str]],
    max_concurrency: int,
) -> List[Dict[str, Any]]:
    if not attachments:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async with aiohttp.ClientSession() as session:
        tasks = [_download_single(att, session, semaphore) for att in attachments]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    return [result for result in results if result]


async def _download_single(
    attachment: Dict[str, str],
    session: ClientSession,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    filename = attachment.get("filename")
    url = attachment.get("url")

    if not filename or not url:
        logger.warning("Skipping attachment with missing fields: %s", attachment)
        return None

    # Handle local file:// URLs
    if url.startswith("file://"):
        file_path = url[7:]  # Remove "file://" prefix
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            logger.info("Read local file: %s (%d bytes)", filename, len(content))
            return {
                "filename": filename,
                "url": url,
                "content": content,
                "size": len(content),
            }
        except FileNotFoundError:
            logger.warning("Local file not found: %s", file_path)
            return None
        except Exception as exc:
            logger.warning("Failed to read local file %s: %s", file_path, exc)
            return None

    max_retries = 3
    base_delay = 1.0  # seconds
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_msg = f"Failed to download {filename} ({url}): HTTP {response.status}"
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning("%s - Retrying in %.1f seconds (attempt %d/%d)", 
                                         error_msg, delay, attempt + 1, max_retries)
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning("%s - All retries exhausted", error_msg)
                            return None
                    content = await response.read()
                    if attempt > 0:
                        logger.info("Successfully downloaded %s on attempt %d/%d", filename, attempt + 1, max_retries)
                    return {
                        "filename": filename,
                        "url": url,
                        "content": content,
                        "size": len(content),
                    }
            except Exception as exc:
                error_msg = f"Failed to download {filename}: {exc}"
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("%s - Retrying in %.1f seconds (attempt %d/%d)", 
                                 error_msg, delay, attempt + 1, max_retries)
                    await asyncio.sleep(delay)
                else:
                    logger.warning("%s - All retries exhausted", error_msg)
                    return None

    return None

