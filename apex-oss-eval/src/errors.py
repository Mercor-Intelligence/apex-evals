"""
Structured error utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import textwrap

PANEL_WIDTH = 88


def _format_section(header: str, lines: Iterable[str], *, bullet: str = "•", numbered: bool = False) -> List[str]:
    """Formats section."""
    items = [line for line in lines if line]
    if not items:
        return []
    
    formatted = [f"{header}"]
    for idx, item in enumerate(items, start=1):
        marker = f"{idx}." if numbered else bullet
        formatted.append(f"  {marker} {item}")
    return formatted


def _format_context(context: Dict[str, Any]) -> List[str]:
    """Formats context pairs."""
    if not context:
        return []
    
    lines = ["Context:"]
    for key, value in context.items():
        val = value() if callable(value) else value
        lines.append(f"  • {key}: {val}")
    return lines


def render_error_panel(
    *,
    category: str,
    title: str,
    summary: str,
    context: Optional[Dict[str, Any]] = None,
    probable_causes: Optional[Iterable[str]] = None,
    next_steps: Optional[Iterable[str]] = None,
    tips: Optional[Iterable[str]] = None,
    docs_link: Optional[str] = None,
) -> str:
    """Renders error message panel."""
    header = f"{category} • {title}"
    summary_block = textwrap.dedent(summary).strip()
    
    lines: List[str] = [
        "\n" + "=" * PANEL_WIDTH,
        header,
        "=" * PANEL_WIDTH,
        "",
        summary_block,
    ]
    
    if context:
        lines.append("")
        lines.extend(_format_context({k: v for k, v in context.items() if v not in [None, ""]}))
    
    if probable_causes:
        lines.append("")
        lines.extend(_format_section("Why this happened:", probable_causes, numbered=True))
    
    if next_steps:
        lines.append("")
        lines.extend(_format_section("How to fix:", next_steps, numbered=True))
    
    if tips:
        lines.append("")
        lines.extend(_format_section("Tips:", tips))
    
    if docs_link:
        lines.append("")
        lines.append(f"Docs: {docs_link}")
    
    lines.append("\n")
    return "\n".join(lines)


@dataclass
class ErrorDetails:
    """Error metadata."""
    title: str
    summary: str
    context: Dict[str, Any] = field(default_factory=dict)
    probable_causes: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    tips: Optional[List[str]] = None
    docs_link: Optional[str] = None


class ApexEvalError(ValueError):
    """Base structured error."""
    
    category = "APEX EVAL ERROR"
    
    def __init__(
        self,
        message: Optional[str] = None,
        *,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        probable_causes: Optional[Iterable[str]] = None,
        next_steps: Optional[Iterable[str]] = None,
        tips: Optional[Iterable[str]] = None,
        docs_link: Optional[str] = None,
    ):
        if message is None:
            if not title or not summary:
                raise ValueError("Provide either `message` or both `title` and `summary`.")
            
            message = render_error_panel(
                category=self.category,
                title=title,
                summary=summary,
                context=context,
                probable_causes=probable_causes,
                next_steps=next_steps,
                tips=tips,
                docs_link=docs_link,
            )
        else:
            # Compatibility check
            if any(
                field is not None
                for field in (title, summary, context, probable_causes, next_steps, tips, docs_link)
            ):
                raise ValueError("Pass either a plain message or structured fields, not both.")
        
        self.details = ErrorDetails(
            title=title or self.__class__.__name__,
            summary=summary or message,
            context=context or {},
            probable_causes=list(probable_causes) if probable_causes else None,
            next_steps=list(next_steps) if next_steps else None,
            tips=list(tips) if tips else None,
            docs_link=docs_link,
        )
        
        super().__init__(message)
    
    def __str__(self) -> str:
        return super().__str__()


class UserInputError(ApexEvalError):
    """Configuration/input error."""
    
    category = "USER ACTION REQUIRED"


class SystemExecutionError(ApexEvalError):
    """System execution error."""
    
    category = "SYSTEM ISSUE DETECTED"


__all__ = [
    "ApexEvalError",
    "UserInputError",
    "SystemExecutionError",
    "render_error_panel",
    "ErrorDetails",
]

