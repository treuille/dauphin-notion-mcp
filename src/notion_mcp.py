"""Notion MCP server with DNN (Dauphin Notion Notation) format.

Provides token-efficient Notion access via two tools:
- notion.read: Read pages/databases in compact DNN format
- notion.apply: Execute mutation scripts

Token: Passed via --token-file <path> CLI argument at startup.
"""

import asyncio
import logging
import random
import re
import string
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("notion-mcp")

# =============================================================================
# Async Rate Limiting
# =============================================================================

# Semaphore to limit concurrent Notion API requests (Notion limit: ~3 req/sec)
_notion_semaphore: Optional[asyncio.Semaphore] = None
_async_client: Optional[httpx.AsyncClient] = None

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_JITTER_MAX = 0.5  # max random jitter to add (seconds)


# =============================================================================
# Refactoring Helpers
# =============================================================================
# These helpers consolidate repeated patterns throughout the codebase to improve
# maintainability and reduce code duplication.


def _compute_retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Compute exponential backoff delay with jitter for rate limiting.

    Consolidates the retry delay calculation that was duplicated in multiple
    places (response 429 handling and exception 429 handling).

    Args:
        attempt: Current retry attempt number (0-indexed).
        retry_after: Optional Retry-After header value from server.

    Returns:
        Delay in seconds, including random jitter to prevent thundering herd.
    """
    base_delay = RETRY_BASE_DELAY * (2 ** attempt)
    if retry_after is not None:
        base_delay = max(retry_after, base_delay)
    return base_delay + random.uniform(0, RETRY_JITTER_MAX)


def _http_error_detail(e: httpx.HTTPStatusError, max_len: int = 300) -> str:
    """Extract error detail from an HTTP status error.

    Consolidates the repeated pattern of extracting error details from
    httpx.HTTPStatusError exceptions found in ~8 exception handlers.

    Args:
        e: The HTTP status error exception.
        max_len: Maximum length of error detail to return.

    Returns:
        Truncated error response text or string representation of the error.
    """
    if e.response is not None:
        return e.response.text[:max_len]
    return str(e)


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the rate-limiting semaphore."""
    global _notion_semaphore
    if _notion_semaphore is None:
        _notion_semaphore = asyncio.Semaphore(50)
    return _notion_semaphore


async def _get_async_client() -> httpx.AsyncClient:
    """Get or create the async HTTP client."""
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(timeout=30.0)
    return _async_client

# =============================================================================
# Credential Management
# =============================================================================

_notion_token: Optional[str] = None


def _get_token() -> str:
    """Get the Notion token (set via --token-file CLI arg)."""
    if _notion_token is None:
        raise RuntimeError(
            "No Notion token. Pass --token-file <path> on the command line."
        )
    return _notion_token


# =============================================================================
# ID System
# =============================================================================

# Base62 alphabet: a-z, A-Z, 0-9 (case-sensitive)
BASE62_ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits

# Regex patterns for ID parsing
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$',
    re.IGNORECASE
)
SHORT_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{4}$')
TYPED_REF_PATTERN = re.compile(r'^([pbr]):([A-Za-z0-9]{4}|[0-9a-f-]{32,36})$', re.IGNORECASE)
NOTION_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?notion\.(?:so|site)/(?:[^/]+/)?([^?#]+)',
    re.IGNORECASE
)


def generate_short_id(existing_ids: set[str] | None = None) -> str:
    """Generate a random 4-character base62 ID.

    Args:
        existing_ids: Set of IDs to avoid collisions with.

    Returns:
        A unique 4-character ID.
    """
    existing = existing_ids or set()
    # With 62^4 = 14.7M possibilities, collisions are rare
    # but we check anyway for safety
    for _ in range(100):
        short_id = ''.join(random.choices(BASE62_ALPHABET, k=4))
        if short_id not in existing:
            return short_id
    raise RuntimeError("Failed to generate unique short ID after 100 attempts")


def normalize_uuid(uuid_str: str) -> str:
    """Normalize a UUID to standard format with dashes.

    Args:
        uuid_str: UUID with or without dashes.

    Returns:
        UUID in format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    Raises:
        ValueError: If input is not a valid UUID (wrong length or invalid chars).
    """
    # Remove any existing dashes and lowercase
    clean = uuid_str.replace('-', '').lower()
    if len(clean) != 32:
        raise ValueError(f"Invalid UUID length: {uuid_str}")
    # Validate hex characters
    if not all(c in '0123456789abcdef' for c in clean):
        raise ValueError(f"Invalid UUID characters: {uuid_str}")
    # Insert dashes at standard positions
    return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}"


def extract_uuid_from_url(url: str) -> Optional[str]:
    """Extract a Notion UUID from a URL.

    Handles formats like:
    - https://notion.so/workspace/Page-Title-abc123def456...
    - https://notion.so/abc123def456...
    - https://www.notion.so/Page-abc123def456...

    Returns:
        Normalized UUID or None if not found.
    """
    match = NOTION_URL_PATTERN.match(url)
    if not match:
        return None

    path_part = match.group(1)
    # UUID is typically at the end, possibly after a title
    # Look for 32 hex chars (with or without dashes)
    uuid_match = re.search(r'([0-9a-f]{32}|[0-9a-f-]{36})$', path_part, re.IGNORECASE)
    if uuid_match:
        return normalize_uuid(uuid_match.group(1))

    # Sometimes the ID is just the last segment without full UUID
    # e.g., notion.so/Page-abc123 where abc123 is a short form
    # Notion uses 32-char IDs, so shorter ones aren't valid UUIDs
    return None


class IdRegistry:
    """Session-scoped bidirectional mapping between short IDs and Notion UUIDs.

    Each session maintains its own registry to ensure short IDs are consistent
    within a conversation but don't conflict across sessions.
    """

    def __init__(self):
        self._short_to_uuid: dict[str, str] = {}
        self._uuid_to_short: dict[str, str] = {}

    def register(self, uuid: str, short_id: Optional[str] = None) -> str:
        """Register a UUID and return its short ID.

        If the UUID is already registered, returns existing short ID.
        If short_id is provided and available, uses it; otherwise generates new.

        Args:
            uuid: Notion UUID (will be normalized).
            short_id: Optional preferred short ID.

        Returns:
            The short ID for this UUID.
        """
        normalized = normalize_uuid(uuid)

        # Already registered?
        if normalized in self._uuid_to_short:
            return self._uuid_to_short[normalized]

        # Use provided short ID if available
        if short_id and short_id not in self._short_to_uuid:
            sid = short_id
        else:
            sid = generate_short_id(set(self._short_to_uuid.keys()))

        self._short_to_uuid[sid] = normalized
        self._uuid_to_short[normalized] = sid
        return sid

    def get_uuid(self, short_id: str) -> Optional[str]:
        """Look up the UUID for a short ID."""
        return self._short_to_uuid.get(short_id)

    def get_short_id(self, uuid: str) -> Optional[str]:
        """Look up the short ID for a UUID."""
        try:
            normalized = normalize_uuid(uuid)
            return self._uuid_to_short.get(normalized)
        except ValueError:
            return None

    def resolve(self, ref: str) -> Optional[str]:
        """Resolve any reference format to a UUID.

        Accepts:
        - Short ID: A1b2
        - Full UUID: 12345678-1234-1234-1234-123456789abc
        - UUID without dashes: 12345678123412341234123456789abc
        - Notion URL: https://notion.so/workspace/Page-abc123...
        - Typed ref: p:A1b2, b:C3d4, r:E5f6

        Returns:
            Normalized UUID or None if reference is invalid/unknown.
        """
        ref = ref.strip()

        # Typed reference (p:, b:, r:)
        typed_match = TYPED_REF_PATTERN.match(ref)
        if typed_match:
            ref = typed_match.group(2)  # Extract the ID part

        # Short ID
        if SHORT_ID_PATTERN.match(ref):
            return self.get_uuid(ref)

        # Full UUID (with or without dashes)
        if UUID_PATTERN.match(ref):
            return normalize_uuid(ref)

        # Notion URL
        if ref.startswith('http'):
            return extract_uuid_from_url(ref)

        return None

    def clear(self):
        """Clear all mappings."""
        self._short_to_uuid.clear()
        self._uuid_to_short.clear()

    def __len__(self) -> int:
        """Return number of registered IDs."""
        return len(self._short_to_uuid)

    def __contains__(self, item: str) -> bool:
        """Check if a short ID or UUID is registered."""
        if SHORT_ID_PATTERN.match(item):
            return item in self._short_to_uuid
        try:
            normalized = normalize_uuid(item)
            return normalized in self._uuid_to_short
        except ValueError:
            return False


def _resolve_or_error(
    registry: IdRegistry,
    id_val: str,
    field_name: str = "target"
) -> tuple[str | None, str | None]:
    """Resolve an ID to UUID, returning error message if resolution fails.

    Consolidates the repeated ID resolution + error check pattern that appears
    30+ times throughout execute_operation_async and related functions.

    Args:
        registry: The ID registry to resolve against.
        id_val: The ID value to resolve (short ID, UUID, or URL).
        field_name: Name of the field for error message (e.g., "parent", "target").

    Returns:
        Tuple of (uuid, error). If successful, uuid is set and error is None.
        If resolution fails, uuid is None and error contains the error message.

    Example:
        uuid, err = _resolve_or_error(registry, op.parent, "parent")
        if err:
            return {}, err
    """
    uuid = registry.resolve(id_val)
    if not uuid:
        return None, f"Unknown {field_name} ID: {id_val}"
    return uuid, None


def _append_dnn_line(
    lines: list[str],
    mode: str,
    short_id: str,
    indent: str,
    content: str
) -> None:
    """Append a DNN-formatted line with mode-aware ID prefix.

    Consolidates the repeated pattern of conditionally including the short ID
    prefix based on mode ("edit" vs "view") that appears 12+ times in render
    functions.

    Args:
        lines: List to append the formatted line to.
        mode: "edit" (includes short ID prefix) or "view" (no prefix).
        short_id: The 4-character short ID (only used in edit mode).
        indent: The indentation string (e.g., "  " per nesting level).
        content: The block content including marker (e.g., "- bullet item").
    """
    if mode == "edit":
        lines.append(f"{short_id} {indent}{content}")
    else:
        lines.append(f"{indent}{content}")


# =============================================================================
# DNN Parser
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any


class ParseState(Enum):
    """State machine states for DNN parsing."""
    HEADER = auto()
    BLOCK = auto()
    CODE = auto()


# Block line pattern: 4-char ID followed by space
BLOCK_LINE_PATTERN = re.compile(r'^([A-Za-z0-9]{4}) (.*)$')

# Header line pattern: @keyword followed by space and value
HEADER_LINE_PATTERN = re.compile(r'^@(\w+)\s*(.*)$')


@dataclass
class DnnHeader:
    """Parsed DNN header metadata."""
    version: int = 1
    page_id: Optional[str] = None
    db_id: Optional[str] = None
    ds_id: Optional[str] = None
    title: Optional[str] = None
    columns: list[dict] = field(default_factory=list)


@dataclass
class DnnBlock:
    """Parsed DNN block."""
    short_id: str
    level: int  # Nesting level (0 = root)
    block_type: str  # paragraph, heading_1, bulleted_list, etc.
    content: str  # Raw content (for non-code blocks)
    raw_lines: list[str] = field(default_factory=list)  # For code blocks
    children: list["DnnBlock"] = field(default_factory=list)
    # Block-specific attributes
    checked: Optional[bool] = None  # For to_do blocks
    language: Optional[str] = None  # For code blocks
    heading_level: Optional[int] = None  # For headings (1, 2, or 3)
    is_toggle: bool = False  # For toggle blocks/headings
    color: Optional[str] = None  # For callout blocks (e.g., "gray_background")
    opaque: bool = False  # For opaque blocks (!type~)
    structural: bool = False  # For structural blocks (!type N)
    warnings: list[str] = field(default_factory=list)  # Parser warnings


# Marker patterns in precedence order (first match wins)
BLOCK_MARKERS = [
    # Toggle headings (must come before regular headings)
    (re.compile(r'^>### (.*)$'), 'heading_3', {'is_toggle': True, 'heading_level': 3}),
    (re.compile(r'^>## (.*)$'), 'heading_2', {'is_toggle': True, 'heading_level': 2}),
    (re.compile(r'^># (.*)$'), 'heading_1', {'is_toggle': True, 'heading_level': 1}),
    # Regular headings
    (re.compile(r'^### (.*)$'), 'heading_3', {'heading_level': 3}),
    (re.compile(r'^## (.*)$'), 'heading_2', {'heading_level': 2}),
    (re.compile(r'^# (.*)$'), 'heading_1', {'heading_level': 1}),
    # Divider (exact match)
    (re.compile(r'^---$'), 'divider', {}),
    # Todo items
    (re.compile(r'^\[x\] (.*)$', re.IGNORECASE), 'to_do', {'checked': True}),
    (re.compile(r'^\[ \] (.*)$'), 'to_do', {'checked': False}),
    # Numbered list (N. where N is any integer)
    (re.compile(r'^(\d+)\. (.*)$'), 'numbered_list_item', {}),
    # Bulleted list
    (re.compile(r'^- (.*)$'), 'bulleted_list_item', {}),
    # Toggle (single >)
    (re.compile(r'^> (.*)$'), 'toggle', {}),
    # Table row (pipe-delimited with trailing pipe): | cell | cell |
    # Must come before quote — trailing | distinguishes table from quote
    (re.compile(r'^(\|.*\|)\s*$'), 'table_row', {}),
    # Quote (pipe, no trailing pipe): | text
    (re.compile(r'^\| (.*)$'), 'quote', {}),
    # Callout / opaque / structural: all !-prefixed blocks handled
    # in parse_block_type() before BLOCK_MARKERS loop. See there for
    # disambiguation order: colors → opaque (~) → structural (N) → default.
    # Child page
    (re.compile(r'^§ (.*)$'), 'child_page', {}),
    # Child database
    (re.compile(r'^⊞ (.*)$'), 'child_database', {}),
    # Link to page
    (re.compile(r'^→ (.*)$'), 'link_to_page', {}),
    # Code block start (3+ backticks, optional language)
    (re.compile(r'^`{3,}(\w*)$'), 'code', {}),
]


def parse_block_type(content: str) -> tuple[str, str, dict[str, Any], list[str]]:
    """Parse content to determine block type and extract text.

    Args:
        content: The content after ID and indentation.

    Returns:
        Tuple of (block_type, extracted_content, attributes, warnings)
    """
    warnings: list[str] = []

    # Handle escaped markers at line start
    if content.startswith('\\'):
        # Remove escape and treat as paragraph
        return 'paragraph', content[1:], {}, warnings

    # Detect unsupported heading levels (####, #####, etc.)
    h4_match = re.match(r'^(#{4,})\s*(.*)$', content)
    if h4_match:
        hashes = h4_match.group(1)
        warnings.append(
            f"Notion only supports h1-h3. '{hashes}' treated as paragraph. "
            f"Did you mean '###'?"
        )
        return 'paragraph', content, {}, warnings

    # Handle all !-prefixed blocks: callout, opaque, structural
    # Disambiguation order: colors → opaque (~) → structural (N) → default callout
    if content.startswith('!'):
        # 1. Colored callout: !color text
        callout_match = re.match(
            r'^!(gray|brown|orange|yellow|green|blue|purple|pink|red) (.*)$', content
        )
        if callout_match:
            matched_text = callout_match.group(2)
            # Double-delimiter warning
            if matched_text:
                double_marker = re.match(r'^(#{1,3}|[-*]|\d+\.|>\s*#{1,3}|>|!\w*|\||\[[ x]\])\s', matched_text)
                if double_marker:
                    warnings.append(
                        f"Content starts with '{double_marker.group(1)}' - "
                        f"looks like nested delimiters. Did you mean to change block type?"
                    )
            return 'callout', matched_text, {'color': callout_match.group(1) + '_background'}, warnings

        # 2. Opaque block: !type~ or !type~ (caption)
        opaque_match = re.match(r'^!(\w+)~(.*)$', content)
        if opaque_match and opaque_match.group(1) in OPAQUE_BLOCK_TYPES:
            opaque_type = opaque_match.group(1)
            caption = opaque_match.group(2).strip()
            if caption.startswith('(') and caption.endswith(')'):
                caption = caption[1:-1]
            return opaque_type, caption, {'opaque': True}, warnings

        # 3. Structural block: !type N
        struct_match = re.match(r'^!(\w+) (\d+)$', content)
        if struct_match and struct_match.group(1) in STRUCTURAL_BLOCK_TYPES:
            real_type = STRUCTURAL_BLOCK_TYPES[struct_match.group(1)]
            return real_type, struct_match.group(2), {'structural': True}, warnings

        # 4. Default callout: ! text (fallback)
        default_match = re.match(r'^! (.*)$', content)
        if default_match:
            matched_text = default_match.group(1)
            # Double-delimiter warning
            if matched_text:
                double_marker = re.match(r'^(#{1,3}|[-*]|\d+\.|>\s*#{1,3}|>|!\w*|\||\[[ x]\])\s', matched_text)
                if double_marker:
                    warnings.append(
                        f"Content starts with '{double_marker.group(1)}' - "
                        f"looks like nested delimiters. Did you mean to change block type?"
                    )
            return 'callout', matched_text, {'color': 'default'}, warnings

    # Try each marker in precedence order
    for pattern, block_type, attrs in BLOCK_MARKERS:
        match = pattern.match(content)
        if match:
            if block_type == 'table_row':
                raw_row = match.group(1)
                # Check for Markdown separator row (|---|---|) — skip silently
                if re.match(r'^\|[\s\-:]+(\|[\s\-:]+)*\|$', raw_row):
                    return 'table_separator', '', {}, warnings
                # Parse cells: split on unescaped pipes, strip outer pipes
                return block_type, raw_row, {}, warnings
            elif block_type == 'numbered_list_item':
                matched_text = match.group(2)
            elif block_type == 'code':
                return block_type, '', {'language': match.group(1) or None}, warnings
            elif block_type == 'divider':
                return block_type, '', attrs.copy(), warnings
            else:
                matched_text = match.group(1)

            # Check if matched_text looks like another block marker (double delimiter)
            if matched_text:
                double_marker = re.match(r'^(#{1,3}|[-*]|\d+\.|>\s*#{1,3}|>|!\w*|\||\[[ x]\])\s', matched_text)
                if double_marker:
                    warnings.append(
                        f"Content starts with '{double_marker.group(1)}' - "
                        f"looks like nested delimiters. Did you mean to change block type?"
                    )

            return block_type, matched_text, attrs.copy() if isinstance(attrs, dict) else attrs, warnings

    # Default: paragraph
    return 'paragraph', content, {}, warnings


def strip_marker_for_block(text: str, target_type: str) -> tuple[str, dict, list[str]]:
    """Strip block marker from text, aware of target block type.

    THE single function for preparing text content for any block mutation.
    Used by both ADD (new blocks) and UPDATE (existing blocks) paths.

    Why this exists:
    - Users may paste "[ ] task" into a to_do block (marker redundant)
    - Users may paste "# heading" into an h1 block (marker redundant)
    - Without stripping, we get "[ ] [ ] task" or "# # heading"

    Args:
        text: Raw text that may include block markers like "[ ] ", "# ", etc.
        target_type: The block's type (existing type for UPDATE, detected for ADD)

    Returns:
        Tuple of (clean_text, attrs, warnings) where:
        - clean_text: Text with marker stripped if applicable
        - attrs: Extracted attributes (e.g., checked=True for "[x]")
        - warnings: Any parser warnings
    """
    # Code blocks: content is literal, never strip markers
    if target_type == 'code':
        return text, {}, []

    # Use existing parse_block_type as the marker detection engine
    detected_type, clean_text, attrs, warnings = parse_block_type(text)

    # No marker detected (plain paragraph) - return original text unchanged
    if detected_type == 'paragraph':
        return text, {}, warnings

    # Marker matches target type - strip it silently
    # e.g., "[ ] task" into to_do block → "task"
    if detected_type == target_type:
        return clean_text, attrs, warnings

    # Special case: toggle heading variants match their base heading type
    # e.g., "># heading" detected as heading_1 with is_toggle=True
    if detected_type in ('heading_1', 'heading_2', 'heading_3'):
        base_target = target_type.replace('toggle_', '')  # hypothetical
        if detected_type == target_type:
            return clean_text, attrs, warnings

    # Mismatched marker - strip it but warn
    # e.g., "- bullet" pasted into a to_do block
    warnings.append(
        f"'{detected_type}' marker stripped; block is '{target_type}'. "
        f"To change block type, delete and recreate."
    )
    return clean_text, attrs, warnings


def parse_table_row_cells(raw_row: str) -> list[str]:
    """Parse pipe-delimited table row into cell strings.

    Input: "| Name | Age | City |"
    Output: ["Name", "Age", "City"]

    Handles escaped pipes (\\|) within cell content.
    """
    # Strip outer pipes and split on unescaped pipes
    inner = raw_row.strip()
    if inner.startswith('|'):
        inner = inner[1:]
    if inner.endswith('|'):
        inner = inner[:-1]

    # Split on unescaped pipes (not preceded by backslash)
    cells = re.split(r'(?<!\\)\|', inner)
    # Unescape pipes and strip whitespace
    return [cell.strip().replace('\\|', '|') for cell in cells]


def cells_to_pipe_row(cells: list[str]) -> str:
    """Convert cell strings to pipe-delimited row.

    Input: ["Name", "Age", "City"]
    Output: "| Name | Age | City |"

    Escapes pipe characters in cell content.
    """
    escaped = [cell.replace('|', '\\|') for cell in cells]
    return "| " + " | ".join(escaped) + " |"


def calculate_indent_level(line_after_id: str) -> tuple[int, str]:
    """Calculate indentation level and extract content.

    Args:
        line_after_id: The line content after the ID and separator space.

    Returns:
        Tuple of (indent_level, content_without_indent)
    """
    # Count leading spaces
    stripped = line_after_id.lstrip(' ')
    spaces = len(line_after_id) - len(stripped)
    level = spaces // 2
    return level, stripped


class DnnParseError(Exception):
    """Error during DNN parsing."""

    def __init__(
        self,
        code: str,
        message: str,
        line: int,
        excerpt: str,
        suggestions: list[str] | None = None,
        autofix: dict | None = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.line = line
        self.excerpt = excerpt
        self.suggestions = suggestions or []
        self.autofix = autofix

    def to_dict(self) -> dict:
        """Convert to error response dict."""
        result = {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "excerpt": self.excerpt,
        }
        if self.suggestions:
            result["suggestions"] = self.suggestions
        if self.autofix:
            result["autofix"] = self.autofix
        return result


@dataclass
class DnnParseResult:
    """Result of parsing a DNN document."""
    header: DnnHeader
    blocks: list[DnnBlock]
    errors: list[DnnParseError] = field(default_factory=list)


# =============================================================================
# Inline Formatting Parser (Parsy-based)
# =============================================================================

import parsy as P

@dataclass
class RichTextSpan:
    """A span of rich text with formatting."""
    text: str
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False
    code: bool = False
    color: Optional[str] = None
    link: Optional[str] = None
    # For special spans
    span_type: str = "text"  # text, mention_user, mention_date, mention_page, equation
    user_id: Optional[str] = None
    page_id: Optional[str] = None  # For page mentions (p:shortID)
    date: Optional[str] = None
    end_date: Optional[str] = None
    expression: Optional[str] = None


# Valid Notion colors
NOTION_COLORS = {
    'default', 'gray', 'brown', 'orange', 'yellow', 'green', 'blue',
    'purple', 'pink', 'red',
    'gray_background', 'brown_background', 'orange_background',
    'yellow_background', 'green_background', 'blue_background',
    'purple_background', 'pink_background', 'red_background',
}


def _apply_formatting(spans: list[RichTextSpan], **kwargs) -> list[RichTextSpan]:
    """Apply formatting attributes to spans, preserving existing values.

    For color: only set if span doesn't already have a color (fixes nested color bug).
    For other attributes: always set (they're additive).
    """
    for span in spans:
        for key, value in kwargs.items():
            if key == 'color':
                # Don't override existing color (fixes nested color bug)
                if span.color is None:
                    span.color = value
            else:
                setattr(span, key, value)
    return spans


def _merge_adjacent_spans(spans: list[RichTextSpan]) -> list[RichTextSpan]:
    """Merge adjacent spans with identical formatting."""
    if not spans:
        return []

    merged: list[RichTextSpan] = []
    for span in spans:
        if (merged and span.span_type == 'text' and
            merged[-1].span_type == 'text' and
            merged[-1].bold == span.bold and
            merged[-1].italic == span.italic and
            merged[-1].strikethrough == span.strikethrough and
            merged[-1].underline == span.underline and
            merged[-1].code == span.code and
            merged[-1].color == span.color and
            merged[-1].link == span.link):
            merged[-1].text += span.text
        else:
            merged.append(span)
    return merged


# Characters that start special syntax (used for literal text boundaries)
_SPECIAL_CHARS = set('\\*~`[$:@')


def _parse_date_mention(date_str: str) -> RichTextSpan:
    """Parse date mention, handling optional range syntax."""
    if '→' in date_str:
        start, end = date_str.split('→')
        return RichTextSpan(
            text='', span_type='mention_date', date=start, end_date=end
        )
    return RichTextSpan(text='', span_type='mention_date', date=date_str)


def _make_inline_parser():
    """Build the inline formatting parser using parsy combinators.

    Returns a parser that converts text to list[RichTextSpan].

    Key insight: For delimited formats like **bold**, we need to parse the
    inner content WITHOUT trying the same delimiter. We use regex for inner
    content that stops at the closing delimiter, then recursively parse that.
    """

    # Helper to recursively parse inner content after extraction
    def parse_inner(text: str) -> list[RichTextSpan]:
        """Recursively parse inner text. Used after extracting delimited content."""
        if not text:
            return [RichTextSpan(text='')]
        try:
            return _inline_parser_impl.parse(text)
        except Exception:
            return [RichTextSpan(text=text)]

    # Escape sequences: \* \~ \` \[ \] \: \$ \\ etc.
    escape_chars = '\\*~`[]$:@#-|!>'
    escaped = (P.string('\\') >> P.char_from(escape_chars)).map(
        lambda c: RichTextSpan(text=c)
    )

    # Equation: $expr$
    equation = (
        P.string('$') >>
        P.regex(r'[^$]+') <<
        P.string('$')
    ).map(lambda expr: RichTextSpan(text='', span_type='equation', expression=expr))

    # User mention: @user:UUID
    mention_user = (
        P.string('@user:') >>
        P.regex(r'[a-f0-9-]{36}', re.IGNORECASE)
    ).map(lambda uid: RichTextSpan(text='', span_type='mention_user', user_id=uid))

    # Date mention: @date:YYYY-MM-DD or @date:YYYY-MM-DD→YYYY-MM-DD
    mention_date = (
        P.string('@date:') >>
        P.regex(r'\d{4}-\d{2}-\d{2}(?:→\d{4}-\d{2}-\d{2})?')
    ).map(_parse_date_mention)

    # Page mention: @p:shortID or @p:full-uuid
    # Creates a Notion @mention that shows the page's actual title
    mention_page = (
        P.string('@p:') >>
        P.regex(r'[A-Za-z0-9-]+')
    ).map(lambda pid: RichTextSpan(text='', span_type='mention_page', page_id=pid))

    # Code: `text` (no nesting allowed)
    code = (
        P.string('`') >>
        P.regex(r'[^`]+') <<
        P.string('`')
    ).map(lambda t: RichTextSpan(text=t, code=True))

    # Bold: **content** - use regex to capture content, then parse recursively
    bold = (
        P.string('**') >>
        P.regex(r'((?:[^*]|\*(?!\*))+)') <<  # Match until ** (non-greedy)
        P.string('**')
    ).map(lambda inner: _apply_formatting(parse_inner(inner), bold=True))

    # Strikethrough: ~~content~~
    strikethrough = (
        P.string('~~') >>
        P.regex(r'((?:[^~]|~(?!~))+)') <<  # Match until ~~
        P.string('~~')
    ).map(lambda inner: _apply_formatting(parse_inner(inner), strikethrough=True))

    # Italic: *content* (single asterisk, not double)
    # More careful: match * then content without * then *
    italic = (
        P.string('*') >>
        P.regex(r'([^*]+)') <<
        P.string('*')
    ).map(lambda inner: _apply_formatting(parse_inner(inner), italic=True))

    # Underline directive: :u[content]
    underline = (
        P.string(':u[') >>
        P.regex(r'((?:[^\[\]]|\[(?:[^\[\]])*\])*)') <<  # Match balanced brackets
        P.string(']')
    ).map(lambda inner: _apply_formatting(parse_inner(inner), underline=True))

    # Color directive: :color[content] or :color-background[content]
    @P.generate
    def color():
        yield P.string(':')
        color_name = yield P.regex(r'[a-z]+(?:-background)?')
        yield P.string('[')
        # Match content with balanced brackets
        inner = yield P.regex(r'((?:[^\[\]]|\[(?:[^\[\]])*\])*)')
        yield P.string(']')
        # Normalize color name (dash to underscore)
        normalized = color_name.replace('-', '_')
        if normalized in NOTION_COLORS or normalized == 'u':
            if normalized == 'u':
                return _apply_formatting(parse_inner(inner), underline=True)
            return _apply_formatting(parse_inner(inner), color=normalized)
        else:
            # Unknown color - return as plain text
            return [RichTextSpan(text=f':{color_name}[{inner}]')]

    # Link: [text](url) or page link: [text](p:shortID)
    # Note: [text](p:shortID) creates a LINK with custom text (not an @mention)
    # For @mentions, use @p:shortID syntax instead
    @P.generate
    def link():
        yield P.string('[')
        text = yield P.regex(r'(?:[^\[\]]|\[(?:[^\[\]])*\])*')  # Balanced brackets
        yield P.string('](')
        url = yield P.regex(r'[^)]+')
        yield P.string(')')
        # p:shortID creates a link (resolved to Notion URL in rich_text_spans_to_notion)
        # NOT an @mention - use @p:shortID for that
        return _apply_formatting(parse_inner(text), link=url)

    # Single literal character (not starting a special sequence)
    def is_literal_char(c):
        return c not in _SPECIAL_CHARS

    # Run of literal characters (optimization: batch them)
    literal_run = P.test_char(is_literal_char, 'literal').at_least(1).map(
        lambda chars: RichTextSpan(text=''.join(chars))
    )

    # Special character that didn't match any pattern (fallback)
    special_fallback = P.any_char.map(lambda c: RichTextSpan(text=c))

    # Order matters: try formats first, then literal runs, then single special chars
    formatted_or_literal = (
        escaped |           # Escape sequences first
        equation |          # $...$
        mention_user |      # @user:UUID
        mention_date |      # @date:YYYY-MM-DD
        mention_page |      # @p:shortID (page @mention)
        code |              # `...`
        bold |              # **...**
        strikethrough |     # ~~...~~ (before italic to avoid ** vs * issues)
        italic |            # *...*
        underline |         # :u[...]
        color |             # :color[...]
        link |              # [...](...) - includes [text](p:shortID) for links
        literal_run |       # Batch of normal characters
        special_fallback    # Single special char that didn't start a pattern
    )

    # Build the main parser
    def flatten(items):
        """Flatten nested lists of spans."""
        flat = []
        for item in items:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    _inline_parser_impl = formatted_or_literal.many().map(flatten)

    return _inline_parser_impl


# Build the parser once at module load
_inline_parser = _make_inline_parser()


# Pattern to detect @date:X → @date:Y and normalize to @date:X→Y
# Matches: @date:YYYY-MM-DD followed by optional spaces, arrow (→ or ->),
# optional spaces, then @date:YYYY-MM-DD
_DATE_RANGE_NORMALIZE_PATTERN = re.compile(
    r'@date:(\d{4}-\d{2}-\d{2})\s*(?:→|->)\s*@date:(\d{4}-\d{2}-\d{2})'
)


def _normalize_date_ranges(text: str) -> str:
    """Normalize two separate date mentions into a single date range.

    Converts: @date:2025-01-15 → @date:2025-01-25
    Into:     @date:2025-01-15→2025-01-25

    This allows the parser to accept both formats, making it more forgiving
    for both human and AI input.

    Args:
        text: Input text potentially containing date patterns.

    Returns:
        Text with normalized date ranges.
    """
    return _DATE_RANGE_NORMALIZE_PATTERN.sub(r'@date:\1→\2', text)


def parse_inline_formatting(text: str) -> list[RichTextSpan]:
    """Parse inline formatting in text and return rich text spans.

    Uses a parsy-based PEG parser for clean handling of:
    - Nested formatting (e.g., :red[**bold**])
    - Nested colors (e.g., :red[:green[inner] outer])
    - Escape sequences (e.g., \\* for literal asterisk)

    Args:
        text: The text content to parse for inline formatting.

    Returns:
        List of RichTextSpan objects representing the formatted text.
    """
    if not text:
        return []

    # Normalize date ranges: @date:X → @date:Y becomes @date:X→Y
    text = _normalize_date_ranges(text)

    try:
        spans = _inline_parser.parse(text)
        return _merge_adjacent_spans(spans)
    except P.ParseError as e:
        # Graceful fallback: return whole text as plain span
        logger.warning(f"Inline formatting parse error: {e}")
        return [RichTextSpan(text=text)]


# =============================================================================
# Notion Rich Text → DNN Conversion
# =============================================================================

def notion_rich_text_to_dnn(rich_text: list[dict]) -> str:
    """Convert Notion rich_text array to DNN inline formatting.

    Args:
        rich_text: Notion API rich_text array.

    Returns:
        DNN formatted string.
    """
    if not rich_text:
        return ""

    parts = []
    for item in rich_text:
        item_type = item.get("type", "text")

        if item_type == "equation":
            expr = item.get("equation", {}).get("expression", "")
            parts.append(f"${expr}$")
            continue

        if item_type == "mention":
            mention = item.get("mention", {})
            mention_type = mention.get("type")
            if mention_type == "user":
                result = f"@user:{mention.get('user', {}).get('id', '')}"
            elif mention_type == "date":
                date_info = mention.get("date", {})
                start = date_info.get("start", "")
                end = date_info.get("end")
                result = f"@date:{start}→{end}" if end else f"@date:{start}"
            elif mention_type == "page":
                page_id = mention.get("page", {}).get("id", "")
                page_title = item.get("plain_text", "page")
                result = f"[{page_title}](p:{page_id})"
            elif mention_type == "database":
                # Databases are pages in Notion's model — use page-link syntax
                db_id = mention.get("database", {}).get("id", "")
                db_title = item.get("plain_text", "database")
                result = f"[{db_title}](p:{db_id})"
            elif mention_type == "link_mention":
                # Rich URL embed — render as standard link (preview card is lost)
                url = mention.get("link_mention", {}).get("href", "")
                title = item.get("plain_text", url)
                result = f"[{title}]({url})" if url else (title or "")
            elif mention_type == "link_preview":
                # Integration-specific rich embed — render as link
                url = mention.get("link_preview", {}).get("url", "")
                title = item.get("plain_text", url)
                result = f"[{title}]({url})" if url else (title or "")
            elif mention_type == "template_mention":
                # Dynamic template value (e.g., "today", "me") — plain text
                result = item.get("plain_text", "")
            else:
                # Unknown mention type — use plain_text to avoid silent drops
                result = item.get("plain_text", "")

            # Apply annotations to mention output (same logic as text)
            if result:
                annotations = item.get("annotations", {})
                if annotations.get("code"):
                    result = f"`{result}`"
                else:
                    if annotations.get("bold"):
                        result = f"**{result}**"
                    if annotations.get("italic"):
                        result = f"*{result}*"
                    if annotations.get("strikethrough"):
                        result = f"~~{result}~~"
                    if annotations.get("underline"):
                        result = f":u[{result}]"
                    color = annotations.get("color", "default")
                    if color and color != "default":
                        color_dnn = color.replace("_", "-")
                        result = f":{color_dnn}[{result}]"

            parts.append(result)
            continue

        # Regular text
        text_obj = item.get("text", {})
        content = text_obj.get("content", "")
        link = text_obj.get("link")
        annotations = item.get("annotations", {})

        # Apply formatting in order (innermost first for proper nesting)
        result = content

        # Code (doesn't nest with other formatting typically)
        if annotations.get("code"):
            result = f"`{result}`"
        else:
            # Apply other formatting
            if annotations.get("bold"):
                result = f"**{result}**"
            if annotations.get("italic"):
                result = f"*{result}*"
            if annotations.get("strikethrough"):
                result = f"~~{result}~~"
            if annotations.get("underline"):
                result = f":u[{result}]"

            # Color
            color = annotations.get("color", "default")
            if color and color != "default":
                # Convert underscore to dash for DNN format
                color_dnn = color.replace("_", "-")
                result = f":{color_dnn}[{result}]"

        # Link wrapping
        if link:
            url = link.get("url", "")
            result = f"[{result}]({url})"

        parts.append(result)

    return "".join(parts)


def rich_text_spans_to_notion(
    spans: list[RichTextSpan],
    registry: Optional['IdRegistry'] = None
) -> list[dict]:
    """Convert RichTextSpan list to Notion API rich_text format.

    Args:
        spans: List of RichTextSpan objects.
        registry: Optional ID registry for resolving page mention short IDs.

    Returns:
        List of Notion rich_text objects.
    """
    result = []
    for span in spans:
        if span.span_type == 'equation':
            result.append({
                "type": "equation",
                "equation": {"expression": span.expression or ""}
            })
        elif span.span_type == 'mention_user':
            result.append({
                "type": "mention",
                "mention": {
                    "type": "user",
                    "user": {"id": span.user_id}
                }
            })
        elif span.span_type == 'mention_date':
            date_obj = {"start": span.date}
            if span.end_date:
                date_obj["end"] = span.end_date
            result.append({
                "type": "mention",
                "mention": {"type": "date", "date": date_obj}
            })
        elif span.span_type == 'mention_page':
            # Resolve short ID to full UUID
            page_uuid = None
            if registry and span.page_id:
                page_uuid = registry.resolve(span.page_id)
            if not page_uuid and span.page_id:
                # If no registry or not found, try using page_id directly
                # (it might already be a full UUID)
                page_uuid = span.page_id
            if page_uuid:
                result.append({
                    "type": "mention",
                    "mention": {
                        "type": "page",
                        "page": {"id": page_uuid}
                    }
                })
            else:
                # Fallback: render as plain text if we can't resolve
                result.append({
                    "type": "text",
                    "text": {"content": span.text}
                })
        else:
            obj: dict = {
                "type": "text",
                "text": {"content": span.text}
            }

            # Add link if present
            if span.link:
                url = span.link
                # Resolve p:shortID to full Notion URL
                if url.startswith('p:'):
                    page_ref = url[2:]
                    page_uuid = None
                    if registry:
                        page_uuid = registry.resolve(page_ref)
                    if not page_uuid:
                        # Try using as-is (might be full UUID)
                        page_uuid = page_ref
                    # Convert to Notion URL (remove dashes for URL format)
                    url = f"https://notion.so/{page_uuid.replace('-', '')}"
                obj["text"]["link"] = {"url": url}

            # Add annotations if any are non-default
            annotations = {}
            if span.bold:
                annotations["bold"] = True
            if span.italic:
                annotations["italic"] = True
            if span.strikethrough:
                annotations["strikethrough"] = True
            if span.underline:
                annotations["underline"] = True
            if span.code:
                annotations["code"] = True
            if span.color and span.color != 'default':
                annotations["color"] = span.color

            if annotations:
                obj["annotations"] = annotations

            result.append(obj)

    return result


def _text_to_rich_text(
    text: str,
    registry: Optional['IdRegistry'] = None
) -> list[dict]:
    """Convert DNN-formatted text to Notion rich_text array.

    Parses @mentions and inline formatting, returns Notion API format.
    Use for titles and any text field that should support @mentions.

    Args:
        text: DNN-formatted text (may contain @mentions, **bold**, etc.)
        registry: Optional ID registry for resolving page mention short IDs.

    Returns:
        List of Notion rich_text objects.
    """
    if not text:
        return []
    spans = parse_inline_formatting(text)
    return rich_text_spans_to_notion(spans, registry)


def parse_dnn(dnn_text: str) -> DnnParseResult:
    """Parse a DNN document into structured form.

    Args:
        dnn_text: The DNN text to parse.

    Returns:
        DnnParseResult with header, blocks, and any errors.
    """
    lines = dnn_text.split('\n')
    header = DnnHeader()
    blocks: list[DnnBlock] = []
    errors: list[DnnParseError] = []

    state = ParseState.HEADER
    current_code_block: Optional[DnnBlock] = None
    code_block_indent: int = 0
    code_block_backtick_count: int = 3

    for line_num, line in enumerate(lines, start=1):
        # Skip empty lines in header
        if state == ParseState.HEADER and not line.strip():
            continue

        # Check for transition from HEADER to BLOCK
        if state == ParseState.HEADER:
            block_match = BLOCK_LINE_PATTERN.match(line)
            if block_match:
                state = ParseState.BLOCK
                # Fall through to process as block
            else:
                # Check for missing space after ID (common error)
                if re.match(r'^[A-Za-z0-9]{4}[^ ]', line):
                    errors.append(DnnParseError(
                        code="MISSING_ID_SPACE",
                        message="Missing space after block ID",
                        line=line_num,
                        excerpt=f"{line_num}|{line}",
                        suggestions=["Add space after 4-char ID"],
                        autofix={
                            "safe": True,
                            "patched": line[:4] + ' ' + line[4:]
                        }
                    ))
                    continue

                # Parse as header line
                header_match = HEADER_LINE_PATTERN.match(line)
                if header_match:
                    keyword = header_match.group(1)
                    value = header_match.group(2).strip()
                    if keyword == 'dnn':
                        try:
                            header.version = int(value)
                        except ValueError:
                            errors.append(DnnParseError(
                                code="INVALID_VERSION",
                                message=f"Invalid DNN version: {value}",
                                line=line_num,
                                excerpt=f"{line_num}|{line}"
                            ))
                    elif keyword == 'page':
                        header.page_id = value
                    elif keyword == 'db':
                        header.db_id = value
                    elif keyword == 'ds':
                        header.ds_id = value
                    elif keyword == 'title':
                        header.title = value
                    elif keyword == 'cols':
                        # Parse column definitions: ID:Name(type) ...
                        cols = []
                        for col_def in value.split():
                            col_match = re.match(
                                r'([A-Za-z0-9]{4}):(\w+)\((\w+)\)',
                                col_def
                            )
                            if col_match:
                                cols.append({
                                    'id': col_match.group(1),
                                    'name': col_match.group(2),
                                    'type': col_match.group(3),
                                })
                        header.columns = cols
                continue

        # CODE state: collect raw lines until closing backticks
        if state == ParseState.CODE:
            # Check for closing delimiter (must match backtick count and indent)
            stripped = line.lstrip(' ')
            indent = len(line) - len(stripped)
            expected_close = '`' * code_block_backtick_count
            if stripped == expected_close and indent == code_block_indent:
                # End of code block
                if current_code_block:
                    blocks.append(current_code_block)
                current_code_block = None
                state = ParseState.BLOCK
            else:
                # Add raw line to code block
                if current_code_block:
                    current_code_block.raw_lines.append(line)
            continue

        # BLOCK state: parse block lines
        if state == ParseState.BLOCK:
            # Empty lines are allowed (could separate logical sections)
            if not line.strip():
                continue

            block_match = BLOCK_LINE_PATTERN.match(line)
            if not block_match:
                # Check for common errors

                # Missing space after ID?
                if re.match(r'^[A-Za-z0-9]{4}[^ ]', line):
                    errors.append(DnnParseError(
                        code="MISSING_ID_SPACE",
                        message="Missing space after block ID",
                        line=line_num,
                        excerpt=f"{line_num}|{line}",
                        suggestions=["Add space after 4-char ID"],
                        autofix={
                            "safe": True,
                            "patched": line[:4] + ' ' + line[4:]
                        }
                    ))
                    continue

                # Unknown line format
                errors.append(DnnParseError(
                    code="PARSE_ERROR",
                    message="Invalid block line format",
                    line=line_num,
                    excerpt=f"{line_num}|{line}",
                    suggestions=["Block lines must start with 4-char ID + space"]
                ))
                continue

            short_id = block_match.group(1)
            rest = block_match.group(2)

            # Check indentation
            level, content = calculate_indent_level(rest)

            # Validate indentation is multiple of 2
            spaces = len(rest) - len(rest.lstrip(' '))
            if spaces % 2 != 0:
                nearest = (spaces // 2) * 2
                if spaces % 2 > 0:
                    nearest = ((spaces + 1) // 2) * 2
                errors.append(DnnParseError(
                    code="INDENT_ERROR",
                    message=f"Indent not multiple of 2 ({spaces} spaces)",
                    line=line_num,
                    excerpt=f"{line_num}|{line}",
                    suggestions=[f"Use {nearest} spaces for level {nearest // 2}"],
                    autofix={
                        "safe": True,
                        "patched": f"{short_id} {' ' * nearest}{content}"
                    }
                ))

            # Check for tabs
            if '\t' in rest:
                errors.append(DnnParseError(
                    code="TABS_IN_INDENT",
                    message="Tabs found in indentation",
                    line=line_num,
                    excerpt=f"{line_num}|{line}",
                    suggestions=["Use spaces (2 per level) instead of tabs"],
                    autofix={
                        "safe": True,
                        "patched": line.replace('\t', '  ')
                    }
                ))

            # Parse block type
            block_type, text, attrs, block_warnings = parse_block_type(content)

            block = DnnBlock(
                short_id=short_id,
                level=level,
                block_type=block_type,
                content=text,
                warnings=block_warnings,
                **attrs
            )

            # Handle code block start
            if block_type == 'code':
                state = ParseState.CODE
                current_code_block = block
                code_block_indent = spaces
                # Count backticks in opening delimiter for matching close
                code_block_backtick_count = len(content) - len(content.lstrip('`'))
                continue

            blocks.append(block)

    # Check for unterminated code block
    if state == ParseState.CODE and current_code_block:
        closing_fence = '`' * code_block_backtick_count
        errors.append(DnnParseError(
            code="CODE_BLOCK_UNTERMINATED",
            message="Code block not closed",
            line=len(lines),
            excerpt=f"EOF",
            suggestions=[f"Add closing {closing_fence} at same indent level"],
            autofix={
                "safe": True,
                "patched": f"{' ' * code_block_indent}{closing_fence}"
            }
        ))
        # Still add the incomplete code block
        blocks.append(current_code_block)

    return DnnParseResult(header=header, blocks=blocks, errors=errors)


# =============================================================================
# Notion API Client
# =============================================================================

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"


def _notion_request(
    method: str,
    endpoint: str,
    json_body: Optional[dict] = None
) -> dict:
    """Make authenticated request to Notion API."""
    token = _get_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    url = f"{NOTION_API_BASE}{endpoint}"

    with httpx.Client(timeout=30.0) as client:
        if method == "GET":
            response = client.get(url, headers=headers)
        elif method == "POST":
            response = client.post(url, headers=headers, json=json_body or {})
        elif method == "PATCH":
            response = client.patch(url, headers=headers, json=json_body or {})
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()


async def _notion_request_async(
    method: str,
    endpoint: str,
    json_body: Optional[dict] = None
) -> dict:
    """Make authenticated async request to Notion API with rate limiting and retry.

    Uses a semaphore to limit concurrent requests and exponential backoff
    for rate limit errors (429).
    """
    token = _get_token()
    sem = _get_semaphore()
    client = await _get_async_client()

    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    url = f"{NOTION_API_BASE}{endpoint}"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, headers=headers, json=json_body or {})
                elif method == "PATCH":
                    response = await client.patch(url, headers=headers, json=json_body or {})
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Handle rate limiting with exponential backoff + jitter
                # Uses _compute_retry_delay helper to consolidate delay calculation
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", RETRY_BASE_DELAY))
                    delay = _compute_retry_delay(attempt, retry_after)
                    logger.warning(f"Rate limited, waiting {delay:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    delay = _compute_retry_delay(attempt)
                    logger.warning(f"Rate limited, waiting {delay:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
                raise

        raise httpx.HTTPStatusError(
            f"Max retries ({MAX_RETRIES}) exceeded",
            request=None,
            response=None
        )


# =============================================================================
# Block Fetching
# =============================================================================

# Block types that are "opaque" (rendered as !type~ placeholders)
OPAQUE_BLOCK_TYPES = {
    'image', 'video', 'file', 'pdf', 'bookmark', 'embed',
    'link_preview', 'synced_block',
    'table_of_contents', 'breadcrumb',
    'equation', 'template', 'link_to_page', 'unsupported'
}
# Note: table and table_row are now rendered with full read/write support

# Structural block types: DNN name → Notion block type
# Rendered as !name N (editable containers with children)
STRUCTURAL_BLOCK_TYPES = {
    'cols': 'column_list',
    'col': 'column',
    'table': 'table',
}

# Callout color names (used in ! prefix disambiguation)
CALLOUT_COLORS = {
    'gray', 'brown', 'orange', 'yellow', 'green',
    'blue', 'purple', 'pink', 'red'
}

# Block types that can have children
PARENT_BLOCK_TYPES = {
    'paragraph', 'heading_1', 'heading_2', 'heading_3',
    'bulleted_list_item', 'numbered_list_item', 'to_do',
    'toggle', 'quote', 'callout', 'column_list', 'column',
    'synced_block', 'template', 'table'
}


def fetch_block_children(
    block_id: str,
    depth: int = 10,
    current_depth: int = 0
) -> list[dict]:
    """Fetch all children of a block, recursively (sync version).

    Args:
        block_id: The parent block/page ID.
        depth: Maximum nesting depth to fetch.
        current_depth: Current recursion depth.

    Returns:
        List of block objects with children populated.
    """
    if current_depth >= depth:
        return []

    blocks = []
    start_cursor = None

    # Paginate through all children
    while True:
        endpoint = f"/blocks/{block_id}/children"
        if start_cursor:
            endpoint += f"?start_cursor={start_cursor}"

        try:
            result = _notion_request("GET", endpoint)
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to fetch children of {block_id}: {e}")
            break

        for block in result.get("results", []):
            # Recursively fetch children if block has them
            if block.get("has_children") and block["type"] in PARENT_BLOCK_TYPES:
                block["_children"] = fetch_block_children(
                    block["id"],
                    depth=depth,
                    current_depth=current_depth + 1
                )
            blocks.append(block)

        if not result.get("has_more"):
            break
        start_cursor = result.get("next_cursor")

    return blocks


async def _fetch_children_one_level(block_id: str) -> list[dict]:
    """Fetch immediate children of a block (async, with pagination)."""
    blocks = []
    start_cursor = None

    while True:
        endpoint = f"/blocks/{block_id}/children"
        if start_cursor:
            endpoint += f"?start_cursor={start_cursor}"

        try:
            result = await _notion_request_async("GET", endpoint)
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to fetch children of {block_id}: {e}")
            break

        blocks.extend(result.get("results", []))

        if not result.get("has_more"):
            break
        start_cursor = result.get("next_cursor")

    return blocks


async def fetch_block_children_async(
    block_id: str,
    depth: int = 10
) -> list[dict]:
    """Fetch all children of a block with parallel fetching (async version).

    Uses breadth-first traversal with parallel requests at each level,
    which is much faster than depth-first sequential fetching.

    Args:
        block_id: The parent block/page ID.
        depth: Maximum nesting depth to fetch.

    Returns:
        List of block objects with children populated.
    """
    if depth <= 0:
        return []

    # Fetch the first level
    blocks = await _fetch_children_one_level(block_id)

    # Process levels breadth-first with parallel fetching
    current_depth = 1
    current_level_blocks = blocks

    while current_depth < depth:
        # Find all blocks at this level that need children fetched
        blocks_needing_children = [
            b for b in current_level_blocks
            if b.get("has_children") and b["type"] in PARENT_BLOCK_TYPES
        ]

        if not blocks_needing_children:
            break

        # Fetch all children in parallel
        children_lists = await asyncio.gather(*[
            _fetch_children_one_level(b["id"])
            for b in blocks_needing_children
        ])

        # Attach children to their parents
        next_level_blocks = []
        for block, children in zip(blocks_needing_children, children_lists):
            block["_children"] = children
            next_level_blocks.extend(children)

        current_level_blocks = next_level_blocks
        current_depth += 1

    return blocks


def fetch_page(page_id: str) -> dict:
    """Fetch page metadata (sync version).

    Args:
        page_id: The page UUID.

    Returns:
        Page object with properties.
    """
    return _notion_request("GET", f"/pages/{page_id}")


async def fetch_page_async(page_id: str) -> dict:
    """Fetch page metadata (async version).

    Args:
        page_id: The page UUID.

    Returns:
        Page object with properties.
    """
    return await _notion_request_async("GET", f"/pages/{page_id}")


def get_page_title(page: dict) -> str:
    """Extract title from page properties."""
    props = page.get("properties", {})

    # Try common title property names
    for key in ["title", "Title", "Name", "name"]:
        if key in props:
            title_prop = props[key]
            if title_prop.get("type") == "title":
                title_array = title_prop.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_array)

    # Fallback: find any title-type property
    for prop in props.values():
        if prop.get("type") == "title":
            title_array = prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in title_array)

    return "Untitled"


# =============================================================================
# Database Fetching (API 2025-09-03 with data sources)
# =============================================================================

async def fetch_database_async(database_id: str) -> dict:
    """Fetch database container metadata (title, data_sources list).

    In API 2025-09-03, databases are containers for data sources.
    This returns the database with its data_sources array.

    Args:
        database_id: The database UUID.

    Returns:
        Database container object with data_sources array.
    """
    return await _notion_request_async("GET", f"/databases/{database_id}")


async def fetch_data_source_async(data_source_id: str) -> dict:
    """Fetch data source metadata (schema/properties).

    In API 2025-09-03, the schema (properties) lives on the data source,
    not the database container.

    Args:
        data_source_id: The data source UUID.

    Returns:
        Data source object with properties schema.
    """
    return await _notion_request_async("GET", f"/data_sources/{data_source_id}")


async def query_data_source_async(
    data_source_id: str,
    filter_obj: Optional[dict] = None,
    sorts: Optional[list] = None,
    limit: int = 50
) -> tuple[list[dict], bool]:
    """Query data source rows (API 2025-09-03).

    Args:
        data_source_id: The data source UUID (not database UUID).
        filter_obj: Optional filter object.
        sorts: Optional list of sort objects.
        limit: Maximum rows to return (default 50).

    Returns:
        Tuple of (rows, has_more) where:
        - rows: List of page objects (database rows)
        - has_more: True if more rows exist beyond the limit
    """
    rows = []
    start_cursor = None
    has_more = False

    while len(rows) < limit:
        body: dict = {"page_size": min(100, limit - len(rows))}
        if filter_obj:
            body["filter"] = filter_obj
        if sorts:
            body["sorts"] = sorts
        if start_cursor:
            body["start_cursor"] = start_cursor

        result = await _notion_request_async(
            "POST",
            f"/data_sources/{data_source_id}/query",
            json_body=body
        )

        rows.extend(result.get("results", []))
        has_more = result.get("has_more", False)

        if not has_more or len(rows) >= limit:
            break
        start_cursor = result.get("next_cursor")

    return rows[:limit], has_more


def get_database_title(database: dict) -> str:
    """Extract title from database metadata."""
    title_array = database.get("title", [])
    return "".join(t.get("plain_text", "") for t in title_array) or "Untitled"


# Property type to DNN type mapping
PROPERTY_TYPE_MAP = {
    "title": "title",
    "rich_text": "text",
    "number": "number",
    "select": "select",
    "multi_select": "multi_select",
    "status": "status",
    "date": "date",
    "people": "people",
    "files": "files",
    "checkbox": "checkbox",
    "url": "url",
    "email": "email",
    "phone_number": "phone",
    "formula": "formula",
    "relation": "relation",
    "rollup": "rollup",
    "created_time": "created",
    "created_by": "created_by",
    "last_edited_time": "edited",
    "last_edited_by": "edited_by",
    "unique_id": "id",
}

# Reverse mapping: DNN short name → Notion API type
# Used by +db and +prop to convert property definitions to API format
DNN_TO_API_TYPE = {v: k for k, v in PROPERTY_TYPE_MAP.items()}

# Property types supported by +db and +prop
CREATABLE_PROPERTY_TYPES = {
    "title", "text", "number", "select", "multi_select",
    "date", "checkbox", "url", "email", "phone",
    "relation", "created", "created_by", "edited", "edited_by", "id",
}

# Number format aliases
NUMBER_FORMATS = {
    "number", "number_with_commas", "percent",
    "dollar", "euro", "pound", "yen",
    "canadian_dollar", "australian_dollar", "swiss_franc",
    "danish_krone", "norwegian_krone", "swedish_krona",
    "singapore_dollar", "hong_kong_dollar", "new_zealand_dollar",
    "mexican_peso", "south_african_rand", "brazilian_real",
    "indian_rupee", "chinese_yuan", "japanese_yen",
    "korean_won", "philippine_peso", "thai_baht",
    "indonesian_rupiah", "turkish_lira", "russian_ruble",
    "colombian_peso", "chilean_peso", "peruvian_sol",
    "argentine_peso", "uruguayan_peso",
    "emirati_dirham", "saudi_riyal", "qatari_riyal",
    "omani_rial", "kuwaiti_dinar", "bahraini_dinar",
    "egyptian_pound", "nigerian_naira", "kenyan_shilling",
    "zloty", "czech_koruna", "forint", "hryvnia", "leu", "lev",
    "shekel", "new_taiwan_dollar", "baht", "ringgit", "rupiah",
}


@dataclass
class PropertyDef:
    """Parsed property definition from Name(type: config) syntax."""
    name: str
    dnn_type: str          # DNN short name (e.g., "select", "text")
    api_type: str          # Notion API type (e.g., "select", "rich_text")
    options: list[str]     # For select/multi_select option names
    number_format: str     # For number type format
    id_prefix: str         # For unique_id prefix
    relation_db: str       # For relation: target database short ID
    relation_dual: str     # For relation dual: back-reference name


def parse_property_def(text: str) -> tuple[Optional[PropertyDef], Optional[str]]:
    """Parse a property definition like Name(type) or Name(type: config).

    Syntax:
        Name(type)
        Name(type: arg)
        Name(type: a, b, c)
        "Quoted Name"(type: arg)
        Name(relation: DbId, dual: "Back Ref Name")

    Args:
        text: Property definition string (stripped).

    Returns:
        Tuple of (PropertyDef, None) on success, or (None, error_message) on failure.
    """
    text = text.strip()
    if not text:
        return None, "Empty property definition"

    # Parse name — either quoted or unquoted
    if text.startswith('"'):
        # Quoted name: find closing quote before (
        close_quote = text.find('"', 1)
        if close_quote == -1:
            return None, f"Unclosed quote in property name: {text}"
        name = text[1:close_quote]
        rest = text[close_quote + 1:]
    else:
        # Unquoted name: everything before (
        paren_idx = text.find('(')
        if paren_idx == -1:
            return None, f"Missing type in property definition: {text} (expected Name(type))"
        name = text[:paren_idx]
        rest = text[paren_idx:]

    if not name:
        return None, f"Empty property name in: {text}"

    # Parse (type: config)
    if not rest.startswith('('):
        return None, f"Expected '(' after property name in: {text}"
    if not rest.endswith(')'):
        return None, f"Expected ')' at end of property definition: {text}"

    inner = rest[1:-1].strip()  # Contents between parens
    if not inner:
        return None, f"Empty type in property definition: {text}"

    # Split type from config
    # For relation with dual, we need careful parsing:
    #   relation: DbId, dual: "Name"
    # Simple split on first colon for type
    colon_idx = inner.find(':')
    if colon_idx == -1:
        # No config, just type
        dnn_type = inner.strip()
        config_str = ""
    else:
        dnn_type = inner[:colon_idx].strip()
        config_str = inner[colon_idx + 1:].strip()

    # Validate DNN type
    if dnn_type not in CREATABLE_PROPERTY_TYPES:
        return None, (
            f"Unsupported property type '{dnn_type}' in: {text}. "
            f"Supported types: {', '.join(sorted(CREATABLE_PROPERTY_TYPES))}"
        )

    api_type = DNN_TO_API_TYPE[dnn_type]

    # Build PropertyDef based on type
    prop_def = PropertyDef(
        name=name,
        dnn_type=dnn_type,
        api_type=api_type,
        options=[],
        number_format="",
        id_prefix="",
        relation_db="",
        relation_dual="",
    )

    if not config_str:
        # Relation requires a target database ID
        if dnn_type == "relation":
            return None, f"Relation property requires a target database ID: {text} (expected relation: DbId)"
        return prop_def, None

    # Parse config based on type
    if dnn_type in ("select", "multi_select"):
        # Config is comma-separated option names
        prop_def.options = [o.strip() for o in config_str.split(",") if o.strip()]
    elif dnn_type == "number":
        fmt = config_str.strip()
        if fmt not in NUMBER_FORMATS:
            return None, (
                f"Unknown number format '{fmt}' in: {text}. "
                f"Common formats: number, number_with_commas, percent, dollar, euro, pound, yen"
            )
        prop_def.number_format = fmt
    elif dnn_type == "id":
        prop_def.id_prefix = config_str.strip()
    elif dnn_type == "relation":
        # Parse: DbId  or  DbId, dual: "Name"
        parts = _split_relation_config(config_str)
        if isinstance(parts, str):
            return None, parts  # Error message
        prop_def.relation_db, prop_def.relation_dual = parts
    else:
        return None, f"Type '{dnn_type}' does not accept config in: {text}"

    return prop_def, None


def _split_relation_config(config: str) -> tuple[str, str] | str:
    """Parse relation config: 'DbId' or 'DbId, dual: "Name"'.

    Returns:
        (db_id, dual_name) tuple on success, or error string on failure.
    """
    config = config.strip()

    # Check for dual: parameter
    # Find 'dual:' that isn't inside quotes
    dual_idx = -1
    in_quotes = False
    for i, ch in enumerate(config):
        if ch == '"':
            in_quotes = not in_quotes
        elif not in_quotes and config[i:].startswith("dual:"):
            # Check that it's preceded by a comma
            dual_idx = i
            break

    if dual_idx == -1:
        # No dual, just db_id
        db_id = config.strip().rstrip(",").strip()
        if not db_id:
            return "Relation property requires a target database ID"
        return (db_id, "")

    # Split into db_id part and dual part
    db_part = config[:dual_idx].strip().rstrip(",").strip()
    dual_part = config[dual_idx + 5:].strip()  # Skip "dual:"

    if not db_part:
        return "Relation property requires a target database ID before 'dual:'"

    # Parse dual name (may be quoted)
    if dual_part.startswith('"'):
        close = dual_part.find('"', 1)
        if close == -1:
            return f"Unclosed quote in dual property name: {dual_part}"
        dual_name = dual_part[1:close]
    else:
        dual_name = dual_part.strip()

    if not dual_name:
        return "dual: requires a property name for the back-reference"

    return (db_part, dual_name)


def property_def_to_notion(prop_def: PropertyDef) -> dict:
    """Convert a PropertyDef to a Notion API property schema object.

    This builds the property configuration used in POST /databases
    (initial_data_source.properties) or PATCH /data_sources.

    Args:
        prop_def: Parsed property definition.

    Returns:
        Notion API property schema dict (e.g., {"select": {"options": [...]}}).
    """
    api_type = prop_def.api_type

    if api_type == "title":
        return {"title": {}}
    elif api_type == "rich_text":
        return {"rich_text": {}}
    elif api_type == "date":
        return {"date": {}}
    elif api_type == "checkbox":
        return {"checkbox": {}}
    elif api_type == "url":
        return {"url": {}}
    elif api_type == "email":
        return {"email": {}}
    elif api_type == "phone_number":
        return {"phone_number": {}}
    elif api_type == "created_time":
        return {"created_time": {}}
    elif api_type == "created_by":
        return {"created_by": {}}
    elif api_type == "last_edited_time":
        return {"last_edited_time": {}}
    elif api_type == "last_edited_by":
        return {"last_edited_by": {}}
    elif api_type == "select":
        config: dict = {}
        if prop_def.options:
            config["options"] = [{"name": o} for o in prop_def.options]
        return {"select": config}
    elif api_type == "multi_select":
        config = {}
        if prop_def.options:
            config["options"] = [{"name": o} for o in prop_def.options]
        return {"multi_select": config}
    elif api_type == "number":
        config = {}
        if prop_def.number_format:
            config["format"] = prop_def.number_format
        return {"number": config}
    elif api_type == "unique_id":
        config = {}
        if prop_def.id_prefix:
            config["prefix"] = prop_def.id_prefix
        return {"unique_id": config}
    elif api_type == "relation":
        # Relation needs special handling — db_id is resolved at execution time
        # Store the target DB short ID; execution will resolve to full UUID
        # and look up the data source ID
        config = {"database_id": prop_def.relation_db}
        if prop_def.relation_dual:
            config["type"] = "dual_property"
            config["dual_property"] = {"synced_property_name": prop_def.relation_dual}
        else:
            config["type"] = "single_property"
            config["single_property"] = {}
        return {"relation": config}
    else:
        # Fallback: empty config
        return {api_type: {}}


# =============================================================================
# Filter DSL — Parser & Compiler
# =============================================================================


@dataclass
class FilterAtom:
    """Leaf node: Property operator Value."""
    property: str
    operator: str   # =, !=, ~, !~, <, >, <=, >=, ?, !?
    value: str      # empty string for unary ops (?, !?)


@dataclass
class FilterCompound:
    """Binary logic node: AND / OR over children."""
    op: str                                    # "&" or "|"
    children: list                             # list[FilterAtom | FilterCompound]


# Union type alias for type hints
FilterNode = FilterAtom | FilterCompound


# -- Tokenizer ---------------------------------------------------------------

# Token types
_TK_LPAREN = "LPAREN"
_TK_RPAREN = "RPAREN"
_TK_AND = "AND"
_TK_OR = "OR"
_TK_OP = "OP"          # comparison operator
_TK_WORD = "WORD"      # unquoted identifier or value
_TK_QUOTED = "QUOTED"  # double-quoted string
_TK_EOF = "EOF"

# Operators recognised by the tokenizer, longest-first for greedy match
_FILTER_OPS = ("!=", "!~", "!?", "<=", ">=", "=", "~", "<", ">", "?")


class FilterParseError(Exception):
    """Raised when the filter DSL cannot be parsed."""


def _tokenize_filter(s: str) -> list[tuple[str, str]]:
    """Tokenize a filter DSL string into (type, value) pairs."""
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(s)

    while i < n:
        # Skip whitespace
        if s[i].isspace():
            i += 1
            continue

        # Parentheses
        if s[i] == "(":
            tokens.append((_TK_LPAREN, "("))
            i += 1
            continue
        if s[i] == ")":
            tokens.append((_TK_RPAREN, ")"))
            i += 1
            continue

        # Logic operators
        if s[i] == "&":
            tokens.append((_TK_AND, "&"))
            i += 1
            continue
        if s[i] == "|":
            tokens.append((_TK_OR, "|"))
            i += 1
            continue

        # Quoted string (property name or value)
        if s[i] == '"':
            j = i + 1
            parts: list[str] = []
            while j < n:
                if s[j] == "\\" and j + 1 < n:
                    parts.append(s[j + 1])
                    j += 2
                elif s[j] == '"':
                    break
                else:
                    parts.append(s[j])
                    j += 1
            if j >= n:
                raise FilterParseError(f"Unterminated quoted string starting at position {i}")
            tokens.append((_TK_QUOTED, "".join(parts)))
            i = j + 1
            continue

        # Comparison operators (try longest match first)
        matched_op = False
        for op in _FILTER_OPS:
            if s[i:i + len(op)] == op:
                tokens.append((_TK_OP, op))
                i += len(op)
                matched_op = True
                break
        if matched_op:
            continue

        # Unquoted word (identifier or value)
        j = i
        while j < n and not s[j].isspace() and s[j] not in '()&|"':
            # Stop if we hit an operator character that starts a valid op
            # But only if it's not part of the word (e.g., don't split "foo!=bar")
            is_op_start = False
            for op in _FILTER_OPS:
                if s[j:j + len(op)] == op:
                    is_op_start = True
                    break
            if is_op_start:
                break
            j += 1
        if j > i:
            tokens.append((_TK_WORD, s[i:j]))
            i = j
        else:
            raise FilterParseError(f"Unexpected character '{s[i]}' at position {i}")

    tokens.append((_TK_EOF, ""))
    return tokens


# -- Recursive descent parser ------------------------------------------------

class _FilterParser:
    """Parse tokenized filter DSL into a FilterNode AST.

    Grammar (& binds tighter than |):
        expr     → and_expr ('|' and_expr)*
        and_expr → atom ('&' atom)*
        atom     → '(' expr ')' | predicate
        predicate→ PROPERTY OP VALUE
                  | PROPERTY '?' | PROPERTY '!?'
    """

    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> tuple[str, str]:
        return self.tokens[self.pos]

    def _advance(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tok_type: str) -> tuple[str, str]:
        tok = self._peek()
        if tok[0] != tok_type:
            raise FilterParseError(
                f"Expected {tok_type}, got {tok[0]} ('{tok[1]}') at token {self.pos}"
            )
        return self._advance()

    def parse(self) -> FilterNode:
        node = self._parse_or()
        if self._peek()[0] != _TK_EOF:
            tok = self._peek()
            raise FilterParseError(
                f"Unexpected token {tok[0]} ('{tok[1]}') at position {self.pos}"
            )
        return node

    def _parse_or(self) -> FilterNode:
        left = self._parse_and()
        children = [left]
        while self._peek()[0] == _TK_OR:
            self._advance()  # consume '|'
            children.append(self._parse_and())
        if len(children) == 1:
            return children[0]
        return FilterCompound(op="|", children=children)

    def _parse_and(self) -> FilterNode:
        left = self._parse_atom()
        children = [left]
        while self._peek()[0] == _TK_AND:
            self._advance()  # consume '&'
            children.append(self._parse_atom())
        if len(children) == 1:
            return children[0]
        return FilterCompound(op="&", children=children)

    def _parse_atom(self) -> FilterNode:
        tok = self._peek()

        # Parenthesized sub-expression
        if tok[0] == _TK_LPAREN:
            self._advance()  # consume '('
            node = self._parse_or()
            self._expect(_TK_RPAREN)
            return node

        # Predicate: property op value
        # Property is WORD or QUOTED
        if tok[0] not in (_TK_WORD, _TK_QUOTED):
            raise FilterParseError(
                f"Expected property name, got {tok[0]} ('{tok[1]}') at token {self.pos}"
            )
        prop_tok = self._advance()
        prop_name = prop_tok[1]

        # Operator
        op_tok = self._peek()
        if op_tok[0] != _TK_OP:
            raise FilterParseError(
                f"Expected operator after '{prop_name}', got {op_tok[0]} ('{op_tok[1]}') at token {self.pos}"
            )
        self._advance()
        operator = op_tok[1]

        # Unary operators (?, !?) have no value
        if operator in ("?", "!?"):
            return FilterAtom(property=prop_name, operator=operator, value="")

        # Value is WORD or QUOTED
        val_tok = self._peek()
        if val_tok[0] not in (_TK_WORD, _TK_QUOTED):
            raise FilterParseError(
                f"Expected value after '{prop_name} {operator}', "
                f"got {val_tok[0]} ('{val_tok[1]}') at token {self.pos}"
            )
        self._advance()
        return FilterAtom(property=prop_name, operator=operator, value=val_tok[1])


def parse_filter_dsl(s: str) -> FilterNode:
    """Parse a filter DSL string into a FilterNode AST.

    Args:
        s: Filter string, e.g. 'Status = Done & Due < 2024-01-15'

    Returns:
        FilterNode (FilterAtom or FilterCompound)

    Raises:
        FilterParseError: On syntax errors.
    """
    tokens = _tokenize_filter(s)
    parser = _FilterParser(tokens)
    return parser.parse()


# -- Schema-aware compiler ----------------------------------------------------

# Operator → Notion condition name, keyed by DNN property type
_OPERATOR_MAP: dict[str, dict[str, str]] = {
    "title": {
        "=": "equals", "!=": "does_not_equal",
        "~": "contains", "!~": "does_not_contain",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "rich_text": {
        "=": "equals", "!=": "does_not_equal",
        "~": "contains", "!~": "does_not_contain",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "number": {
        "=": "equals", "!=": "does_not_equal",
        "<": "less_than", ">": "greater_than",
        "<=": "less_than_or_equal_to", ">=": "greater_than_or_equal_to",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "select": {
        "=": "equals", "!=": "does_not_equal",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "multi_select": {
        "~": "contains", "!~": "does_not_contain",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "status": {
        "=": "equals", "!=": "does_not_equal",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "date": {
        "=": "equals", "!=": "does_not_equal",
        "<": "before", ">": "after",
        "<=": "on_or_before", ">=": "on_or_after",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "checkbox": {
        "=": "equals",
    },
    "url": {
        "=": "equals", "!=": "does_not_equal",
        "~": "contains", "!~": "does_not_contain",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "email": {
        "=": "equals", "!=": "does_not_equal",
        "~": "contains", "!~": "does_not_contain",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "phone_number": {
        "=": "equals", "!=": "does_not_equal",
        "~": "contains", "!~": "does_not_contain",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "created_time": {
        "=": "equals", "!=": "does_not_equal",
        "<": "before", ">": "after",
        "<=": "on_or_before", ">=": "on_or_after",
        "?": "is_empty", "!?": "is_not_empty",
    },
    "last_edited_time": {
        "=": "equals", "!=": "does_not_equal",
        "<": "before", ">": "after",
        "<=": "on_or_before", ">=": "on_or_after",
        "?": "is_empty", "!?": "is_not_empty",
    },
}


def _find_property(schema: dict, name: str) -> tuple[str, str]:
    """Find a property by name (case-insensitive) and return (canonical_name, notion_type).

    Args:
        schema: Data source object with 'properties' dict.
        name: Property name to look up.

    Returns:
        (canonical_name, notion_type) e.g. ("Status", "select")

    Raises:
        FilterParseError: If property not found.
    """
    properties = schema.get("properties", {})
    name_lower = name.lower()
    for prop_name, prop_dict in properties.items():
        if prop_name.lower() == name_lower:
            return prop_name, prop_dict.get("type", "unknown")
    available = sorted(properties.keys())
    raise FilterParseError(
        f"Unknown property '{name}'. Available: {', '.join(available)}"
    )


_RELATIVE_DATE_RE = re.compile(r'^-(\d+)d$')
_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$')


def _coerce_value(value: str, notion_type: str) -> object:
    """Coerce a string value to the appropriate Python type for the Notion API.

    Security: Every type branch must either return a validated value or raise
    FilterParseError.  No raw user input may reach the Notion API unvalidated.

    Args:
        value: Raw string value from the DSL.
        notion_type: Notion property type.

    Returns:
        Coerced value (str, int, float, bool).

    Raises:
        FilterParseError: If the value fails validation for its type.
    """
    if notion_type == "checkbox":
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise FilterParseError(
            f"Invalid checkbox value '{value}'. Use 1/0, true/false, or yes/no."
        )
    if notion_type == "number":
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            raise FilterParseError(f"Invalid number value '{value}'.")
    if notion_type in ("date", "created_time", "last_edited_time"):
        m = _RELATIVE_DATE_RE.match(value)
        if m:
            from datetime import date, timedelta
            days = int(m.group(1))
            return (date.today() - timedelta(days=days)).isoformat()
        if _ISO_DATE_RE.match(value):
            return value
        raise FilterParseError(
            f"Invalid date value '{value}'. Use YYYY-MM-DD, "
            f"YYYY-MM-DDTHH:MM, or -Nd (e.g. -14d)."
        )
    return value


def _compile_atom(atom: FilterAtom, schema: dict) -> dict:
    """Compile a single FilterAtom to a Notion API filter condition.

    Args:
        atom: Parsed filter atom.
        schema: Data source object with 'properties' dict.

    Returns:
        Notion filter dict, e.g. {"property": "Status", "select": {"equals": "Done"}}

    Raises:
        FilterParseError: On unknown property or unsupported operator.
    """
    canonical_name, notion_type = _find_property(schema, atom.property)

    # Find operator mapping for this type
    type_ops = _OPERATOR_MAP.get(notion_type)
    if type_ops is None:
        raise FilterParseError(
            f"Filtering not supported for property '{canonical_name}' (type: {notion_type})."
        )

    condition_name = type_ops.get(atom.operator)
    if condition_name is None:
        valid_ops = ", ".join(sorted(type_ops.keys()))
        raise FilterParseError(
            f"Operator '{atom.operator}' not valid for '{canonical_name}' "
            f"(type: {notion_type}). Valid: {valid_ops}"
        )

    # Build the Notion filter condition
    if atom.operator in ("?", "!?"):
        # is_empty / is_not_empty — value is a boolean on the condition itself
        return {
            "property": canonical_name,
            notion_type: {condition_name: True},
        }

    value = _coerce_value(atom.value, notion_type)
    return {
        "property": canonical_name,
        notion_type: {condition_name: value},
    }


def compile_filter(node: FilterNode, schema: dict) -> dict:
    """Compile a FilterNode AST into a Notion API filter object.

    Args:
        node: FilterNode from parse_filter_dsl().
        schema: Data source object with 'properties' dict.

    Returns:
        Notion filter dict ready for query_data_source_async().

    Raises:
        FilterParseError: On compilation errors.
    """
    if isinstance(node, FilterAtom):
        return _compile_atom(node, schema)

    if isinstance(node, FilterCompound):
        compiled_children = [compile_filter(child, schema) for child in node.children]
        logic_key = "and" if node.op == "&" else "or"
        return {logic_key: compiled_children}

    raise FilterParseError(f"Unknown filter node type: {type(node)}")


# -- Sort parser --------------------------------------------------------------

def parse_sort_dsl(s: str, schema: dict) -> list[dict]:
    """Parse compact sort DSL into Notion API sort objects.

    Format: "Due desc, Status asc" or "Due desc" or '"Last Contact" desc'
    Default direction is ascending.

    Args:
        s: Sort DSL string.
        schema: Data source object with 'properties' dict.

    Returns:
        List of Notion sort objects.

    Raises:
        FilterParseError: On unknown property or invalid direction.
    """
    if not s.strip():
        return []

    sorts: list[dict] = []
    # Split on commas, but respect quotes
    specs = _split_sort_specs(s)

    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue

        # Parse property name (possibly quoted) and optional direction
        prop_name, direction = _parse_sort_spec(spec)

        canonical_name, _ = _find_property(schema, prop_name)
        sorts.append({
            "property": canonical_name,
            "direction": direction,
        })

    return sorts


def _split_sort_specs(s: str) -> list[str]:
    """Split sort string on commas, respecting quoted property names."""
    specs: list[str] = []
    current: list[str] = []
    in_quotes = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
        elif ch == "," and not in_quotes:
            specs.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        specs.append("".join(current))
    return specs


def _parse_sort_spec(spec: str) -> tuple[str, str]:
    """Parse a single sort spec into (property_name, direction).

    Handles: 'Due desc', 'Due', '"Last Contact" desc', '"Last Contact"'
    """
    spec = spec.strip()
    if spec.startswith('"'):
        # Quoted property name
        end_quote = spec.index('"', 1)
        prop_name = spec[1:end_quote]
        remainder = spec[end_quote + 1:].strip()
    else:
        parts = spec.split(None, 1)
        prop_name = parts[0]
        remainder = parts[1].strip() if len(parts) > 1 else ""

    if not remainder:
        return prop_name, "ascending"

    direction_lower = remainder.lower()
    if direction_lower in ("asc", "ascending"):
        return prop_name, "ascending"
    if direction_lower in ("desc", "descending"):
        return prop_name, "descending"

    raise FilterParseError(
        f"Invalid sort direction '{remainder}'. Use asc or desc."
    )


# -- Columns parser -----------------------------------------------------------

def parse_columns_dsl(s: str, schema: dict) -> list[str]:
    """Parse comma-separated column names, validated against schema.

    Args:
        s: Column names string, e.g. "Name, Status, Due".
        schema: Data source object with 'properties' dict.

    Returns:
        List of canonical property names (order preserved, deduped).

    Raises:
        FilterParseError: On unknown column name.
    """
    if not s.strip():
        return []  # empty means "all columns" (caller handles this)

    result: list[str] = []
    seen: set[str] = set()

    for col in s.split(","):
        col = col.strip()
        if not col:
            continue

        canonical_name, _ = _find_property(schema, col)
        if canonical_name not in seen:
            result.append(canonical_name)
            seen.add(canonical_name)

    return result


def extract_property_value(prop: dict, registry: Optional[IdRegistry] = None) -> str:
    """Extract displayable value from a Notion property.

    Args:
        prop: Property object from page.properties.
        registry: Optional ID registry for resolving relation IDs to short IDs.

    Returns:
        String representation of the value.
    """
    prop_type = prop.get("type", "")

    if prop_type == "title":
        title = prop.get("title", [])
        return "".join(t.get("plain_text", "") for t in title)

    elif prop_type == "rich_text":
        rich_text = prop.get("rich_text", [])
        return "".join(t.get("plain_text", "") for t in rich_text)

    elif prop_type == "number":
        num = prop.get("number")
        return str(num) if num is not None else ""

    elif prop_type == "select":
        select = prop.get("select")
        return select.get("name", "") if select else ""

    elif prop_type == "multi_select":
        options = prop.get("multi_select", [])
        return ",".join(opt.get("name", "") for opt in options)

    elif prop_type == "status":
        status = prop.get("status")
        return status.get("name", "") if status else ""

    elif prop_type == "date":
        date_obj = prop.get("date")
        if not date_obj:
            return ""
        start = date_obj.get("start", "")
        end = date_obj.get("end")
        if end:
            return f"{start}→{end}"
        return start

    elif prop_type == "checkbox":
        return "1" if prop.get("checkbox") else "0"

    elif prop_type == "url":
        return prop.get("url") or ""

    elif prop_type == "email":
        return prop.get("email") or ""

    elif prop_type == "phone_number":
        return prop.get("phone_number") or ""

    elif prop_type == "people":
        people = prop.get("people", [])
        return ",".join(p.get("name", p.get("id", "")) for p in people)

    elif prop_type == "files":
        files = prop.get("files", [])
        return ",".join(f.get("name", "") for f in files)

    elif prop_type == "formula":
        formula = prop.get("formula", {})
        formula_type = formula.get("type", "")
        if formula_type == "string":
            return formula.get("string") or ""
        elif formula_type == "number":
            num = formula.get("number")
            return str(num) if num is not None else ""
        elif formula_type == "boolean":
            return "1" if formula.get("boolean") else "0"
        elif formula_type == "date":
            date_obj = formula.get("date")
            return date_obj.get("start", "") if date_obj else ""
        return ""

    elif prop_type == "relation":
        relations = prop.get("relation", [])
        if registry:
            # Register each related page and return short IDs
            short_ids = []
            for r in relations:
                rel_id = r.get("id", "")
                if rel_id:
                    try:
                        short_ids.append(registry.register(rel_id))
                    except ValueError:
                        # Invalid UUID, use truncated form
                        short_ids.append(rel_id[:8])
            return ",".join(short_ids)
        else:
            # Fallback: truncated UUIDs
            return ",".join(r.get("id", "")[:8] for r in relations)

    elif prop_type == "rollup":
        rollup = prop.get("rollup", {})
        rollup_type = rollup.get("type", "")
        if rollup_type == "number":
            num = rollup.get("number")
            return str(num) if num is not None else ""
        elif rollup_type == "array":
            # Summarize array results
            arr = rollup.get("array", [])
            return f"[{len(arr)} items]"
        return ""

    elif prop_type == "created_time":
        return prop.get("created_time", "")[:10]  # Just date part

    elif prop_type == "last_edited_time":
        return prop.get("last_edited_time", "")[:10]

    elif prop_type == "created_by":
        user = prop.get("created_by", {})
        return user.get("name", user.get("id", ""))

    elif prop_type == "last_edited_by":
        user = prop.get("last_edited_by", {})
        return user.get("name", user.get("id", ""))

    elif prop_type == "unique_id":
        uid = prop.get("unique_id", {})
        prefix = uid.get("prefix", "")
        number = uid.get("number", "")
        return f"{prefix}-{number}" if prefix else str(number)

    return ""


# =============================================================================
# DNN Renderer
# =============================================================================


def _csv_escape(value: str) -> str:
    """Escape a value for CSV output.

    Rules:
    - If value contains comma, quote, or newline → wrap in double quotes
    - Double quotes inside are escaped by doubling: " → ""
    """
    if not value:
        return ""

    needs_quoting = any(c in value for c in ',"\n\r')

    if needs_quoting:
        # Escape quotes by doubling them
        escaped = value.replace('"', '""')
        return f'"{escaped}"'

    return value


# Mapping from Notion block type to DNN marker
BLOCK_TYPE_TO_MARKER = {
    'paragraph': '',
    'heading_1': '# ',
    'heading_2': '## ',
    'heading_3': '### ',
    'bulleted_list_item': '- ',
    'numbered_list_item': '1. ',
    'to_do': None,  # Special handling for checked state
    'toggle': '> ',
    'quote': '| ',
    'callout': '! ',
    'divider': '---',
    'code': None,  # Special handling
    'child_page': '§ ',
    'child_database': '⊞ ',
}


def render_column_to_dnn(
    block: dict,
    registry: IdRegistry,
    level: int,
    mode: str,
    column_number: int
) -> list[str]:
    """Render a column block with its position number.

    Args:
        block: Notion column block object.
        registry: ID registry for short ID assignment.
        level: Current nesting level.
        mode: "edit" or "view".
        column_number: 1-indexed column position.

    Returns:
        List of DNN lines for this column and its children.
    """
    lines = []
    block_id = block.get("id", "")

    # Generate short ID
    short_id = registry.register(block_id) if mode == "edit" else ""

    # Build indent
    indent = "  " * level

    # Render column marker with number
    _append_dnn_line(lines, mode, short_id, indent, f"!col {column_number}")

    # Render column's children
    children = block.get("_children", [])
    for child in children:
        lines.extend(render_block_to_dnn(child, registry, level + 1, mode))

    return lines


def render_block_to_dnn(
    block: dict,
    registry: IdRegistry,
    level: int = 0,
    mode: str = "edit"
) -> list[str]:
    """Render a Notion block to DNN format.

    Args:
        block: Notion block object.
        registry: ID registry for short ID assignment.
        level: Current nesting level.
        mode: "edit" (with IDs) or "view" (no IDs).

    Returns:
        List of DNN lines for this block and its children.
    """
    lines = []
    block_id = block.get("id", "")
    block_type = block.get("type", "unsupported")

    # Generate short ID
    short_id = registry.register(block_id) if mode == "edit" else ""

    # Build indent
    indent = "  " * level

    # Handle opaque blocks — all get !type~ suffix
    if block_type in OPAQUE_BLOCK_TYPES:
        type_data = block.get(block_type, {})
        caption = ""
        if "caption" in type_data:
            caption = notion_rich_text_to_dnn(type_data["caption"])

        placeholder = f"!{block_type}~"
        if caption:
            placeholder += f" ({caption})"

        _append_dnn_line(lines, mode, short_id, indent, placeholder)
        return lines

    # Get block content
    type_data = block.get(block_type, {})

    # Handle special block types
    if block_type == "divider":
        _append_dnn_line(lines, mode, short_id, indent, "---")
        return lines

    if block_type == "column_list":
        # Render column_list with !cols N marker
        children = block.get("_children", [])
        num_columns = len(children)

        _append_dnn_line(lines, mode, short_id, indent, f"!cols {num_columns}")

        # Render each column with its number
        for i, column in enumerate(children, 1):
            lines.extend(render_column_to_dnn(column, registry, level + 1, mode, i))

        return lines

    if block_type == "table":
        # Render table with !table N marker (N = table_width)
        table_width = type_data.get("table_width", 0)
        _append_dnn_line(lines, mode, short_id, indent, f"!table {table_width}")

        # Render table_row children
        children = block.get("_children", [])
        for child in children:
            lines.extend(render_block_to_dnn(child, registry, level + 1, mode))

        return lines

    if block_type == "table_row":
        # Render as | cell | cell | with pipe-delimited cells
        cells = type_data.get("cells", [])
        cell_texts = []
        for cell in cells:
            cell_dnn = notion_rich_text_to_dnn(cell)
            # Escape literal pipes in cell content
            cell_dnn = cell_dnn.replace('|', '\\|')
            cell_texts.append(cell_dnn)
        row_content = "| " + " | ".join(cell_texts) + " |"
        _append_dnn_line(lines, mode, short_id, indent, row_content)
        return lines

    if block_type == "code":
        language = type_data.get("language", "")
        rich_text = type_data.get("rich_text", [])
        code_content = "".join(t.get("plain_text", "") for t in rich_text)

        # Count longest backtick run in content to avoid collision
        backtick_runs = re.findall(r'`+', code_content)
        longest = max((len(r) for r in backtick_runs), default=0)
        fence = '`' * max(3, longest + 1)

        _append_dnn_line(lines, mode, short_id, indent, f"{fence}{language}")

        for code_line in code_content.split("\n"):
            lines.append(code_line)

        lines.append(f"{indent}{fence}")
        return lines

    if block_type == "to_do":
        checked = type_data.get("checked", False)
        marker = "[x] " if checked else "[ ] "
        rich_text = type_data.get("rich_text", [])
        content = notion_rich_text_to_dnn(rich_text)
        _append_dnn_line(lines, mode, short_id, indent, f"{marker}{content}")

    elif block_type == "child_page":
        title = type_data.get("title", "Untitled")
        _append_dnn_line(lines, mode, short_id, indent, f"§ {title}")

    elif block_type == "child_database":
        title = type_data.get("title", "Untitled")
        _append_dnn_line(lines, mode, short_id, indent, f"⊞ {title}")

    elif block_type in ("heading_1", "heading_2", "heading_3"):
        # Check if it's a toggle heading
        is_toggle = type_data.get("is_toggleable", False)
        rich_text = type_data.get("rich_text", [])
        content = notion_rich_text_to_dnn(rich_text)

        if is_toggle:
            marker = {"heading_1": "># ", "heading_2": ">## ", "heading_3": ">### "}[block_type]
        else:
            marker = BLOCK_TYPE_TO_MARKER[block_type]

        _append_dnn_line(lines, mode, short_id, indent, f"{marker}{content}")

    elif block_type == "callout":
        # Callout with optional color
        rich_text = type_data.get("rich_text", [])
        content = notion_rich_text_to_dnn(rich_text)
        color = type_data.get("color", "default")

        # Build marker: !color or just !
        if color and color != "default" and color.endswith("_background"):
            color_name = color.replace("_background", "")
            marker = f"!{color_name} "
        else:
            marker = "! "

        _append_dnn_line(lines, mode, short_id, indent, f"{marker}{content}")

    else:
        # Standard block with rich_text
        marker = BLOCK_TYPE_TO_MARKER.get(block_type, '')
        rich_text = type_data.get("rich_text", [])
        content = notion_rich_text_to_dnn(rich_text)
        _append_dnn_line(lines, mode, short_id, indent, f"{marker}{content}")

    # Render children
    children = block.get("_children", [])
    for child in children:
        lines.extend(render_block_to_dnn(child, registry, level + 1, mode))

    return lines


def render_page_to_dnn(
    page: dict,
    blocks: list[dict],
    registry: IdRegistry,
    mode: str = "edit"
) -> str:
    """Render a complete page to DNN format.

    Args:
        page: Page metadata object.
        blocks: List of block objects (with _children populated).
        registry: ID registry.
        mode: "edit" or "view".

    Returns:
        Complete DNN document string.
    """
    lines = []

    # Header
    lines.append("@dnn 1")

    page_id = page.get("id", "")
    if mode == "edit":
        short_id = registry.register(page_id)
        lines.append(f"@page {short_id}")
    else:
        lines.append(f"@page {page_id}")

    icon = page.get("icon") or {}
    if icon.get("type") == "emoji":
        lines.append(f"@icon {icon['emoji']}")
    elif icon.get("type") == "external":
        url = icon.get("external", {}).get("url", "")
        if url:
            lines.append(f"@icon {url}")

    title = get_page_title(page)
    lines.append(f"@title {title}")

    # Empty line before blocks
    lines.append("")

    # Render blocks
    for block in blocks:
        lines.extend(render_block_to_dnn(block, registry, level=0, mode=mode))

    return "\n".join(lines)


def render_blocks_to_dnn(
    blocks: list[dict],
    registry: IdRegistry,
    mode: str = "edit"
) -> str:
    """Render a list of blocks to DNN format (no page header).

    Used when reading individual blocks rather than full pages.

    Args:
        blocks: List of block objects (with _children populated).
        registry: ID registry.
        mode: "edit" or "view".

    Returns:
        DNN formatted string of the blocks.
    """
    lines = []
    lines.append("@dnn 1")
    lines.append("@block")  # Indicate this is a block read, not a page
    lines.append("")

    for block in blocks:
        lines.extend(render_block_to_dnn(block, registry, level=0, mode=mode))

    return "\n".join(lines)


def render_database_to_dnn(
    database: dict,
    rows: list[dict],
    registry: IdRegistry,
    mode: str = "edit",
    columns: Optional[list[str]] = None,
    has_more: bool = False
) -> str:
    """Render a database to DNN format with TSV rows.

    Args:
        database: Database metadata object with schema.
        rows: List of row (page) objects from query.
        registry: ID registry.
        mode: "edit" or "view".
        columns: Optional list of property names to include.
        has_more: True if more rows exist beyond what's returned.

    Returns:
        Complete DNN database document string.
    """
    lines = []

    # Header
    lines.append("@dnn 1")

    # In API 2025-09-03, database_id and data_source_id are different
    # database_id is the container, data_source_id has the schema
    database_id = database.get("database_id", database.get("id", ""))
    data_source_id = database.get("id", "")

    if mode == "edit":
        db_short_id = registry.register(database_id)
        lines.append(f"@db {db_short_id}")
        ds_short_id = registry.register(data_source_id)
        lines.append(f"@ds {ds_short_id}")
    else:
        lines.append(f"@db {database_id}")
        lines.append(f"@ds {data_source_id}")

    title = get_database_title(database)
    lines.append(f"@title {title}")

    # Get property schema
    properties = database.get("properties", {})

    # Filter columns if specified, otherwise use all
    if columns:
        prop_names = [n for n in columns if n in properties]
    else:
        prop_names = list(properties.keys())

    # Sort to put title first
    def sort_key(name):
        prop = properties[name]
        if prop.get("type") == "title":
            return (0, name)
        return (1, name)

    prop_names.sort(key=sort_key)

    # Build @cols line with human-readable names
    col_defs = []
    for name in prop_names:
        prop = properties[name]
        prop_type = prop.get("type", "unknown")
        dnn_type = PROPERTY_TYPE_MAP.get(prop_type, prop_type)

        # Quote column names that need escaping (spaces, parens, special chars)
        if any(c in name for c in ' \t()[]"\\'):
            # Escape quotes and backslashes inside the name
            escaped_name = name.replace('\\', '\\\\').replace('"', '\\"')
            col_defs.append(f'"{escaped_name}"({dnn_type})')
        else:
            col_defs.append(f"{name}({dnn_type})")

    lines.append(f"@cols (icon),{','.join(col_defs)}")

    # Row count with truncation indicator
    row_count = len(rows)
    if has_more:
        lines.append(f"@rows {row_count} (truncated)")
    else:
        lines.append(f"@rows {row_count}")

    # Empty line before rows
    lines.append("")

    # Render rows as CSV
    for row in rows:
        row_id = row.get("id", "")
        row_props = row.get("properties", {})

        icon = row.get("icon") or {}
        icon_str = ""
        if icon.get("type") == "emoji":
            icon_str = icon.get("emoji", "")

        if mode == "edit":
            row_sid = registry.register(row_id)
            cells = [icon_str, row_sid]  # icon, then row ID
        else:
            cells = [icon_str]

        for name in prop_names:
            if name in row_props:
                value = extract_property_value(row_props[name], registry=registry)
            else:
                value = ""
            cells.append(_csv_escape(value))

        lines.append(",".join(cells))

    return "\n".join(lines)


# =============================================================================
# Apply Script Parser
# =============================================================================


class ApplyCommand(Enum):
    """Types of commands in apply scripts."""
    ADD = "+"           # Add blocks
    DELETE = "x"        # Delete/archive blocks
    MOVE = "m"          # Move block
    UPDATE = "u"        # Update block text
    TOGGLE = "t"        # Toggle todo checkbox
    ADD_PAGE = "+page"  # Create page
    MOVE_PAGE = "mpage" # Move page
    DELETE_PAGE = "xpage"  # Archive page
    UPDATE_PAGE = "upage"  # Update page properties
    COPY_PAGE = "cpage"    # Copy page
    ADD_ROW = "+row"    # Add database row
    UPDATE_ROW = "urow" # Update row properties
    DELETE_ROW = "xrow" # Delete row
    ADD_DB = "+db"         # Create database
    ADD_PROP = "+prop"     # Add property to database
    DELETE_PROP = "xprop"  # Delete property from database
    RENAME_PROP = "uprop"  # Rename property in database


@dataclass
class ApplyOp:
    """Parsed apply operation."""
    command: ApplyCommand
    line_num: int
    # For + command
    parent: Optional[str] = None
    after: Optional[str] = None
    content_blocks: list[DnnBlock] = field(default_factory=list)
    # For x command (can have multiple targets)
    targets: list[str] = field(default_factory=list)
    # For m command
    source: Optional[str] = None
    dest_parent: Optional[str] = None
    dest_after: Optional[str] = None
    # For u command
    target: Optional[str] = None
    new_text: Optional[str] = None
    # For t command
    checked: Optional[bool] = None
    # For page commands
    title: Optional[str] = None
    icon: Optional[str] = None
    cover: Optional[str] = None
    props: dict[str, str] = field(default_factory=dict)
    # For row commands
    database: Optional[str] = None
    row_values: dict[str, str] = field(default_factory=dict)
    # For +db command
    property_defs: list['PropertyDef'] = field(default_factory=list)
    # For +prop command (single property def)
    property_def: Optional['PropertyDef'] = None
    # For uprop command (rename)
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    # For +page positional insertion
    position: Optional[str] = None  # "start" for pos=start


@dataclass
class ApplyParseResult:
    """Result of parsing an apply script."""
    operations: list[ApplyOp]
    errors: list[DnnParseError]


# Regex patterns for command parsing
ADD_CMD_PATTERN = re.compile(r'^\+\s+parent=(\S+)(?:\s+after=(\S+))?$')
DELETE_CMD_PATTERN = re.compile(r'^x\s+(.+)$')
MOVE_CMD_PATTERN = re.compile(r'^m\s+(\S+)\s+->\s+parent=(\S+)(?:\s+after=(\S+))?$')
# Patterns for detecting empty after= (common user error attempting to insert at beginning)
ADD_CMD_EMPTY_AFTER = re.compile(r'^\+\s+parent=(\S+)\s+after=\s*$')
MOVE_CMD_EMPTY_AFTER = re.compile(r'^m\s+(\S+)\s+->\s+parent=(\S+)\s+after=\s*$')
MOVE_CMD_MISSING_PARENT = re.compile(r'^m\s+(\S+)\s+->\s+after=(\S+)$')
UPDATE_CMD_PATTERN = re.compile(r'^u\s+(\S+)\s+=\s+"(.*)"\s*$')
TOGGLE_CMD_PATTERN = re.compile(r'^t\s+(\S+)\s+=\s+([01])$')


def _decode_escape_sequences(text: str) -> str:
    """Decode common escape sequences in a string.

    Handles: \\n (newline), \\t (tab), \\\\ (backslash), \\" (quote)

    Args:
        text: String with potential escape sequences.

    Returns:
        String with escape sequences converted to actual characters.
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char == 'n':
                result.append('\n')
                i += 2
                continue
            elif next_char == 't':
                result.append('\t')
                i += 2
                continue
            elif next_char == '\\':
                result.append('\\')
                i += 2
                continue
            elif next_char == '"':
                result.append('"')
                i += 2
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)

# Page command patterns
# Groups: 1=parent, 2=after (optional), 3=pos (optional), 4=title, 5=icon, 6=cover
ADD_PAGE_PATTERN = re.compile(
    r'^\+page\s+parent=(\S+)'
    r'(?:\s+after=(\S+))?'
    r'(?:\s+pos=(\S+))?'
    r'\s+title="([^"]*)"'
    r'(?:\s+icon=(\S+))?(?:\s+cover=(\S+))?$'
)
MOVE_PAGE_PATTERN = re.compile(r'^mpage\s+(\S+)\s+->\s+parent=(\S+)$')
DELETE_PAGE_PATTERN = re.compile(r'^xpage\s+(\S+)$')
UPDATE_PAGE_PATTERN = re.compile(
    r'^upage\s+(\S+)(?:\s+=\s+"([^"]*)")?'
    r'(?:\s+icon=(\S+))?(?:\s+cover=(\S+))?$'
)
COPY_PAGE_PATTERN = re.compile(
    r'^cpage\s+(\S+)\s+->\s+parent=(\S+)(?:\s+title="([^"]*)")?$'
)

# Row command patterns
ADD_ROW_PATTERN = re.compile(r'^\+row\s+db=(\S+)(?:\s+icon=(\S+))?$')
UPDATE_ROW_PATTERN = re.compile(r'^urow\s+(\S+)$')
DELETE_ROW_PATTERN = re.compile(r'^xrow\s+(\S+)$')

# Database command patterns
ADD_DB_PATTERN = re.compile(
    r'^\+db\s+parent=(\S+)\s+title="([^"]*)"'
    r'(?:\s+icon=(\S+))?$'
)
ADD_PROP_PATTERN = re.compile(r'^\+prop\s+db=(\S+)\s+(.+)$')
DELETE_PROP_PATTERN = re.compile(r'^xprop\s+db=(\S+)\s+(.+)$')
RENAME_PROP_PATTERN = re.compile(
    r'^uprop\s+db=(\S+)\s+(.+?)\s+->\s+(.+)$'
)


def parse_apply_script(script: str, registry: IdRegistry) -> ApplyParseResult:
    """Parse an apply script into operations.

    Args:
        script: The apply script text.
        registry: ID registry for resolving references.

    Returns:
        ApplyParseResult with operations and any errors.
    """
    lines = script.split('\n')
    operations: list[ApplyOp] = []
    errors: list[DnnParseError] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # 0-based line numbers in output (matches array indexing)
        line_num = i

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # Try each command pattern
        op = None

        # Check for empty after= parameter (common error: trying to insert at beginning)
        if ADD_CMD_EMPTY_AFTER.match(line):
            errors.append(DnnParseError(
                code="EMPTY_AFTER_PARAM",
                message="after= requires a block ID. Note: Notion API does not support "
                        "inserting at the beginning of a page. Omit after= to append "
                        "at end, or use after=BLOCK_ID to insert after a specific block.",
                line=line_num,
                excerpt=f"{line_num}|{line}",
                suggestions=[
                    "Remove 'after=' to append at end",
                    "Use 'after=BLOCK_ID' to insert after specific block",
                ]
            ))
            i += 1
            continue

        if MOVE_CMD_EMPTY_AFTER.match(line):
            errors.append(DnnParseError(
                code="EMPTY_AFTER_PARAM",
                message="after= requires a block ID. Note: Notion API does not support "
                        "moving to the beginning of a parent. Omit after= to move to "
                        "end, or use after=BLOCK_ID to move after a specific block.",
                line=line_num,
                excerpt=f"{line_num}|{line}",
                suggestions=[
                    "Remove 'after=' to move to end",
                    "Use 'after=BLOCK_ID' to move after specific block",
                ]
            ))
            i += 1
            continue

        # Check for move missing parent= (common error: m X -> after=Y without parent=)
        match = MOVE_CMD_MISSING_PARENT.match(line)
        if match:
            errors.append(DnnParseError(
                code="MOVE_MISSING_PARENT",
                message="Move requires parent=. To reposition within same parent, use: "
                        "m SOURCE -> parent=PARENT after=SIBLING",
                line=line_num,
                excerpt=f"{line_num}|{line[:60]}",
                suggestions=[
                    f"m {match.group(1)} -> parent=??? after={match.group(2)}"
                ]
            ))
            i += 1
            continue

        # + parent=X [after=Y]
        match = ADD_CMD_PATTERN.match(line)
        if match:
            parent = match.group(1)
            after = match.group(2)
            op = ApplyOp(
                command=ApplyCommand.ADD,
                line_num=line_num,
                parent=parent,
                after=after
            )
            # Collect indented content lines (code-block-aware)
            i += 1
            content_lines = []
            in_code_block = False
            code_fence_count = 0
            while i < len(lines):
                content_line = lines[i]
                if in_code_block:
                    # Inside code block: collect all lines until closing fence
                    content_lines.append(content_line)
                    stripped_cl = content_line.lstrip()
                    expected_close = '`' * code_fence_count
                    if stripped_cl == expected_close:
                        in_code_block = False
                    i += 1
                elif content_line and content_line[0] in ' \t':
                    content_lines.append(content_line)
                    # Check if this line opens a code block
                    stripped_cl = content_line.lstrip()
                    fence_match = re.match(r'^(`{3,})\w*$', stripped_cl)
                    if fence_match:
                        in_code_block = True
                        code_fence_count = len(fence_match.group(1))
                    i += 1
                else:
                    break
            # Parse content as DNN blocks (without IDs)
            if content_lines:
                op.content_blocks = _parse_content_blocks(content_lines, errors, line_num)
            operations.append(op)
            continue

        # x ID1 [ID2 ...]
        match = DELETE_CMD_PATTERN.match(line)
        if match:
            targets = match.group(1).split()
            op = ApplyOp(
                command=ApplyCommand.DELETE,
                line_num=line_num,
                targets=targets
            )
            operations.append(op)
            i += 1
            continue

        # m SOURCE -> parent=DEST [after=Y]
        match = MOVE_CMD_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.MOVE,
                line_num=line_num,
                source=match.group(1),
                dest_parent=match.group(2),
                dest_after=match.group(3)
            )
            operations.append(op)
            i += 1
            continue

        # u ID = "new text"
        match = UPDATE_CMD_PATTERN.match(line)
        if match:
            # Decode escape sequences (\n, \t, \\, \") in the text
            decoded_text = _decode_escape_sequences(match.group(2))
            op = ApplyOp(
                command=ApplyCommand.UPDATE,
                line_num=line_num,
                target=match.group(1),
                new_text=decoded_text
            )
            operations.append(op)
            i += 1
            continue

        # t ID = 0|1
        match = TOGGLE_CMD_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.TOGGLE,
                line_num=line_num,
                target=match.group(1),
                checked=match.group(2) == '1'
            )
            operations.append(op)
            i += 1
            continue

        # +page parent=X [after=Y] [pos=start] title="Z" [icon=W] [cover=V]
        match = ADD_PAGE_PATTERN.match(line)
        if match:
            after_id = match.group(2)
            pos_value = match.group(3)
            op = ApplyOp(
                command=ApplyCommand.ADD_PAGE,
                line_num=line_num,
                parent=match.group(1),
                after=after_id,
                position=pos_value,
                title=match.group(4),
                icon=match.group(5),
                cover=match.group(6)
            )
            # Collect content blocks (code-block-aware)
            i += 1
            content_lines = []
            in_code_block = False
            code_fence_count = 0
            while i < len(lines):
                content_line = lines[i]
                if in_code_block:
                    content_lines.append(content_line)
                    stripped_cl = content_line.lstrip()
                    expected_close = '`' * code_fence_count
                    if stripped_cl == expected_close:
                        in_code_block = False
                    i += 1
                elif content_line and content_line[0] in ' \t':
                    content_lines.append(content_line)
                    stripped_cl = content_line.lstrip()
                    fence_match = re.match(r'^(`{3,})\w*$', stripped_cl)
                    if fence_match:
                        in_code_block = True
                        code_fence_count = len(fence_match.group(1))
                    i += 1
                else:
                    break
            if content_lines:
                op.content_blocks = _parse_content_blocks(content_lines, errors, line_num)
            operations.append(op)
            continue

        # mpage SOURCE -> parent=DEST
        match = MOVE_PAGE_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.MOVE_PAGE,
                line_num=line_num,
                source=match.group(1),
                dest_parent=match.group(2)
            )
            operations.append(op)
            i += 1
            continue

        # xpage ID
        match = DELETE_PAGE_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.DELETE_PAGE,
                line_num=line_num,
                target=match.group(1)
            )
            operations.append(op)
            i += 1
            continue

        # upage ID [= "title"] [icon=X] [cover=Y]
        match = UPDATE_PAGE_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.UPDATE_PAGE,
                line_num=line_num,
                target=match.group(1),
                title=match.group(2),
                icon=match.group(3),
                cover=match.group(4)
            )
            operations.append(op)
            i += 1
            continue

        # cpage SOURCE -> parent=DEST [title="X"]
        match = COPY_PAGE_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.COPY_PAGE,
                line_num=line_num,
                source=match.group(1),
                dest_parent=match.group(2),
                title=match.group(3)
            )
            operations.append(op)
            i += 1
            continue

        # +row db=X
        match = ADD_ROW_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.ADD_ROW,
                line_num=line_num,
                database=match.group(1),
                icon=match.group(2)
            )
            # Next line should have tab-separated values (accepts spaces or tabs)
            i += 1
            if i < len(lines) and lines[i] and lines[i][0] in ' \t':
                op.row_values = _parse_row_values(lines[i].strip())
                i += 1
            operations.append(op)
            continue

        # urow ID
        match = UPDATE_ROW_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.UPDATE_ROW,
                line_num=line_num,
                target=match.group(1)
            )
            # Next line should have values (accepts spaces or tabs)
            i += 1
            if i < len(lines) and lines[i] and lines[i][0] in ' \t':
                op.row_values = _parse_row_values(lines[i].strip())
                i += 1
            operations.append(op)
            continue

        # xrow ID
        match = DELETE_ROW_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.DELETE_ROW,
                line_num=line_num,
                target=match.group(1)
            )
            operations.append(op)
            i += 1
            continue

        # +db parent=X title="Y" [icon=Z]
        match = ADD_DB_PATTERN.match(line)
        if match:
            op = ApplyOp(
                command=ApplyCommand.ADD_DB,
                line_num=line_num,
                parent=match.group(1),
                title=match.group(2),
                icon=match.group(3),
            )
            # Collect indented property definition lines
            i += 1
            has_title_prop = False
            while i < len(lines):
                prop_line = lines[i]
                if not prop_line or prop_line[0] not in ' \t':
                    break
                stripped = prop_line.strip()
                if not stripped or stripped.startswith('#'):
                    i += 1
                    continue
                prop_def, prop_err = parse_property_def(stripped)
                if prop_err:
                    errors.append(DnnParseError(
                        code="BAD_PROPERTY_DEF",
                        message=prop_err,
                        line=i,
                        excerpt=f"{i}|{stripped[:60]}",
                    ))
                else:
                    op.property_defs.append(prop_def)
                    if prop_def.dnn_type == "title":
                        has_title_prop = True
                i += 1
            if not has_title_prop and not errors:
                errors.append(DnnParseError(
                    code="MISSING_TITLE_PROPERTY",
                    message="+db requires at least one title property",
                    line=line_num,
                    excerpt=f"{line_num}|{line[:60]}",
                    suggestions=["Add a line like: Name(title)"]
                ))
            operations.append(op)
            continue

        # +prop db=X PropDef
        match = ADD_PROP_PATTERN.match(line)
        if match:
            db_id = match.group(1)
            prop_text = match.group(2).strip()
            prop_def, prop_err = parse_property_def(prop_text)
            if prop_err:
                errors.append(DnnParseError(
                    code="BAD_PROPERTY_DEF",
                    message=prop_err,
                    line=line_num,
                    excerpt=f"{line_num}|{line[:60]}",
                ))
            else:
                op = ApplyOp(
                    command=ApplyCommand.ADD_PROP,
                    line_num=line_num,
                    database=db_id,
                    property_def=prop_def,
                )
                operations.append(op)
            i += 1
            continue

        # xprop db=X PropertyName
        match = DELETE_PROP_PATTERN.match(line)
        if match:
            db_id = match.group(1)
            prop_name = match.group(2).strip()
            # Strip quotes if present
            if prop_name.startswith('"') and prop_name.endswith('"'):
                prop_name = prop_name[1:-1]
            op = ApplyOp(
                command=ApplyCommand.DELETE_PROP,
                line_num=line_num,
                database=db_id,
                target=prop_name,
            )
            operations.append(op)
            i += 1
            continue

        # uprop db=X OldName -> NewName
        match = RENAME_PROP_PATTERN.match(line)
        if match:
            db_id = match.group(1)
            old_name = match.group(2).strip()
            new_name = match.group(3).strip()
            # Strip quotes if present
            if old_name.startswith('"') and old_name.endswith('"'):
                old_name = old_name[1:-1]
            if new_name.startswith('"') and new_name.endswith('"'):
                new_name = new_name[1:-1]
            op = ApplyOp(
                command=ApplyCommand.RENAME_PROP,
                line_num=line_num,
                database=db_id,
                old_name=old_name,
                new_name=new_name,
            )
            operations.append(op)
            i += 1
            continue

        # Unknown command
        errors.append(DnnParseError(
            code="UNKNOWN_COMMAND",
            message="Unknown apply command",
            line=line_num,
            excerpt=f"{line_num}|{line[:50]}",
            suggestions=["Valid commands: +, x, m, u, t, +page, mpage, xpage, upage, cpage, +row, urow, xrow, +db, +prop, xprop, uprop"]
        ))
        i += 1

    return ApplyParseResult(operations=operations, errors=errors)


def _parse_content_blocks(lines: list[str], errors: list[DnnParseError], start_line: int) -> list[DnnBlock]:
    """Parse indented content lines into DNN blocks.

    Content blocks don't have IDs - they're new blocks to be created.
    Handles code block state: backtick-counted fences collect raw lines
    until a matching closing fence.
    """
    blocks = []
    # Code block state
    current_code_block: Optional[DnnBlock] = None
    code_block_indent: int = 0
    code_block_backtick_count: int = 3

    for offset, line in enumerate(lines):
        stripped = line.lstrip()

        # CODE state: collect raw lines until closing fence
        if current_code_block is not None:
            expected_close = '`' * code_block_backtick_count
            if stripped == expected_close:
                # Closing fence — finalize code block
                current_code_block = None
                continue
            # Raw content line (preserve as-is)
            current_code_block.raw_lines.append(line)
            continue

        if not stripped:
            continue

        # Calculate indent level
        indent = len(line) - len(stripped)
        level = indent // 2

        # Parse block type from content
        block_type, content, attrs, block_warnings = parse_block_type(stripped)

        block = DnnBlock(
            short_id="",  # No ID for new blocks
            level=level,
            block_type=block_type,
            content=content,
            warnings=block_warnings,
            **attrs
        )
        blocks.append(block)

        # Enter CODE state if this is a code block
        if block_type == 'code':
            current_code_block = block
            code_block_indent = indent
            # Count backticks from the original line (content is empty for code blocks)
            backtick_match = re.match(r'^(`{3,})', stripped)
            code_block_backtick_count = len(backtick_match.group(1)) if backtick_match else 3
            continue

    # Handle unterminated code block
    if current_code_block is not None:
        closing_fence = '`' * code_block_backtick_count
        errors.append(DnnParseError(
            code="CODE_BLOCK_UNTERMINATED",
            message="Code block not closed in content",
            line=start_line + len(lines),
            excerpt="EOF",
            suggestions=[f"Add closing {closing_fence} at same indent level"],
        ))

    # Auto-group consecutive table_row blocks into a table parent.
    # If table rows appear without an explicit !table N parent,
    # wrap them in a synthetic table block.
    blocks = _auto_group_table_rows(blocks)

    return blocks


def _auto_group_table_rows(blocks: list[DnnBlock]) -> list[DnnBlock]:
    """Auto-group consecutive table_row blocks into table parents.

    Consecutive table_row blocks at the same level that don't already
    have a table parent are wrapped in a synthetic !table N block.
    Separator rows (table_separator) are silently dropped.
    """
    result = []
    i = 0
    while i < len(blocks):
        block = blocks[i]

        # Drop separator rows
        if block.block_type == 'table_separator':
            i += 1
            continue

        # Check for consecutive table rows at same level
        if block.block_type == 'table_row':
            # Collect all consecutive table rows at this level
            rows = []
            level = block.level
            while i < len(blocks) and blocks[i].level == level and \
                    blocks[i].block_type in ('table_row', 'table_separator'):
                if blocks[i].block_type == 'table_row':
                    rows.append(blocks[i])
                i += 1

            # Infer table_width from first row's cell count
            first_cells = parse_table_row_cells(rows[0].content)
            table_width = len(first_cells)

            # Create synthetic table parent
            table_block = DnnBlock(
                short_id="",
                level=level,
                block_type='table',
                content=str(table_width),
                structural=True,
                children=rows,
            )
            # Adjust row levels to be children of the table
            for row in rows:
                row.level = level + 1
            result.append(table_block)
        else:
            result.append(block)
            i += 1

    return result


def _parse_row_values(line: str) -> dict[str, str]:
    """Parse tab-separated col_id=value pairs."""
    values = {}
    for pair in line.split('\t'):
        if '=' in pair:
            key, value = pair.split('=', 1)
            values[key.strip()] = value.strip()
    return values


# =============================================================================
# Self-Healing Error Messages
# =============================================================================


def _error(code: str, message: str, hint: str | None = None, ref: str | None = None) -> str:
    """Format error with optional self-healing hint.

    Args:
        code: Error code (e.g., UNKNOWN_ID, REF_GONE)
        message: Human-readable description
        hint: Suggestion on how to fix the issue
        ref: The reference that failed (for context)

    Returns:
        Formatted error string with hint if provided.
    """
    parts = [f"error: {code} - {message}"]
    if ref:
        parts.append(f"ref: {ref}")
    if hint:
        parts.append(f"hint: {hint}")
    return "\n".join(parts)


# Common error hints
HINTS = {
    "unknown_id": "Use notion_search to find the page/database by title, or provide a full Notion UUID/URL.",
    "ref_gone": "The object may be deleted, in trash, or not shared with this integration. Check Notion UI or search by title.",
    "missing_capability": "Share the page/database with the integration: open in Notion → Share → invite the integration.",
    "no_data_sources": "Unusual database state. Try opening in Notion UI first, or use the database's page URL.",
    "rate_limited": "Too many requests. Wait a moment and try again.",
    "invalid_token": "Token is invalid or expired. Check secrets/notion_token file.",
    "db_params_on_page": "filter/sort/columns only work on databases. This ref resolved to a page or block.",
    "filter_parse_error": "Filter syntax: Property op Value, joined with & (and) or | (or). Quote names with spaces: \"Last Contact\" >= 2024-01-01",
    "db_params_batch": "filter/sort/columns require exactly one ref. Remove DB params or reduce to a single ref.",  # kept for backwards compat
}


# =============================================================================
# Block Mutations (Apply Tool Execution)
# =============================================================================


def dnn_block_to_notion(block: DnnBlock, registry: IdRegistry) -> dict:
    """Convert a DnnBlock to Notion API block format.

    Args:
        block: Parsed DNN block.
        registry: ID registry for resolving refs in content.

    Returns:
        Notion API block object ready for append.
    """
    block_type = block.block_type

    # Parse inline formatting to rich_text
    rich_text = rich_text_spans_to_notion(parse_inline_formatting(block.content), registry)

    if block_type == "paragraph":
        return {
            "type": "paragraph",
            "paragraph": {"rich_text": rich_text}
        }

    # Heading blocks: consolidated from 3 nearly-identical cases.
    # All headings share the same structure with rich_text and is_toggleable.
    elif block_type in ("heading_1", "heading_2", "heading_3"):
        return {
            "type": block_type,
            block_type: {
                "rich_text": rich_text,
                "is_toggleable": block.is_toggle
            }
        }

    # Simple rich text blocks: consolidated from 4 nearly-identical cases.
    # These blocks only contain rich_text with no additional properties.
    elif block_type in ("bulleted_list_item", "numbered_list_item", "toggle", "quote"):
        return {
            "type": block_type,
            block_type: {"rich_text": rich_text}
        }

    elif block_type == "to_do":
        return {
            "type": "to_do",
            "to_do": {
                "rich_text": rich_text,
                "checked": block.checked or False
            }
        }

    elif block_type == "callout":
        callout_data: dict = {"rich_text": rich_text}
        if block.color and block.color != "default":
            callout_data["color"] = block.color
        return {
            "type": "callout",
            "callout": callout_data
        }

    elif block_type == "divider":
        return {"type": "divider", "divider": {}}

    elif block_type == "code":
        # Code block content is in raw_lines
        code_content = "\n".join(block.raw_lines)
        return {
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": code_content}}],
                "language": block.language or "plain text"
            }
        }

    elif block_type == "table":
        # Table block — content is the column count, children are table_rows
        table_width = int(block.content) if block.content else 0
        row_children = []
        for child in block.children:
            if child.block_type == 'table_row':
                row_children.append(dnn_block_to_notion(child, registry))
            # Skip separator rows
        return {
            "type": "table",
            "table": {
                "table_width": table_width,
                "has_column_header": len(row_children) > 0,
                "has_row_header": False,
                "children": row_children,
            },
        }

    elif block_type == "table_row":
        # Parse cells from pipe-delimited content
        cells = parse_table_row_cells(block.content)
        notion_cells = []
        for cell_text in cells:
            cell_rich_text = rich_text_spans_to_notion(
                parse_inline_formatting(cell_text), registry
            )
            notion_cells.append(cell_rich_text)
        return {
            "type": "table_row",
            "table_row": {"cells": notion_cells}
        }

    elif block_type == "child_page":
        return {
            "type": "child_page",
            "child_page": {"title": block.content}
        }

    elif block_type == "child_database":
        return {
            "type": "child_database",
            "child_database": {"title": block.content}
        }

    else:
        # Fallback: treat as paragraph
        return {
            "type": "paragraph",
            "paragraph": {"rich_text": rich_text}
        }


def build_block_tree(blocks: list[DnnBlock], registry: IdRegistry) -> list[dict]:
    """Build nested Notion block structure from flat DnnBlock list.

    Handles indentation levels to create parent-child relationships.

    Args:
        blocks: Flat list of DnnBlocks with level attributes.
        registry: ID registry for resolving refs.

    Returns:
        List of Notion blocks with children nested.
    """
    if not blocks:
        return []

    # Build tree using a stack to track parent at each level
    result: list[dict] = []
    stack: list[tuple[int, dict]] = []  # (level, notion_block)

    for block in blocks:
        notion_block = dnn_block_to_notion(block, registry)
        level = block.level

        # Pop stack until we find parent level
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            # Add as child of top of stack
            parent = stack[-1][1]
            parent_type = parent["type"]
            if "children" not in parent[parent_type]:
                parent[parent_type]["children"] = []
            parent[parent_type]["children"].append(notion_block)
        else:
            # Top-level block
            result.append(notion_block)

        # Push to stack if this block type can have children
        if block.block_type in PARENT_BLOCK_TYPES:
            stack.append((level, notion_block))

    return result


async def append_blocks_async(
    parent_id: str,
    blocks: list[dict],
    after_id: Optional[str] = None
) -> list[dict]:
    """Append blocks to a parent using Notion API.

    Args:
        parent_id: UUID of parent block or page.
        blocks: List of Notion API block objects.
        after_id: Optional UUID to insert after.

    Returns:
        List of created block objects with IDs.
    """
    body: dict = {"children": blocks}
    if after_id:
        body["after"] = after_id

    result = await _notion_request_async(
        "PATCH",
        f"/blocks/{parent_id}/children",
        json_body=body
    )
    return result.get("results", [])


async def delete_block_async(block_id: str) -> dict:
    """Delete (archive) a block.

    Args:
        block_id: UUID of block to delete.

    Returns:
        Deleted block object.
    """
    return await _notion_request_async("DELETE", f"/blocks/{block_id}")


async def update_block_text_async(
    block_id: str,
    new_text: str,
    block_type: Optional[str] = None,
    registry: Optional[IdRegistry] = None
) -> tuple[dict, Optional[str]]:
    """Update block text, or replace block if type change is needed.

    The Notion API doesn't allow changing block types via PATCH. This function
    detects when the new text implies a different block type (e.g., "## heading"
    when updating an h3 block) and automatically replaces the block:
    1. Clone children
    2. Delete old block
    3. Create new block at same position with children

    Type changes that trigger replacement:
    - Heading level change (h1→h2, h2→h3, etc.)
    - Block type change (paragraph→bullet, bullet→todo, etc.)

    Changes that can be done via PATCH (no replacement):
    - Toggle state change (># heading → # heading)
    - Text/formatting changes within same type

    Args:
        block_id: UUID of block to update.
        new_text: New DNN-formatted text content (may include block markers).
        block_type: Block type (fetched if not provided).
        registry: Optional ID registry for resolving page mention short IDs.

    Returns:
        Tuple of (result_block, new_short_id_or_none):
        - If updated in place: (updated_block, None)
        - If replaced: (new_block, new_short_id)
    """
    # Always fetch block to get current state
    block = await _notion_request_async("GET", f"/blocks/{block_id}")
    current_type = block.get("type")
    current_data = block.get(current_type, {})

    # Parse new text to detect desired type
    detected_type, clean_text, attrs, warnings = parse_block_type(new_text)

    # Determine if we need to replace the block (type change)
    needs_replace = False
    update_toggle_state = None  # None = no change, True/False = new state

    if detected_type != 'paragraph':  # A marker was detected
        if detected_type != current_type:
            # Different block type → needs replacement
            needs_replace = True
        elif detected_type in ('heading_1', 'heading_2', 'heading_3'):
            # Same heading type - check if toggle state changed
            current_toggleable = current_data.get('is_toggleable', False)
            new_toggleable = attrs.get('is_toggle', False)
            if current_toggleable != new_toggleable:
                update_toggle_state = new_toggleable

    if needs_replace:
        # === REPLACE BLOCK (type change) ===
        # Get parent info for positioning
        parent = block.get("parent", {})
        parent_type = parent.get("type")
        if parent_type == "page_id":
            parent_id = parent.get("page_id")
        elif parent_type == "block_id":
            parent_id = parent.get("block_id")
        else:
            raise ValueError(f"Unknown parent type: {parent_type}")

        # Clone children before deletion
        children_to_copy = []
        if block.get("has_children"):
            for child in await _fetch_children_one_level(block_id):
                cloned, _ = await clone_block_with_children(child.get("id"))
                children_to_copy.append(cloned)

        # Find position - get the block BEFORE this one
        siblings = await _fetch_children_one_level(parent_id)
        after_id = None
        for i, sibling in enumerate(siblings):
            if sibling.get("id") == block_id:
                if i > 0:
                    after_id = siblings[i - 1].get("id")
                break

        # Delete the old block
        await delete_block_async(block_id)

        # Build new block
        new_dnn_block = DnnBlock(
            short_id="",
            level=0,
            block_type=detected_type,
            content=clean_text,
            **attrs
        )
        new_notion_block = dnn_block_to_notion(new_dnn_block, registry)

        # Add children if any
        if children_to_copy:
            new_notion_block[detected_type]["children"] = children_to_copy

        # Create at the right position
        created = await append_blocks_async(parent_id, [new_notion_block], after_id)

        if created:
            new_id = created[0].get("id", "")
            new_short = registry.register(new_id) if registry and new_id else None
            return created[0], new_short

        raise RuntimeError("Failed to create replacement block")

    # === UPDATE IN PLACE (no type change) ===
    # Strip marker since we're keeping the same type
    clean_text, strip_attrs, strip_warnings = strip_marker_for_block(new_text, current_type)
    if strip_warnings:
        logger.warning(f"update_block_text: {'; '.join(strip_warnings)}")

    # Parse the cleaned text to rich_text
    rich_text = rich_text_spans_to_notion(parse_inline_formatting(clean_text), registry)

    # Build update payload based on block type
    update_payload: dict = {}

    if current_type in ("paragraph", "bulleted_list_item", "numbered_list_item",
                        "toggle", "quote", "callout"):
        update_payload[current_type] = {"rich_text": rich_text}

    elif current_type in ("heading_1", "heading_2", "heading_3"):
        update_payload[current_type] = {"rich_text": rich_text}
        # Update toggle state if changed
        if update_toggle_state is not None:
            update_payload[current_type]["is_toggleable"] = update_toggle_state

    elif current_type == "to_do":
        update_payload["to_do"] = {"rich_text": rich_text}

    elif current_type == "code":
        update_payload["code"] = {"rich_text": rich_text}

    elif current_type == "table_row":
        # Parse pipe-delimited cells from new text
        cells = parse_table_row_cells(clean_text)
        notion_cells = []
        for cell_text in cells:
            cell_rich_text = rich_text_spans_to_notion(
                parse_inline_formatting(cell_text), registry
            )
            notion_cells.append(cell_rich_text)
        update_payload["table_row"] = {"cells": notion_cells}

    else:
        raise ValueError(f"Cannot update text of block type: {current_type}")

    result = await _notion_request_async(
        "PATCH",
        f"/blocks/{block_id}",
        json_body=update_payload
    )
    return result, None


async def toggle_todo_async(block_id: str, checked: bool) -> dict:
    """Toggle a to_do block's checkbox.

    Args:
        block_id: UUID of to_do block.
        checked: New checked state.

    Returns:
        Updated block object.
    """
    return await _notion_request_async(
        "PATCH",
        f"/blocks/{block_id}",
        json_body={"to_do": {"checked": checked}}
    )


@dataclass
class ApplyLineResult:
    """Result of a single apply command."""
    line: int
    ok: bool
    op: str
    created: list[str] = field(default_factory=list)  # new IDs from add
    moved: Optional[tuple[str, str]] = None  # (old_id, new_id) from move
    replaced: Optional[tuple[str, str]] = None  # (old_id, new_id) from update with type change
    error: Optional[str] = None
    pages_at_bottom: list[str] = field(default_factory=list)  # pages created at bottom (no positioning)
    warnings: list[str] = field(default_factory=list)  # non-fatal warnings (e.g., link_mention conversion)


@dataclass
class ApplyResult:
    """Result of applying a script."""
    ok: bool
    results: list[ApplyLineResult] = field(default_factory=list)
    id_map: dict[str, str] = field(default_factory=dict)  # aggregated moves


def _sanitize_mention_for_write(mention_obj: dict) -> Optional[dict]:
    """Sanitize a mention object for writing to Notion API.

    The Notion API returns mentions with extra fields (like full user objects),
    but when writing, it expects minimal structure. This function:
    1. Validates the mention has required type-specific fields
    2. Strips extra fields to minimal write format
    3. Returns None if the mention is invalid (caller should use plain_text)

    IMPORTANT - Notion API Mention Type Asymmetry:

    The Notion API can READ more mention types than it can WRITE. When writing
    rich_text, only these 6 mention types are accepted:

    WRITABLE (can be created via API):
    - user: @mention a workspace member
    - date: @mention a date or date range
    - page: Link to another page
    - database: Link to a database
    - template_mention: Dynamic template placeholders (today, me)
    - custom_emoji: Custom workspace emoji (undocumented but works)

    READ-ONLY (returned by API but cannot be written):
    - link_mention: Rich link embeds created by Notion UI when pasting URLs.
      Shows a preview card with title, description, favicon. This is an
      UNDOCUMENTED internal type - not in official API docs.
    - link_preview: Rich embeds from Link Preview integrations (Slack, Figma,
      GitHub, etc.). Requires partnership with Notion - cannot be created
      via standard API.

    When we encounter link_mention during block cloning (e.g., move operations),
    we convert it to a text+link which preserves the URL and title but loses
    the rich preview card. This is the best available fallback.

    Args:
        mention_obj: The mention object from rich_text.

    Returns:
        Sanitized mention object, or None if invalid.
    """
    mention = mention_obj.get("mention", {})
    mention_type = mention.get("type")

    if not mention_type:
        return None

    # Handle each mention type
    if mention_type == "user":
        user = mention.get("user")
        if not user or not isinstance(user, dict):
            return None
        user_id = user.get("id")
        if not user_id:
            return None
        return {
            "type": "mention",
            "mention": {
                "type": "user",
                "user": {"id": user_id}
            }
        }

    elif mention_type == "date":
        date = mention.get("date")
        if not date or not isinstance(date, dict):
            return None
        start = date.get("start")
        if not start:
            return None
        sanitized_date = {"start": start}
        if date.get("end"):
            sanitized_date["end"] = date["end"]
        if date.get("time_zone"):
            sanitized_date["time_zone"] = date["time_zone"]
        return {
            "type": "mention",
            "mention": {
                "type": "date",
                "date": sanitized_date
            }
        }

    elif mention_type == "page":
        page = mention.get("page")
        if not page or not isinstance(page, dict):
            return None
        page_id = page.get("id")
        if not page_id:
            return None
        return {
            "type": "mention",
            "mention": {
                "type": "page",
                "page": {"id": page_id}
            }
        }

    elif mention_type == "database":
        database = mention.get("database")
        if not database or not isinstance(database, dict):
            return None
        db_id = database.get("id")
        if not db_id:
            return None
        return {
            "type": "mention",
            "mention": {
                "type": "database",
                "database": {"id": db_id}
            }
        }

    elif mention_type == "template_mention":
        template = mention.get("template_mention")
        if not template or not isinstance(template, dict):
            return None
        template_type = template.get("type")
        if template_type == "template_mention_date":
            date_value = template.get("template_mention_date")
            if not date_value:
                return None
            return {
                "type": "mention",
                "mention": {
                    "type": "template_mention",
                    "template_mention": {
                        "type": "template_mention_date",
                        "template_mention_date": date_value
                    }
                }
            }
        elif template_type == "template_mention_user":
            user_value = template.get("template_mention_user")
            if not user_value:
                return None
            return {
                "type": "mention",
                "mention": {
                    "type": "template_mention",
                    "template_mention": {
                        "type": "template_mention_user",
                        "template_mention_user": user_value
                    }
                }
            }
        return None

    elif mention_type == "link_preview":
        link_preview = mention.get("link_preview")
        if not link_preview or not isinstance(link_preview, dict):
            return None
        url = link_preview.get("url")
        if not url:
            return None
        return {
            "type": "mention",
            "mention": {
                "type": "link_preview",
                "link_preview": {"url": url}
            }
        }

    elif mention_type == "link_mention":
        # link_mention is a rich link embed - Notion API doesn't accept it for writes.
        # Return None to trigger special handling in _sanitize_rich_text_for_write,
        # which will convert it to a text+link instead of just plain_text.
        return None

    # Unknown mention type (e.g., custom_emoji) - return None to fall back to plain_text
    logger.warning(f"Unknown mention type: {mention_type}")
    return None


def _sanitize_rich_text_for_write(
    rich_text: list[dict]
) -> tuple[list[dict], list[str]]:
    """Sanitize a rich_text array for writing to Notion API.

    Handles mentions that may have incomplete or malformed data by:
    - Validating mention structure
    - Converting invalid mentions to plain text
    - Preserving annotations where possible

    Args:
        rich_text: Array of rich_text objects from Notion API.

    Returns:
        Tuple of (sanitized_array, warnings) where warnings lists any
        conversions that caused fidelity loss (e.g., link_mention → text+link).
    """
    if not rich_text:
        return rich_text, []

    sanitized = []
    warnings: list[str] = []

    for item in rich_text:
        item_type = item.get("type")

        if item_type == "mention":
            # Try to sanitize the mention
            sanitized_mention = _sanitize_mention_for_write(item)

            if sanitized_mention:
                # Valid mention - add annotations if present
                if item.get("annotations"):
                    sanitized_mention["annotations"] = item["annotations"]
                sanitized.append(sanitized_mention)
            else:
                # Invalid mention - convert to text, preserving link if possible
                mention = item.get("mention", {})
                mention_type = mention.get("type")
                plain_text = item.get("plain_text", "")

                # For link_mention, extract URL and create text+link
                if mention_type == "link_mention":
                    link_mention = mention.get("link_mention", {})
                    href = link_mention.get("href")
                    # Use title if available, else plain_text, else URL
                    title = link_mention.get("title") or plain_text or href
                    if href:
                        text_obj: dict = {
                            "type": "text",
                            "text": {"content": title, "link": {"url": href}}
                        }
                        if item.get("annotations"):
                            text_obj["annotations"] = item["annotations"]
                        sanitized.append(text_obj)
                        # Record warning about fidelity loss
                        warnings.append(f"link preview → text link: {title[:30]}...")
                        continue

                # Default fallback: plain text without link
                if plain_text:
                    text_obj = {
                        "type": "text",
                        "text": {"content": plain_text}
                    }
                    if item.get("annotations"):
                        text_obj["annotations"] = item["annotations"]
                    sanitized.append(text_obj)
                # If no plain_text, skip this item entirely

        elif item_type == "text":
            # Text objects are generally safe, but ensure structure
            text_data = item.get("text", {})
            text_obj = {
                "type": "text",
                "text": {
                    "content": text_data.get("content", "")
                }
            }
            if text_data.get("link"):
                text_obj["text"]["link"] = text_data["link"]
            if item.get("annotations"):
                text_obj["annotations"] = item["annotations"]
            sanitized.append(text_obj)

        elif item_type == "equation":
            # Equation objects - preserve expression
            equation = item.get("equation", {})
            eq_obj: dict = {
                "type": "equation",
                "equation": {"expression": equation.get("expression", "")}
            }
            if item.get("annotations"):
                eq_obj["annotations"] = item["annotations"]
            sanitized.append(eq_obj)

        else:
            # Unknown type - try to convert to plain text
            plain_text = item.get("plain_text", "")
            if plain_text:
                sanitized.append({
                    "type": "text",
                    "text": {"content": plain_text}
                })

    return sanitized, warnings


def _sanitize_block_for_write(
    block_type: str,
    type_data: dict
) -> tuple[dict, list[str]]:
    """Sanitize a block's type_data for writing to Notion API.

    Finds and sanitizes rich_text arrays within the block data.

    Args:
        block_type: The block type (e.g., "paragraph", "toggle").
        type_data: The block's type-specific data.

    Returns:
        Tuple of (sanitized_type_data, warnings).
    """
    warnings: list[str] = []

    # Most block types have rich_text at the top level
    if "rich_text" in type_data:
        type_data["rich_text"], rt_warnings = _sanitize_rich_text_for_write(type_data["rich_text"])
        warnings.extend(rt_warnings)

    # Caption field (used by image, video, etc.)
    if "caption" in type_data:
        type_data["caption"], cap_warnings = _sanitize_rich_text_for_write(type_data["caption"])
        warnings.extend(cap_warnings)

    return type_data, warnings


async def _reupload_notion_file(file_obj: dict) -> tuple[dict, Optional[str]]:
    """Re-upload a Notion-hosted file via the File Upload API.

    Notion-hosted files use presigned S3 URLs that expire after 1 hour.
    This downloads the file and re-uploads it via single_part mode to get
    a permanent file_upload reference.

    Args:
        file_obj: File object from block type_data (e.g., image block's
                  nested file object). Has shape {"type": "file"|"external"|
                  "file_upload", ...}.

    Returns:
        Tuple of (new_file_obj, warning). new_file_obj uses file_upload
        type if re-upload succeeded, or original if not needed/failed.
        warning is set on failure or if URL was already permanent.
    """
    file_type = file_obj.get("type")

    # External URLs don't expire — no re-upload needed
    if file_type == "external":
        return file_obj, None

    # Already a file_upload reference — no re-upload needed
    if file_type == "file_upload":
        return file_obj, None

    # Only process Notion-hosted files (type: "file")
    if file_type != "file":
        return file_obj, None

    presigned_url = file_obj.get("file", {}).get("url", "")
    if not presigned_url:
        return file_obj, "file has no URL, skipping re-upload"

    # Extract filename from URL path, or use default
    from urllib.parse import urlparse
    try:
        path = urlparse(presigned_url).path
        filename = path.split("/")[-1] if "/" in path else "file"
        if not filename:
            filename = "file"
    except Exception:
        filename = "file"

    # Guess content_type from extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type_map = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml",
        "pdf": "application/pdf", "mp4": "video/mp4", "mov": "video/quicktime",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    try:
        # Step 1: Download file from presigned URL
        client = await _get_async_client()
        download_resp = await client.get(presigned_url)
        download_resp.raise_for_status()
        file_bytes = download_resp.content

        # Step 2: Create file upload (single_part mode)
        upload_response = await _notion_request_async(
            "POST",
            "/file_uploads",
            json_body={
                "mode": "single_part",
                "filename": filename,
                "content_type": content_type,
            }
        )

        upload_id = upload_response.get("id")
        upload_url = upload_response.get("upload_url")
        if not upload_id or not upload_url:
            return file_obj, "file upload API returned no ID/URL, using expiring URL"

        # Step 3: Upload binary via multipart form to the upload URL
        token = _get_token()
        upload_resp = await client.post(
            upload_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
            },
            files={"file": (filename, file_bytes, content_type)},
        )
        upload_resp.raise_for_status()

        # Step 4: Poll until uploaded (with backoff, timeout ~30s)
        max_polls = 10
        for poll_attempt in range(max_polls):
            await asyncio.sleep(min(1.0 * (1.5 ** poll_attempt), 5.0))
            status_response = await _notion_request_async(
                "GET",
                f"/file_uploads/{upload_id}"
            )
            status = status_response.get("status")
            if status == "uploaded":
                return {
                    "type": "file_upload",
                    "file_upload": {"id": upload_id}
                }, None
            if status in ("failed", "expired"):
                return file_obj, f"file re-upload {status}, using expiring URL"

        return file_obj, "file re-upload timed out, using expiring URL"

    except httpx.HTTPStatusError as e:
        body = ""
        if e.response is not None:
            try:
                body = e.response.text
            except Exception:
                pass
        logger.warning(f"File re-upload failed: {e} body={body}")
        return file_obj, f"file re-upload failed ({e.response.status_code if e.response else '?'}: {body or e}), using expiring URL"
    except Exception as e:
        logger.warning(f"File re-upload failed: {e}")
        return file_obj, f"file re-upload failed ({e}), using expiring URL"


# Block types that contain Notion-hosted file references
FILE_BEARING_BLOCK_TYPES = {"image", "file", "video", "pdf"}

# Block types that cannot be moved via clone+archive
UNMOVABLE_BLOCK_TYPES = {
    "synced_block": "Cannot move synced_block: sync relationship would be broken. "
                    "Move the original content manually if you want an independent copy.",
    "link_preview": "Cannot move link_preview: integration-specific embed cannot be "
                    "recreated via API.",
    "table_of_contents": "Cannot move table_of_contents: positional block with no "
                         "portable content.",
    "breadcrumb": "Cannot move breadcrumb: positional block with no portable content.",
    "unsupported": "Cannot move unsupported block type.",
}


def _check_movability(source_block: dict) -> tuple[list[str], Optional[str]]:
    """Pre-flight check for block movability.

    Returns:
        Tuple of (warnings, error). If error is set, the move should be
        refused. Warnings are non-fatal notes about fidelity.
    """
    source_type = source_block.get("type", "")
    warnings: list[str] = []

    # Category C: Unmovable blocks
    if source_type in UNMOVABLE_BLOCK_TYPES:
        return [], UNMOVABLE_BLOCK_TYPES[source_type]

    # Category B: File-bearing blocks (info about re-upload)
    if source_type in FILE_BEARING_BLOCK_TYPES:
        type_data = source_block.get(source_type, {})
        if type_data.get("type") == "file":
            warnings.append("File will be re-uploaded to preserve it")

    # Scan rich_text for link_mention items
    type_data = source_block.get(source_type, {})
    for rt_item in type_data.get("rich_text", []):
        if rt_item.get("type") == "mention":
            mention = rt_item.get("mention", {})
            if mention.get("type") == "link_mention":
                warnings.append(
                    "Block contains rich URL embeds that will be "
                    "downgraded to plain links"
                )
                break  # One warning is enough

    return warnings, None


async def clone_block_with_children(
    block_id: str,
    max_depth: int = 10
) -> tuple[dict, list[str]]:
    """Recursively fetch a block and its children for cloning.

    Args:
        block_id: UUID of block to clone.
        max_depth: Maximum recursion depth.

    Returns:
        Tuple of (notion_block, warnings) where notion_block is the API block
        object with children included, and warnings lists any fidelity issues
        (e.g., link_mention conversions).
    """
    warnings: list[str] = []

    # Fetch the block
    block = await _notion_request_async("GET", f"/blocks/{block_id}")
    block_type = block.get("type", "")

    # Build the clone structure
    type_data = block.get(block_type, {})

    # Remove read-only fields from type_data
    read_only_fields = ["id", "created_time", "last_edited_time", "created_by",
                        "last_edited_by", "parent", "archived", "in_trash"]
    for field in read_only_fields:
        type_data.pop(field, None)

    # Sanitize rich_text to handle malformed mentions
    type_data, block_warnings = _sanitize_block_for_write(block_type, type_data)
    warnings.extend(block_warnings)

    # Re-upload Notion-hosted files to get permanent references
    if block_type in FILE_BEARING_BLOCK_TYPES:
        # File-bearing blocks store file info directly in type_data
        # e.g., image block: type_data = {"type": "file", "file": {"url": "..."}}
        file_obj = {
            "type": type_data.get("type", ""),
            type_data.get("type", ""): type_data.get(type_data.get("type", ""), {})
        }
        if file_obj.get("type") == "file":
            new_file_obj, reupload_warning = await _reupload_notion_file(file_obj)
            if reupload_warning:
                warnings.append(reupload_warning)
            old_type = type_data.get("type", "")
            if new_file_obj.get("type") == "file_upload":
                # Replace the file reference with the permanent upload
                type_data.pop(old_type, None)
                type_data["type"] = "file_upload"
                type_data["file_upload"] = new_file_obj["file_upload"]
            else:
                # Fallback: convert to external URL (presigned, expires ~1h)
                # Notion write API only accepts external or file_upload, not file
                presigned_url = file_obj.get("file", {}).get("url", "")
                if presigned_url:
                    type_data.pop(old_type, None)
                    type_data["type"] = "external"
                    type_data["external"] = {"url": presigned_url}
                    warnings.append("using expiring URL as external link (~1h)")
                else:
                    warnings.append("file has no URL, clone may fail")

    clone = {"type": block_type, block_type: type_data}

    # If block has children, fetch and include them
    if block.get("has_children") and max_depth > 0:
        children_response = await _notion_request_async(
            "GET",
            f"/blocks/{block_id}/children?page_size=100"
        )
        children = children_response.get("results", [])

        # Recursively clone each child
        cloned_children = []
        for child in children:
            child_clone, child_warnings = await clone_block_with_children(
                child["id"],
                max_depth=max_depth - 1
            )
            cloned_children.append(child_clone)
            warnings.extend(child_warnings)

        # Add children to the clone
        if cloned_children:
            clone[block_type]["children"] = cloned_children

    return clone, warnings


async def _validate_after_is_sibling(
    parent_uuid: str,
    after_uuid: str,
    registry: IdRegistry,
    after_short_id: str
) -> tuple[bool, Optional[str]]:
    """Validate that `after` block is a direct child of `parent`.

    The Notion API requires the `after` parameter to reference a block
    that is a direct child of the parent. This function verifies that
    constraint before making the API call.

    Args:
        parent_uuid: UUID of the parent block/page.
        after_uuid: UUID of the block to insert after.
        registry: ID registry for generating short IDs in error messages.
        after_short_id: Short ID of the after block for error messages.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, returns (True, None).
        If invalid, returns (False, descriptive_error).
    """
    try:
        # Fetch the after block to check its parent
        after_block = await _notion_request_async("GET", f"/blocks/{after_uuid}")
        after_parent = after_block.get("parent", {})

        # Check parent type and ID
        parent_type = after_parent.get("type", "")
        if parent_type == "page_id":
            actual_parent = after_parent.get("page_id", "")
        elif parent_type == "block_id":
            actual_parent = after_parent.get("block_id", "")
        else:
            actual_parent = ""

        # Normalize for comparison
        try:
            actual_parent_normalized = normalize_uuid(actual_parent) if actual_parent else ""
            parent_normalized = normalize_uuid(parent_uuid)
        except ValueError:
            return False, f"Invalid UUID format in parent reference"

        if actual_parent_normalized != parent_normalized:
            # Register actual parent to get a short ID for the error message
            actual_parent_short = registry.register(actual_parent_normalized)
            return False, (
                f"'after' block {after_short_id} is not a child of target parent. "
                f"Its actual parent is {actual_parent_short}. "
                f"Use after={actual_parent_short} to insert after that block, "
                f"or pick a different 'after' that's a direct child of your target parent."
            )

        return True, None

    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 404:
            return False, f"'after' block not found: {after_short_id}"
        return False, f"Failed to validate 'after' block: HTTP {e.response.status_code if e.response else 'error'}"
    except Exception as e:
        return False, f"Failed to validate 'after' block: {e}"


def _remap_id(short_id: Optional[str], id_map: dict[str, str]) -> Optional[str]:
    """Remap a short ID if it was moved in a previous operation."""
    if short_id is None:
        return None
    return id_map.get(short_id, short_id)


def _build_property_value(
    prop_type: str,
    value: str,
    registry: Optional['IdRegistry'] = None
) -> Optional[dict]:
    """Convert a DNN value to Notion property format based on property type.

    Args:
        prop_type: Notion property type (title, rich_text, select, etc.)
        value: String value from DNN row
        registry: Optional ID registry for resolving @mentions in text fields

    Returns:
        Notion API property value dict, or None if conversion fails
    """
    if prop_type == "title":
        return {"title": _text_to_rich_text(value, registry)}
    elif prop_type == "rich_text":
        return {"rich_text": _text_to_rich_text(value, registry)}
    elif prop_type == "select":
        return {"select": {"name": value}}
    elif prop_type == "multi_select":
        # Comma-separated values
        names = [n.strip() for n in value.split(",")]
        return {"multi_select": [{"name": n} for n in names]}
    elif prop_type == "checkbox":
        return {"checkbox": value.lower() in ("true", "1", "yes", "x")}
    elif prop_type == "number":
        try:
            return {"number": float(value)}
        except ValueError:
            return None
    elif prop_type == "date":
        # Support date ranges: "2025-01-01→2025-01-15" or single "2025-01-01"
        if "→" in value:
            start, end = value.split("→", 1)
            return {"date": {"start": start.strip(), "end": end.strip()}}
        else:
            return {"date": {"start": value}}
    elif prop_type == "url":
        return {"url": value}
    elif prop_type == "email":
        return {"email": value}
    elif prop_type == "phone_number":
        return {"phone_number": value}
    return None


def _build_row_properties(
    row_values: dict[str, str],
    db_properties: dict,
    registry: Optional['IdRegistry'] = None
) -> dict[str, dict]:
    """Build Notion page properties from DNN row values.

    Handles case-insensitive property name matching, and tolerates
    trailing/leading whitespace in Notion property names.

    Args:
        row_values: Dict of column_name -> value from DNN
        db_properties: Database schema properties dict
        registry: Optional ID registry for resolving @mentions in text fields

    Returns:
        Dict of property_name -> Notion property value
    """
    page_props: dict = {}
    for col_name, value in row_values.items():
        # Find the property definition
        prop_def = db_properties.get(col_name)
        if not prop_def:
            # Try normalized match: case-insensitive + strip whitespace.
            # Notion property names may have trailing spaces (e.g., "Paid on ")
            # but DNN input is stripped during parsing, so we normalize here.
            for pname, pdef in db_properties.items():
                if pname.strip().lower() == col_name.lower():
                    prop_def = pdef
                    col_name = pname  # Use original Notion name for API call
                    break

        if not prop_def:
            continue  # Skip unknown properties

        prop_type = prop_def.get("type")
        prop_value = _build_property_value(prop_type, value, registry)
        if prop_value is not None:
            page_props[col_name] = prop_value

    return page_props


async def _fetch_database_schema(
    database_id: str
) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """Fetch database schema properties via data_source.

    In Notion API 2025-09-03, the database schema is in the data_source,
    not the database container.

    Args:
        database_id: The database container UUID

    Returns:
        Tuple of (data_source_id, properties_dict, error_message).
        On success: (data_source_id, properties, None)
        On failure: (None, None, error_message)
    """
    try:
        database = await _notion_request_async("GET", f"/databases/{database_id}")
        data_sources = database.get("data_sources", [])
        if not data_sources:
            return None, None, "Database has no data sources"
        data_source_id = data_sources[0].get("id")
        if not data_source_id:
            return None, None, "Data source missing ID"
        db_info = await _notion_request_async("GET", f"/data_sources/{data_source_id}")
        return data_source_id, db_info.get("properties", {}), None
    except httpx.HTTPStatusError as e:
        # Use _http_error_detail helper for consistent error extraction
        return None, None, f"Could not fetch database schema: {_http_error_detail(e, 200)}"


async def execute_apply_op(
    op: ApplyOp,
    registry: IdRegistry,
    id_map: Optional[dict[str, str]] = None
) -> tuple[dict, Optional[str]]:
    """Execute a single apply operation.

    Args:
        op: Parsed apply operation.
        registry: ID registry for resolving refs.
        id_map: Mapping of old→new IDs from previous moves in this apply call.

    Returns:
        Tuple of (result_dict, error_message or None).
    """
    if id_map is None:
        id_map = {}

    try:
        if op.command == ApplyCommand.ADD:
            # Resolve parent ID (with remapping for moved blocks)
            # Uses _resolve_or_error helper to consolidate repeated resolution pattern
            parent_id = _remap_id(op.parent, id_map)
            parent_uuid, err = _resolve_or_error(registry, parent_id, "parent")
            if err:
                return {}, err

            # Resolve after ID if provided (with remapping)
            after_uuid = None
            after_id = _remap_id(op.after, id_map)
            if after_id:
                after_uuid, err = _resolve_or_error(registry, after_id, "after")
                if err:
                    return {}, err
                # Validate after is a sibling of parent
                is_valid, err_msg = await _validate_after_is_sibling(
                    parent_uuid, after_uuid, registry, op.after
                )
                if not is_valid:
                    return {}, err_msg

            # Check for child_page/child_database blocks - these need special handling
            # (Notion API doesn't support creating these via Append Block Children)
            child_pages = [b for b in op.content_blocks if b.block_type == "child_page"]
            child_dbs = [b for b in op.content_blocks if b.block_type == "child_database"]
            regular_blocks = [b for b in op.content_blocks
                             if b.block_type not in ("child_page", "child_database")]

            # Error if trying to position child pages/databases with 'after'
            if (child_pages or child_dbs) and op.after:
                items = [f"§ {b.content}" for b in child_pages]
                items += [f"⊞ {b.content}" for b in child_dbs]
                return {}, (
                    f"Cannot position child pages/databases with 'after'. "
                    f"Items [{', '.join(items)}] would be created at bottom of parent. "
                    f"Use '+page parent=X title=\"...\"' without 'after', or position manually in Notion UI."
                )

            created_ids = []
            pages_at_bottom = []

            # Create child pages via Create Page API (will appear at bottom)
            for block in child_pages:
                page_body: dict = {
                    "parent": {"page_id": parent_uuid},
                    "properties": {"title": {"title": _text_to_rich_text(block.content, registry)}}
                }
                try:
                    new_page = await _notion_request_async("POST", "/pages", json_body=page_body)
                    page_id = new_page.get("id", "")
                    if page_id:
                        short_id = registry.register(page_id)
                        created_ids.append(short_id)
                        pages_at_bottom.append(block.content)
                except httpx.HTTPStatusError as e:
                    return {}, f"Failed to create child page '{block.content}': {_http_error_detail(e, 200)}"

            # Child databases cannot be created via API - return error
            if child_dbs:
                db_names = [b.content for b in child_dbs]
                return {}, (
                    f"Cannot create child databases via API. "
                    f"Databases [{', '.join(db_names)}] must be created manually in Notion UI."
                )

            # Process regular blocks normally
            if regular_blocks:
                # If parent is a table, unwrap auto-grouped table blocks
                # back to raw table_rows (auto-grouping is wrong when
                # explicitly adding rows to an existing table).
                try:
                    parent_block = await _notion_request_async(
                        "GET", f"/blocks/{parent_uuid}"
                    )
                    if parent_block.get("type") == "table":
                        unwrapped = []
                        for b in regular_blocks:
                            if b.block_type == "table" and not b.short_id:
                                # Synthetic table from auto-grouping — unwrap
                                unwrapped.extend(b.children)
                            else:
                                unwrapped.append(b)
                        regular_blocks = unwrapped
                except Exception:
                    pass  # If fetch fails, proceed with original blocks

                notion_blocks = build_block_tree(regular_blocks, registry)
                num_created = len(notion_blocks)

                created = await append_blocks_async(parent_uuid, notion_blocks, after_uuid)

                for block in created[:num_created]:
                    block_id = block.get("id", "")
                    if block_id:
                        short_id = registry.register(block_id)
                        created_ids.append(short_id)

            # Collect warnings from all content blocks
            all_warnings = []
            for block in op.content_blocks:
                all_warnings.extend(block.warnings)

            result: dict = {"op": "+", "parent": op.parent, "created": created_ids}
            if pages_at_bottom:
                result["pages_at_bottom"] = pages_at_bottom
            if all_warnings:
                result["warnings"] = all_warnings
            return result, None

        elif op.command == ApplyCommand.DELETE:
            deleted = []
            for target in op.targets:
                target_id = _remap_id(target, id_map)
                target_uuid, err = _resolve_or_error(registry, target_id, "target")
                if err:
                    return {}, err
                await delete_block_async(target_uuid)
                deleted.append(target)

            return {"op": "x", "deleted": deleted}, None

        elif op.command == ApplyCommand.UPDATE:
            target_id = _remap_id(op.target, id_map)
            target_uuid, err = _resolve_or_error(registry, target_id, "target")
            if err:
                return {}, err

            # Check if it's a page-backed block (child_page/child_database)
            # — auto-route to page title update, same pattern as m→mpage
            block = await _notion_request_async("GET", f"/blocks/{target_uuid}")
            block_type = block.get("type", "")
            if block_type in ("child_page", "child_database"):
                page_id = block.get(block_type, {}).get("id", target_uuid)
                title = op.new_text or ""
                try:
                    await _notion_request_async(
                        "PATCH", f"/pages/{page_id}",
                        json_body={"properties": {"title": {"title": _text_to_rich_text(title, registry)}}}
                    )
                except httpx.HTTPStatusError as e:
                    return {}, f"Page title update failed: {_http_error_detail(e)}"
                return {"op": "u", "updated": op.target}, None

            _, new_short_id = await update_block_text_async(
                target_uuid, op.new_text or "", registry=registry
            )
            if new_short_id:
                # Block was replaced (type change) - ID changed
                return {"op": "u", "updated": op.target, "replaced": new_short_id}, None
            return {"op": "u", "updated": op.target}, None

        elif op.command == ApplyCommand.TOGGLE:
            target_id = _remap_id(op.target, id_map)
            target_uuid, err = _resolve_or_error(registry, target_id, "target")
            if err:
                return {}, err

            await toggle_todo_async(target_uuid, op.checked or False)
            return {"op": "t", "toggled": op.target}, None

        elif op.command == ApplyCommand.MOVE:
            # Move is clone+archive - IDs change
            source_id = _remap_id(op.source, id_map)
            source_uuid, err = _resolve_or_error(registry, source_id, "source")
            if err:
                return {}, err

            dest_id = _remap_id(op.dest_parent, id_map)
            dest_uuid, err = _resolve_or_error(registry, dest_id, "destination")
            if err:
                return {}, err

            after_uuid = None
            after_id = _remap_id(op.dest_after, id_map)
            if after_id:
                after_uuid, err = _resolve_or_error(registry, after_id, "after")
                if err:
                    return {}, err
                # Validate after is a sibling of destination parent
                is_valid, err_msg = await _validate_after_is_sibling(
                    dest_uuid, after_uuid, registry, op.dest_after
                )
                if not is_valid:
                    return {}, err_msg

            # Fetch source block to check type and movability
            source_block = await _notion_request_async("GET", f"/blocks/{source_uuid}")
            source_type = source_block.get("type", "")

            # Pre-flight movability check
            move_warnings, move_error = _check_movability(source_block)
            if move_error:
                return {}, move_error

            # Check if it's a page-backed block (use page move instead)
            if source_type in ("child_page", "child_database"):
                # Page move - ID doesn't change
                # Get the page ID from the block
                page_id = source_block.get(source_type, {}).get("id", source_uuid)
                await _notion_request_async(
                    "POST",
                    f"/pages/{page_id}/move",
                    json_body={"parent": {"page_id": dest_uuid}}
                )
                return {"op": "m", "moved": {op.source: op.source}}, None

            # Clone the block with all children recursively
            clone_block, clone_warnings = await clone_block_with_children(source_uuid)

            # Append to destination
            created = await append_blocks_async(dest_uuid, [clone_block], after_uuid)

            # Archive the original
            await delete_block_async(source_uuid)

            # Build ID mapping for result
            move_result = {}
            if created:
                new_id = created[0].get("id", "")
                if new_id:
                    new_short = registry.register(new_id)
                    move_result[op.source] = new_short

            result: dict = {"op": "m", "moved": move_result}
            all_warnings = move_warnings + clone_warnings
            if all_warnings:
                result["warnings"] = all_warnings
            return result, None

        elif op.command == ApplyCommand.ADD_ROW:
            # Add a row to a database
            db_id = _remap_id(op.database, id_map)
            db_uuid, err = _resolve_or_error(registry, db_id, "database")
            if err:
                return {}, err

            # In API 2025-09-03, we need both:
            # - database_id (container) for page creation (parent.database_id)
            # - data_source_id for schema lookup (/data_sources/{id})
            # The user might pass either @db (container) or @ds (data_source) ID
            # We always need the database container ID for parent.database_id
            database_id = db_uuid  # For parent.database_id
            data_source_id = db_uuid  # For schema lookup
            try:
                # First, try to fetch as database container
                database = await _notion_request_async("GET", f"/databases/{db_uuid}")
                data_sources = database.get("data_sources", [])
                if data_sources:
                    data_source_id = data_sources[0].get("id", db_uuid)
                    db_info = await _notion_request_async("GET", f"/data_sources/{data_source_id}")
                else:
                    return {}, f"Database has no data sources (db_uuid={db_uuid})"
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 404:
                    # Maybe it's a data_source_id - try fetching schema directly
                    # But we still need the database_id for parent - get it from data_source
                    try:
                        db_info = await _notion_request_async("GET", f"/data_sources/{db_uuid}")
                        # data_source has parent.database_id
                        parent = db_info.get("parent", {})
                        database_id = parent.get("database_id", db_uuid)
                        data_source_id = db_uuid
                    except httpx.HTTPStatusError as e2:
                        return {}, f"Could not find database or data source: {op.database} (err={_http_error_detail(e2, 200)})"
                else:
                    return {}, f"Database lookup failed: {_http_error_detail(e, 200)}"
            db_properties = db_info.get("properties", {})

            # Build page properties
            page_props = _build_row_properties(op.row_values, db_properties, registry)

            # Build page body
            # IMPORTANT: parent.database_id must be the database CONTAINER ID,
            # NOT the data_source_id. The field name is confusing but the API
            # expects the container UUID here. Schema comes from data_source.
            page_body: dict = {
                "parent": {"database_id": database_id},
                "properties": page_props
            }

            # Add icon if provided (emoji or external URL)
            if op.icon:
                if op.icon.startswith("http"):
                    page_body["icon"] = {"type": "external", "external": {"url": op.icon}}
                else:
                    page_body["icon"] = {"type": "emoji", "emoji": op.icon}

            # Create the page
            try:
                result = await _notion_request_async(
                    "POST",
                    "/pages",
                    json_body=page_body
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Page creation failed (database_id={database_id}): {_http_error_detail(e)}"

            # Register the new row
            row_id = result.get("id", "")
            short_id = registry.register(row_id) if row_id else ""

            return {"op": "+row", "database": op.database, "created": short_id}, None

        elif op.command == ApplyCommand.UPDATE_ROW:
            # Update a database row's properties
            target_id = _remap_id(op.target, id_map)
            row_uuid, err = _resolve_or_error(registry, target_id, "row")
            if err:
                return {}, err

            if not op.row_values:
                return {}, f"No properties to update for row {op.target}"

            # Fetch the page to get its parent database
            page = await _notion_request_async("GET", f"/pages/{row_uuid}")
            parent = page.get("parent", {})
            database_id = parent.get("database_id")
            if not database_id:
                return {}, f"Row {op.target} is not in a database"

            # Fetch database schema (need to go through database → data_source)
            _, db_properties, schema_err = await _fetch_database_schema(database_id)
            if schema_err:
                return {}, schema_err

            # Build property updates
            page_props = _build_row_properties(op.row_values, db_properties, registry)

            if not page_props:
                return {}, f"No valid properties to update"

            # Update the page
            try:
                await _notion_request_async(
                    "PATCH",
                    f"/pages/{row_uuid}",
                    json_body={"properties": page_props}
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Row update failed: {_http_error_detail(e)}"

            return {"op": "urow", "updated": op.target}, None

        elif op.command == ApplyCommand.DELETE_ROW:
            # Archive a database row (rows are pages)
            target_id = _remap_id(op.target, id_map)
            row_uuid, err = _resolve_or_error(registry, target_id, "row")
            if err:
                return {}, err

            try:
                await _notion_request_async(
                    "PATCH",
                    f"/pages/{row_uuid}",
                    json_body={"archived": True}
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Row deletion failed: {_http_error_detail(e)}"

            return {"op": "xrow", "deleted": op.target}, None

        elif op.command == ApplyCommand.ADD_PAGE:
            # Create a new page
            parent_id = _remap_id(op.parent, id_map)
            parent_uuid, err = _resolve_or_error(registry, parent_id, "parent")
            if err:
                return {}, err

            # Validate parent is a page or database, not a regular block
            # Try to fetch as a block first to check its type
            try:
                block_info = await _notion_request_async("GET", f"/blocks/{parent_uuid}")
                block_type = block_info.get("type", "")
                # Only child_page and child_database blocks can be page parents
                if block_type not in ("child_page", "child_database"):
                    return {}, (
                        f"+page parent={op.parent} failed: parent is a '{block_type}' block. "
                        f"Pages can only be created under pages or databases, not inside blocks."
                    )
            except httpx.HTTPStatusError:
                # If block fetch fails, it might be a top-level page - let page creation try
                pass

            # Build page properties (title)
            page_props: dict = {}
            if op.title:
                page_props["title"] = {"title": _text_to_rich_text(op.title, registry)}

            # Build page body
            page_body: dict = {
                "parent": {"page_id": parent_uuid},
                "properties": page_props
            }

            # Add icon if provided (emoji or external URL)
            if op.icon:
                if op.icon.startswith("http"):
                    page_body["icon"] = {"type": "external", "external": {"url": op.icon}}
                else:
                    page_body["icon"] = {"type": "emoji", "emoji": op.icon}

            # Add cover if provided
            if op.cover:
                page_body["cover"] = {"type": "external", "external": {"url": op.cover}}

            # Add position if after= or pos= specified
            if op.after:
                after_uuid_val = _remap_id(op.after, id_map)
                after_resolved, after_err = _resolve_or_error(registry, after_uuid_val, "after block")
                if after_err:
                    return {}, after_err
                page_body["position"] = {
                    "type": "after_block",
                    "after_block": {"id": after_resolved}
                }
            elif op.position == "start":
                page_body["position"] = {"type": "page_start"}

            try:
                new_page = await _notion_request_async("POST", "/pages", json_body=page_body)
            except httpx.HTTPStatusError as e:
                return {}, f"Page creation failed: {_http_error_detail(e)}"

            page_id = new_page.get("id", "")
            short_id = registry.register(page_id) if page_id else ""

            # Add content blocks if provided
            created_ids = [short_id]
            if op.content_blocks and page_id:
                notion_blocks = build_block_tree(op.content_blocks, registry)
                created = await append_blocks_async(page_id, notion_blocks)
                for block in created:
                    block_id = block.get("id", "")
                    if block_id:
                        created_ids.append(registry.register(block_id))

            return {"op": "+page", "created": created_ids}, None

        elif op.command == ApplyCommand.UPDATE_PAGE:
            # Update page properties (title, icon, cover)
            target_id = _remap_id(op.target, id_map)
            page_uuid, err = _resolve_or_error(registry, target_id, "page")
            if err:
                return {}, err

            update_body: dict = {}

            # Update title if provided
            if op.title is not None:
                update_body["properties"] = {
                    "title": {"title": _text_to_rich_text(op.title, registry)}
                }

            # Update icon if provided
            if op.icon:
                if op.icon == "none":
                    update_body["icon"] = None
                elif op.icon.startswith("http"):
                    update_body["icon"] = {"type": "external", "external": {"url": op.icon}}
                else:
                    update_body["icon"] = {"type": "emoji", "emoji": op.icon}

            # Update cover if provided
            if op.cover:
                if op.cover == "none":
                    update_body["cover"] = None
                else:
                    update_body["cover"] = {"type": "external", "external": {"url": op.cover}}

            if not update_body:
                return {}, "upage: nothing to update (provide title, icon, or cover)"

            try:
                await _notion_request_async("PATCH", f"/pages/{page_uuid}", json_body=update_body)
            except httpx.HTTPStatusError as e:
                return {}, f"Page update failed: {_http_error_detail(e)}"

            return {"op": "upage", "updated": op.target}, None

        elif op.command == ApplyCommand.MOVE_PAGE:
            # Move page to new parent
            source_id = _remap_id(op.source, id_map)
            page_uuid, err = _resolve_or_error(registry, source_id, "page")
            if err:
                return {}, err

            dest_id = _remap_id(op.dest_parent, id_map)
            dest_uuid, err = _resolve_or_error(registry, dest_id, "destination")
            if err:
                return {}, err

            try:
                await _notion_request_async(
                    "POST",
                    f"/pages/{page_uuid}/move",
                    json_body={"parent": {"page_id": dest_uuid}}
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Page move failed: {_http_error_detail(e)}"

            return {"op": "mpage", "moved": op.source}, None

        elif op.command == ApplyCommand.DELETE_PAGE:
            # Archive a page
            target_id = _remap_id(op.target, id_map)
            page_uuid, err = _resolve_or_error(registry, target_id, "page")
            if err:
                return {}, err

            try:
                await _notion_request_async(
                    "PATCH",
                    f"/pages/{page_uuid}",
                    json_body={"archived": True}
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Page deletion failed: {_http_error_detail(e)}"

            return {"op": "xpage", "deleted": op.target}, None

        elif op.command == ApplyCommand.COPY_PAGE:
            # Copy a page with its content to a new parent
            source_id = _remap_id(op.source, id_map)
            source_uuid, err = _resolve_or_error(registry, source_id, "source")
            if err:
                return {}, err

            dest_id = _remap_id(op.dest_parent, id_map)
            dest_uuid, err = _resolve_or_error(registry, dest_id, "destination")
            if err:
                return {}, err

            # Fetch source page to get properties
            try:
                source_page = await _notion_request_async("GET", f"/pages/{source_uuid}")
            except httpx.HTTPStatusError as e:
                return {}, f"Failed to read source page: {_http_error_detail(e)}"

            # Get title from source or use provided title
            source_props = source_page.get("properties", {})
            title_prop = source_props.get("title", {})
            if op.title:
                new_title = op.title
            else:
                # Extract title from source
                title_arr = title_prop.get("title", [])
                new_title = "".join(t.get("plain_text", "") for t in title_arr) if title_arr else "Copy"

            # Build new page
            page_body: dict = {
                "parent": {"page_id": dest_uuid},
                "properties": {"title": {"title": _text_to_rich_text(new_title, registry)}}
            }

            # Copy icon if present
            source_icon = source_page.get("icon")
            if source_icon:
                page_body["icon"] = source_icon

            # Copy cover if present
            source_cover = source_page.get("cover")
            if source_cover:
                page_body["cover"] = source_cover

            try:
                new_page = await _notion_request_async("POST", "/pages", json_body=page_body)
            except httpx.HTTPStatusError as e:
                return {}, f"Page copy failed: {_http_error_detail(e)}"

            new_page_id = new_page.get("id", "")
            new_short_id = registry.register(new_page_id) if new_page_id else ""

            # Copy child blocks from source
            copy_warnings: list[str] = []
            if new_page_id:
                try:
                    children = await _notion_request_async(
                        "GET",
                        f"/blocks/{source_uuid}/children?page_size=100"
                    )
                    results = children.get("results", [])
                    if results:
                        # Clone each block
                        cloned_blocks = []
                        for block in results:
                            cloned, block_warnings = await clone_block_with_children(block.get("id", ""))
                            cloned_blocks.append(cloned)
                            copy_warnings.extend(block_warnings)
                        if cloned_blocks:
                            await append_blocks_async(new_page_id, cloned_blocks)
                except httpx.HTTPStatusError:
                    # Failed to copy children - page still created
                    pass

            result: dict = {"op": "cpage", "copied": new_short_id, "from": op.source}
            if copy_warnings:
                result["warnings"] = copy_warnings
            return result, None

        elif op.command == ApplyCommand.ADD_DB:
            # Create a new database
            parent_id = _remap_id(op.parent, id_map)
            parent_uuid, err = _resolve_or_error(registry, parent_id, "parent")
            if err:
                return {}, err

            if not op.property_defs:
                return {}, "+db: no property definitions provided"

            # Build properties schema
            properties: dict = {}
            for prop_def in op.property_defs:
                notion_schema = property_def_to_notion(prop_def)
                # For relation properties, resolve target DB to data_source_id
                if prop_def.api_type == "relation" and prop_def.relation_db:
                    rel_uuid, rel_err = _resolve_or_error(
                        registry, prop_def.relation_db, "relation target database"
                    )
                    if rel_err:
                        return {}, f"+db property '{prop_def.name}': {rel_err}"
                    target_ds_id, _, target_err = await _fetch_database_schema(rel_uuid)
                    if target_err:
                        return {}, f"+db property '{prop_def.name}' relation target: {target_err}"
                    notion_schema["relation"].pop("database_id", None)
                    notion_schema["relation"]["data_source_id"] = target_ds_id
                properties[prop_def.name] = notion_schema

            # Build database body
            db_body: dict = {
                "parent": {"type": "page_id", "page_id": parent_uuid},
                "title": [{"type": "text", "text": {"content": op.title or "Untitled"}}],
                "is_inline": False,
                "initial_data_source": {
                    "type": "external",
                    "external": {},
                    "properties": properties,
                },
            }

            # Add icon if provided
            if op.icon:
                if op.icon.startswith("http"):
                    db_body["icon"] = {"type": "external", "external": {"url": op.icon}}
                else:
                    db_body["icon"] = {"type": "emoji", "emoji": op.icon}

            try:
                result = await _notion_request_async(
                    "POST", "/databases", json_body=db_body
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Database creation failed: {_http_error_detail(e)}"

            db_id = result.get("id", "")
            short_id = registry.register(db_id) if db_id else ""

            return {"op": "+db", "created": short_id}, None

        elif op.command == ApplyCommand.ADD_PROP:
            # Add a property to an existing database
            db_id = _remap_id(op.database, id_map)
            db_uuid, err = _resolve_or_error(registry, db_id, "database")
            if err:
                return {}, err

            if not op.property_def:
                return {}, "+prop: no property definition provided"

            prop_def = op.property_def
            notion_schema = property_def_to_notion(prop_def)

            # Fetch data source ID for the database being modified
            data_source_id, _, schema_err = await _fetch_database_schema(db_uuid)
            if schema_err:
                return {}, f"+prop: {schema_err}"

            # For relation, resolve target DB to its data source ID
            # PATCH /data_sources requires data_source_id, not database_id
            if prop_def.api_type == "relation" and prop_def.relation_db:
                rel_uuid, rel_err = _resolve_or_error(
                    registry, prop_def.relation_db, "relation target database"
                )
                if rel_err:
                    return {}, f"+prop: {rel_err}"
                target_ds_id, _, target_err = await _fetch_database_schema(rel_uuid)
                if target_err:
                    return {}, f"+prop relation target: {target_err}"
                # Replace database_id placeholder with actual data_source_id
                notion_schema["relation"].pop("database_id", None)
                notion_schema["relation"]["data_source_id"] = target_ds_id

            # PATCH data source to add property
            patch_body = {
                "properties": {
                    prop_def.name: notion_schema
                }
            }
            try:
                await _notion_request_async(
                    "PATCH", f"/data_sources/{data_source_id}",
                    json_body=patch_body
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Add property failed: {_http_error_detail(e)}"

            return {"op": "+prop", "database": op.database, "added": prop_def.name}, None

        elif op.command == ApplyCommand.DELETE_PROP:
            # Delete a property from a database
            db_id = _remap_id(op.database, id_map)
            db_uuid, err = _resolve_or_error(registry, db_id, "database")
            if err:
                return {}, err

            prop_name = op.target
            if not prop_name:
                return {}, "xprop: no property name provided"

            # Fetch data source and verify property exists
            data_source_id, db_properties, schema_err = await _fetch_database_schema(db_uuid)
            if schema_err:
                return {}, f"xprop: {schema_err}"

            # Find the property (case-insensitive match)
            actual_name = None
            for pname, pdef in db_properties.items():
                if pname.strip().lower() == prop_name.lower():
                    # Cannot delete title property
                    if pdef.get("type") == "title":
                        return {}, f"xprop: cannot delete title property '{pname}'"
                    actual_name = pname
                    break

            if not actual_name:
                return {}, f"xprop: property '{prop_name}' not found in database"

            # PATCH data source to delete property (set to None)
            patch_body = {
                "properties": {
                    actual_name: None
                }
            }
            try:
                await _notion_request_async(
                    "PATCH", f"/data_sources/{data_source_id}",
                    json_body=patch_body
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Delete property failed: {_http_error_detail(e)}"

            return {"op": "xprop", "database": op.database, "deleted": actual_name}, None

        elif op.command == ApplyCommand.RENAME_PROP:
            # Rename a property in a database
            db_id = _remap_id(op.database, id_map)
            db_uuid, err = _resolve_or_error(registry, db_id, "database")
            if err:
                return {}, err

            if not op.old_name or not op.new_name:
                return {}, "uprop: old and new property names required"

            # Fetch data source and verify property exists
            data_source_id, db_properties, schema_err = await _fetch_database_schema(db_uuid)
            if schema_err:
                return {}, f"uprop: {schema_err}"

            # Find the property (case-insensitive match)
            actual_name = None
            for pname in db_properties:
                if pname.strip().lower() == op.old_name.lower():
                    actual_name = pname
                    break

            if not actual_name:
                return {}, f"uprop: property '{op.old_name}' not found in database"

            # PATCH data source to rename property
            patch_body = {
                "properties": {
                    actual_name: {"name": op.new_name}
                }
            }
            try:
                await _notion_request_async(
                    "PATCH", f"/data_sources/{data_source_id}",
                    json_body=patch_body
                )
            except httpx.HTTPStatusError as e:
                return {}, f"Rename property failed: {_http_error_detail(e)}"

            return {"op": "uprop", "database": op.database,
                    "renamed": {op.old_name: op.new_name}}, None

        else:
            return {}, f"Command not implemented: {op.command.value}"

    except httpx.HTTPStatusError as e:
        if e.response is not None:
            if e.response.status_code == 404:
                return {}, f"Object not found (may be deleted)"
            elif e.response.status_code == 403:
                return {}, f"Permission denied"
            elif e.response.status_code == 429:
                return {}, f"Rate limited"
            elif e.response.status_code == 400:
                # Parse Notion validation errors for better messages
                try:
                    err_json = e.response.json()
                    err_msg = err_json.get("message", _http_error_detail(e))
                    err_code = err_json.get("code", "validation_error")
                    return {}, f"{err_code}: {err_msg}"
                except Exception:
                    return {}, f"HTTP 400: {_http_error_detail(e)}"
            else:
                return {}, f"HTTP {e.response.status_code}: {_http_error_detail(e)}"
        return {}, str(e)
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


# =============================================================================
# Conflict Detection for Parallel Execution
# =============================================================================


def _get_op_targets(op: ApplyOp) -> list[str]:
    """Extract block IDs that this operation directly targets.

    "Target" means the operation reads or modifies this specific block.
    This is distinct from "parent" or "after" which are positional refs.

    Target IDs (cause conflicts if duplicated):
    - op.target: For u, t, upage, xpage, urow, xrow
    - op.targets: For x (delete multiple)
    - op.source: For m, mpage, cpage (the block being moved/copied)

    NOT targets (safe to duplicate):
    - op.parent: Where to add children (+ command)
    - op.after: Insertion position
    - op.dest_parent: Move destination
    - op.database: Which DB to add row to

    Returns:
        List of block IDs this operation targets.
    """
    targets = []
    if op.target:
        targets.append(op.target)
    if op.targets:
        targets.extend(op.targets)
    if op.source:
        targets.append(op.source)
    return targets


def _detect_conflicts(ops: list[ApplyOp]) -> list[str]:
    """Detect operations that conflict in parallel execution.

    Parallel execution requires:
    1. Each block be targeted by at most ONE operation per script
    2. No 'after' references to blocks being moved/deleted (ID will change)

    Conflict types:
    - Two updates to same block (last-write-wins, non-deterministic)
    - Update + delete same block (update may 404)
    - Move + update same block (move archives original, update 404s)
    - Two moves of same block (second archive may fail)
    - after=X where X is being moved (X gets new ID, after fails)
    - after=X where X is being deleted (X archived, after may fail)

    Note: parent references are NOT conflicts - multiple ops can add
    children to the same parent.

    Args:
        ops: List of parsed ApplyOp objects.

    Returns:
        List of error messages. Empty if no conflicts.
    """
    target_map: dict[str, list[int]] = {}  # id -> [line_nums]
    moved_ids: dict[str, int] = {}  # source id -> line_num (for moves)
    deleted_ids: dict[str, int] = {}  # target id -> line_num (for deletes)

    # First pass: collect targets, moved IDs, and deleted IDs
    for op in ops:
        targets = _get_op_targets(op)
        for target in targets:
            target_map.setdefault(target, []).append(op.line_num)

        # Track move sources (these IDs will become invalid)
        if op.command == ApplyCommand.MOVE and op.source:
            moved_ids[op.source] = op.line_num
        if op.command == ApplyCommand.MOVE_PAGE and op.source:
            # Page moves don't change ID, so don't track them
            pass

        # Track deleted IDs
        if op.command == ApplyCommand.DELETE and op.targets:
            for t in op.targets:
                deleted_ids[t] = op.line_num
        if op.command == ApplyCommand.DELETE_PAGE and op.target:
            deleted_ids[op.target] = op.line_num
        if op.command == ApplyCommand.DELETE_ROW and op.target:
            deleted_ids[op.target] = op.line_num

    errors = []

    # Check for duplicate targets
    for target, lines in target_map.items():
        if len(lines) > 1:
            errors.append(f"ID {target} targeted by lines {lines}")

    # Second pass: check for illegal 'after' references
    for op in ops:
        after_id = None
        if op.command == ApplyCommand.ADD and op.after:
            after_id = op.after
        elif op.command == ApplyCommand.MOVE and op.dest_after:
            after_id = op.dest_after

        if after_id:
            if after_id in moved_ids:
                errors.append(
                    f"Line {op.line_num}: after={after_id} invalid - "
                    f"block is moved on line {moved_ids[after_id]} (ID will change)"
                )
            if after_id in deleted_ids:
                errors.append(
                    f"Line {op.line_num}: after={after_id} invalid - "
                    f"block is deleted on line {deleted_ids[after_id]}"
                )

    return errors


def _identify_move_chains(ops: list[ApplyOp]) -> list[list[int]]:
    """Identify consecutive moves to the same parent needing sequential execution.

    When multiple moves target the same parent without explicit after=,
    parallel execution produces non-deterministic ordering. This function
    finds runs of such moves so the executor can run them sequentially
    and auto-chain each move after the previous one.

    Returns list of chains, each a list of op indices (length >= 2).
    Single moves and non-move ops are not included.
    """
    chains: list[list[int]] = []
    current_chain: list[int] = []
    current_parent: Optional[str] = None

    for i, op in enumerate(ops):
        is_chainable = (
            op.command == ApplyCommand.MOVE
            and op.dest_parent is not None
            and op.dest_after is None
        )

        if is_chainable and op.dest_parent == current_parent:
            current_chain.append(i)
        else:
            if len(current_chain) >= 2:
                chains.append(current_chain)
            if is_chainable:
                current_chain = [i]
                current_parent = op.dest_parent
            else:
                current_chain = []
                current_parent = None

    if len(current_chain) >= 2:
        chains.append(current_chain)

    return chains


async def execute_apply_script(
    script: str,
    registry: IdRegistry,
    dry_run: bool = False,
    allow_partial: bool = False  # Deprecated, ignored
) -> ApplyResult:
    """Execute an apply script with parallel execution.

    EXECUTION MODEL:
    All operations run concurrently via asyncio.gather(). This assumes:

    1. All IDs pre-exist - Every ID in the script must already be in the
       registry from a prior notion_read/search. Ops cannot reference IDs
       created by other ops in the same script.

    2. No duplicate targets - Each block ID can be targeted by only ONE
       operation. Conflict detection runs before execution; if the same
       ID appears in multiple ops, returns CONFLICT_DETECTED error.

    3. Parent/after are not conflicts - Multiple ops can add to the same
       parent or use the same after anchor.

    4. Move changes IDs - The m command clones and archives; new ID is
       returned in id_map. Old ID becomes invalid after the move.

    RATE LIMITING:
    Shared semaphore (3 concurrent) in _notion_request_async handles
    Notion API limits. Exponential backoff on 429.

    RESULT ORDERING:
    Results ordered by script line number, not completion order.

    Args:
        script: The apply script text.
        registry: ID registry for resolving refs.
        dry_run: If True, validate only without executing.
        allow_partial: Deprecated, ignored. All operations always run.

    Returns:
        ApplyResult with per-line outcomes.
    """
    # Parse the script
    parse_result = parse_apply_script(script, registry)

    if parse_result.errors:
        # Return parse error as a line result
        err = parse_result.errors[0]
        return ApplyResult(
            ok=False,
            results=[ApplyLineResult(
                line=err.line,
                ok=False,
                op="parse",
                error=err.message
            )]
        )

    # Detect conflicts before execution (same ID targeted by multiple ops)
    conflicts = _detect_conflicts(parse_result.operations)
    if conflicts:
        return ApplyResult(
            ok=False,
            results=[ApplyLineResult(
                line=0,
                ok=False,
                op="conflict",
                error=f"CONFLICT_DETECTED: {'; '.join(conflicts)}"
            )]
        )

    if dry_run:
        return ApplyResult(
            ok=True,
            results=[ApplyLineResult(line=op.line_num, ok=True, op=op.command.value)
                     for op in parse_result.operations]
        )

    # Identify move chains: consecutive moves to the same parent without
    # explicit after=. These must run sequentially to preserve script order.
    ops = parse_result.operations
    chains = _identify_move_chains(ops)
    chained_indices = {idx for chain in chains for idx in chain}

    if not chains:
        # Fast path: no chains, execute everything in parallel
        op_results = await asyncio.gather(
            *[execute_apply_op(op, registry) for op in ops],
            return_exceptions=True
        )
    else:
        # Mixed execution: parallel for independent ops, sequential for chains
        op_results_arr: list[Any] = [None] * len(ops)

        async def _run_move_chain(
            chain_indices: list[int],
        ) -> list[tuple[int, Any]]:
            """Execute a move chain sequentially, auto-chaining after=."""
            results: list[tuple[int, Any]] = []
            prev_new_short: Optional[str] = None
            for idx in chain_indices:
                op = ops[idx]
                # Inject after= from previous move's new block ID
                if prev_new_short is not None:
                    op.dest_after = prev_new_short
                try:
                    res = await execute_apply_op(op, registry)
                    results.append((idx, res))
                    op_result, error = res
                    if not error and isinstance(op_result.get("moved"), dict):
                        for _, new_id in op_result["moved"].items():
                            prev_new_short = new_id
                    # On error, keep prev_new_short — next move chains
                    # after the last successfully moved block
                except Exception as e:
                    results.append((idx, e))
                    # Keep prev_new_short as-is
            return results

        # Build coroutines: one per independent op, one per chain
        parallel_coros = []
        parallel_indices = []
        for i, op in enumerate(ops):
            if i not in chained_indices:
                parallel_coros.append(execute_apply_op(op, registry))
                parallel_indices.append(i)

        chain_coros = [_run_move_chain(chain) for chain in chains]

        # Run everything in parallel (chains are internally sequential)
        all_results = await asyncio.gather(
            *parallel_coros, *chain_coros,
            return_exceptions=True
        )

        # Map parallel results back to their original indices
        for i, idx in enumerate(parallel_indices):
            op_results_arr[idx] = all_results[i]

        # Map chain results back to their original indices
        for i, chain in enumerate(chains):
            chain_result = all_results[len(parallel_coros) + i]
            if isinstance(chain_result, Exception):
                for idx in chain:
                    op_results_arr[idx] = chain_result
            else:
                for idx, res in chain_result:
                    op_results_arr[idx] = res

        op_results = op_results_arr

    # Process results in original line order
    result = ApplyResult(ok=True)

    for op, op_result_or_exc in zip(ops, op_results):
        # Handle exception case
        if isinstance(op_result_or_exc, Exception):
            result.ok = False
            result.results.append(ApplyLineResult(
                line=op.line_num,
                ok=False,
                op=op.command.value,
                error=f"{type(op_result_or_exc).__name__}: {op_result_or_exc}"
            ))
            continue

        op_result, error = op_result_or_exc

        if error:
            result.ok = False
            result.results.append(ApplyLineResult(
                line=op.line_num,
                ok=False,
                op=op.command.value,
                error=error
            ))
        else:
            # Extract created IDs and moved mapping
            created = op_result.get("created", [])
            moved = None
            if "moved" in op_result and isinstance(op_result["moved"], dict):
                for old_id, new_id in op_result["moved"].items():
                    moved = (old_id, new_id)
                    result.id_map[old_id] = new_id

            # Extract replaced mapping (for update with type change)
            replaced = None
            if "replaced" in op_result:
                old_id = op_result.get("updated", "")
                new_id = op_result["replaced"]
                replaced = (old_id, new_id)
                result.id_map[old_id] = new_id

            # Extract pages_at_bottom (child pages created without positioning)
            pages_at_bottom = op_result.get("pages_at_bottom", [])

            # Extract warnings (e.g., link_mention conversions)
            warnings = op_result.get("warnings", [])

            result.results.append(ApplyLineResult(
                line=op.line_num,
                ok=True,
                op=op.command.value,
                created=created,
                moved=moved,
                replaced=replaced,
                pages_at_bottom=pages_at_bottom,
                warnings=warnings
            ))

    return result


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP("notion-mcp", host="127.0.0.1", port=2052)

# Global ID registry (session-scoped in real implementation)
_id_registry = IdRegistry()


async def _read_impl(
    ref: str, mode: str, depth: int, limit: int,
    filter_dsl: str = "", sort_dsl: str = "", columns_dsl: str = "",
) -> str:
    """Core implementation for reading a Notion page/database/block.

    Returns DNN formatted content or an error string.
    """
    global _id_registry

    has_db_params = bool(filter_dsl.strip() or sort_dsl.strip() or columns_dsl.strip())

    # Resolve reference to UUID
    object_id = _id_registry.resolve(ref)
    if not object_id:
        # Try parsing as UUID or URL directly
        if UUID_PATTERN.match(ref):
            object_id = normalize_uuid(ref)
        else:
            extracted = extract_uuid_from_url(ref)
            if extracted:
                object_id = extracted
            else:
                return _error(
                    "UNKNOWN_ID",
                    "Could not resolve reference",
                    hint=HINTS["unknown_id"],
                    ref=ref
                )

    # Try as page first, then as database
    try:
        page = await fetch_page_async(object_id)
        # Success - it's a page
        if has_db_params:
            return _error(
                "DB_PARAMS_ON_PAGE",
                "filter/sort/columns only apply to databases, but ref resolved to a page",
                hint=HINTS["db_params_on_page"],
                ref=ref
            )
        blocks = await fetch_block_children_async(object_id, depth=depth)
        dnn = render_page_to_dnn(page, blocks, _id_registry, mode=mode)
        return dnn

    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code in (404, 400):
            # Not a page (404) or is a database (400), try as database
            pass
        elif e.response is not None and e.response.status_code == 403:
            return _error(
                "MISSING_CAPABILITY",
                "Integration lacks access to this page",
                hint=HINTS["missing_capability"],
                ref=object_id
            )
        elif e.response is not None and e.response.status_code == 429:
            return _error("RATE_LIMITED", "Too many requests", hint=HINTS["rate_limited"])
        elif e.response is not None:
            return _error("HTTP_ERROR", f"HTTP {e.response.status_code}: {_http_error_detail(e, 100)}")
        else:
            return _error("HTTP_ERROR", str(e))

    # Try as database (API 2025-09-03: database → data_source two-step)
    try:
        # Step 1: Get database container to find data_source_id
        database = await fetch_database_async(object_id)
        data_sources = database.get("data_sources", [])

        if not data_sources:
            return _error(
                "NO_DATA_SOURCES",
                "Database has no data sources",
                hint=HINTS["no_data_sources"],
                ref=object_id
            )

        # Use the first data source (most databases have exactly one)
        data_source_id = data_sources[0].get("id")
        if not data_source_id:
            return _error("INVALID_DATA_SOURCE", "Data source missing ID", ref=object_id)

        if has_db_params:
            # Need schema first to compile filter/sort/columns
            data_source = await fetch_data_source_async(data_source_id)

            # Parse and compile filter
            filter_obj = None
            if filter_dsl.strip():
                try:
                    ast = parse_filter_dsl(filter_dsl)
                    filter_obj = compile_filter(ast, data_source)
                except FilterParseError as e:
                    return _error(
                        "FILTER_ERROR", str(e),
                        hint=HINTS["filter_parse_error"], ref=filter_dsl
                    )

            # Parse sort
            sorts = None
            if sort_dsl.strip():
                try:
                    sorts = parse_sort_dsl(sort_dsl, data_source)
                except FilterParseError as e:
                    return _error("SORT_ERROR", str(e), ref=sort_dsl)

            # Parse columns
            columns_list: Optional[list[str]] = None
            if columns_dsl.strip():
                try:
                    columns_list = parse_columns_dsl(columns_dsl, data_source)
                except FilterParseError as e:
                    return _error("COLUMNS_ERROR", str(e), ref=columns_dsl)

            # Query with compiled params
            rows, has_more = await query_data_source_async(
                data_source_id, filter_obj=filter_obj, sorts=sorts, limit=limit
            )
        else:
            # No DB params — fetch schema and rows in parallel (original fast path)
            data_source, (rows, has_more) = await asyncio.gather(
                fetch_data_source_async(data_source_id),
                query_data_source_async(data_source_id, limit=limit)
            )
            columns_list = None

        # Merge database title into data_source for rendering
        # (database has the title, data_source has the properties)
        data_source["title"] = database.get("title", [])
        data_source["database_id"] = object_id

        dnn = render_database_to_dnn(
            data_source, rows, _id_registry, mode=mode,
            columns=columns_list, has_more=has_more
        )
        return dnn

    except httpx.HTTPStatusError as e:
        if e.response is not None:
            if e.response.status_code == 404:
                # Not a page or database - try as a block
                if has_db_params:
                    return _error(
                        "DB_PARAMS_ON_PAGE",
                        "filter/sort/columns only apply to databases, but ref resolved to a block",
                        hint=HINTS["db_params_on_page"],
                        ref=ref
                    )
                try:
                    block = await _notion_request_async("GET", f"/blocks/{object_id}")
                    # Success! Render this block (and children if any)
                    blocks = [block]
                    if block.get("has_children"):
                        children = await fetch_block_children_async(object_id, depth=depth)
                        blocks.extend(children)
                    # Render as DNN (block mode - no page header)
                    dnn = render_blocks_to_dnn(blocks, _id_registry, mode=mode)
                    return dnn
                except Exception:
                    # Block fetch also failed - truly not found
                    return _error(
                        "REF_GONE",
                        "Object not found",
                        hint=HINTS["ref_gone"],
                        ref=object_id
                    )
            elif e.response.status_code == 403:
                return _error(
                    "MISSING_CAPABILITY",
                    "Integration lacks access to this database",
                    hint=HINTS["missing_capability"],
                    ref=object_id
                )
            elif e.response.status_code == 429:
                return _error("RATE_LIMITED", "Too many requests", hint=HINTS["rate_limited"])
            else:
                return _error("HTTP_ERROR", f"HTTP {e.response.status_code}: {_http_error_detail(e, 100)}")
        else:
            return _error("HTTP_ERROR", str(e))
    except Exception as e:
        return _error("UNEXPECTED", f"{type(e).__name__}: {e}")


@mcp.tool()
async def notion_read(
    pages: list[dict],
    depth: int = 10,
    limit: int = 50,
    filter: str = "",
    sort: str = "",
    columns: str = "",
) -> str:
    """Read Notion pages or databases in compact DNN format.

    Args:
        pages: List of pages/databases to read. Each entry has:
            - ref (required): short ID, full UUID, or Notion URL
            - mode (optional): "edit" (default) or "view"
            - depth (optional): page nesting depth for this ref
            - limit (optional): DB row limit for this ref
            - filter (optional): DB filter DSL for this ref
            - sort (optional): DB sort for this ref
            - columns (optional): DB columns for this ref
            Per-ref filter/sort/columns override the top-level defaults,
            enabling batch reads that mix pages and filtered databases:
            [{"ref": "page1"}, {"ref": "db1", "filter": "Status = Done", "columns": "Name, Status"}]
        depth: Maximum nesting depth for pages (default 10)
        limit: Maximum database rows (default 50)
        filter: Database filter DSL (default for all refs).
            Operators: = != ~ !~ < > <= >= ? !?
            Logic: & (and), | (or), () grouping.
            Quote names with spaces: "Last Contact" >= 2024-01-01
        sort: Database sort (default for all refs).
            E.g. 'Due desc, Status asc'. Default direction: ascending.
        columns: Database columns (default for all refs).
            Comma-separated (e.g. 'Name, Status, Due'). Empty = all columns.

    Returns:
        DNN formatted content. Multiple pages separated by ▤ headers.
        Returns error if filter/sort/columns used on a non-database ref.
    """
    # Parse page specs — per-ref params override top-level defaults
    specs: list[tuple[str, str, int, int, str, str, str]] = []
    # (ref, mode, depth, limit, filter, sort, columns)
    for entry in pages:
        ref = entry.get("ref", "")
        if not ref:
            continue
        mode = entry.get("mode", "edit")
        if mode not in ("edit", "view"):
            mode = "edit"
        ref_depth = entry.get("depth", depth)
        ref_limit = entry.get("limit", limit)
        ref_filter = entry.get("filter", filter)
        ref_sort = entry.get("sort", sort)
        ref_columns = entry.get("columns", columns)
        specs.append((ref, mode, ref_depth, ref_limit, ref_filter, ref_sort, ref_columns))

    if not specs:
        return _error("EMPTY_BATCH", "No pages specified")

    # Read all pages in parallel
    results = await asyncio.gather(
        *(_read_impl(ref, mode, d, l, f, s, c)
          for ref, mode, d, l, f, s, c in specs)
    )

    # Single page: return content directly (no header)
    if len(specs) == 1:
        return results[0]

    # Multiple pages: format with ▤ headers
    total = len(specs)
    sections = []
    for i, ((ref, mode, *_rest), content) in enumerate(zip(specs, results), 1):
        sections.append(f"▤ [{i}/{total}] {ref} ({mode})\n{content}")

    return "\n\n".join(sections)


@mcp.tool()
def notion_check_auth() -> str:
    """Verify Notion authentication and return workspace info.

    Returns a message indicating whether authentication succeeded
    and basic info about the connected workspace.
    """
    try:
        token = _get_token()
    except RuntimeError as e:
        return _error("NO_TOKEN", str(e), hint=HINTS["invalid_token"])

    try:
        result = _notion_request("GET", "/users/me")

        bot_name = result.get("name", "Unknown")
        bot_type = result.get("type", "unknown")
        bot_info = result.get("bot", {})
        workspace_name = bot_info.get("workspace_name", "Unknown workspace")

        return (
            f"authenticated as '{bot_name}' ({bot_type}) "
            f"in workspace '{workspace_name}'"
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return _error("INVALID_TOKEN", "Token is invalid or expired", hint=HINTS["invalid_token"])
        elif e.response.status_code == 403:
            return _error("INSUFFICIENT_PERMISSIONS", "Token lacks required permissions")
        else:
            return _error("HTTP_ERROR", f"HTTP {e.response.status_code}")
    except Exception as e:
        return _error("UNEXPECTED", f"{type(e).__name__}: {e}")


@mcp.tool()
def notion_get_url(ref: str) -> str:
    """Get the Notion URL for a page, block, or database.

    Converts any reference format to a full Notion URL that can be opened
    in a browser.

    Args:
        ref: Any reference format:
            - Short ID (from this session): A1b2
            - Full UUID: 12345678-1234-1234-1234-123456789abc
            - Notion URL (returned as-is)

    Returns:
        Notion URL: https://notion.so/<uuid-without-dashes>
        Or error message if reference cannot be resolved.
    """
    global _id_registry

    # If it's already a Notion URL, return it
    if ref.startswith("http"):
        uuid = extract_uuid_from_url(ref)
        if uuid:
            return f"https://notion.so/{uuid.replace('-', '')}"
        return ref  # Return as-is if we can't parse it

    # Try to resolve the reference
    uuid = _id_registry.resolve(ref)
    if uuid:
        return f"https://notion.so/{uuid.replace('-', '')}"

    # Try parsing as UUID directly
    if UUID_PATTERN.match(ref):
        try:
            normalized = normalize_uuid(ref)
            return f"https://notion.so/{normalized.replace('-', '')}"
        except ValueError:
            pass

    # Check if it looks like a short ID (4 alphanumeric chars)
    if SHORT_ID_PATTERN.match(ref):
        return _error(
            "UNKNOWN_ID",
            f"Short ID '{ref}' not found in session registry",
            hint="Short IDs are session-scoped and cleared on server restart. "
                 "Use notion_search to find the page by title, or provide a full Notion UUID/URL."
        )

    return _error(
        "UNKNOWN_ID",
        f"Could not resolve reference: {ref}",
        hint=HINTS["unknown_id"]
    )


@mcp.tool()
async def notion_search(
    query: str,
    filter_type: str = "all",
    limit: int = 20
) -> str:
    """Search Notion by title.

    Args:
        query: Search query (matched against page/database titles).
        filter_type: Filter results - "page", "database", or "all" (default).
        limit: Maximum results to return (default 20, max 100).

    Returns:
        Compact search results with short IDs for use with notion_read.
    """
    global _id_registry

    # Build request body
    body: dict = {"query": query, "page_size": min(limit, 100)}

    # API 2025-09-03: filter uses "data_source" instead of "database"
    if filter_type == "page":
        body["filter"] = {"property": "object", "value": "page"}
    elif filter_type == "database":
        body["filter"] = {"property": "object", "value": "data_source"}

    # Sort by last edited (most recent first)
    body["sort"] = {"direction": "descending", "timestamp": "last_edited_time"}

    try:
        result = await _notion_request_async("POST", "/search", json_body=body)
    except httpx.HTTPStatusError as e:
        if e.response is not None:
            if e.response.status_code == 429:
                return _error("RATE_LIMITED", "Too many requests", hint=HINTS["rate_limited"])
            return _error("HTTP_ERROR", f"HTTP {e.response.status_code}: {_http_error_detail(e, 100)}")
        return _error("HTTP_ERROR", str(e))
    except Exception as e:
        return _error("UNEXPECTED", f"{type(e).__name__}: {e}")

    # Format results compactly
    lines = []
    for item in result.get("results", []):
        obj_type = item.get("object", "unknown")
        obj_id = item.get("id", "")

        # Extract title and icon
        icon = item.get("icon", {})
        icon_str = ""
        if icon and icon.get("type") == "emoji":
            icon_str = icon.get("emoji", "") + " "

        if obj_type == "page":
            # Register page ID
            try:
                short_id = _id_registry.register(obj_id)
            except ValueError:
                short_id = obj_id[:8]

            # Extract title from properties
            props = item.get("properties", {})
            title = ""
            for prop in props.values():
                if prop.get("type") == "title":
                    title_arr = prop.get("title", [])
                    title = "".join(t.get("plain_text", "") for t in title_arr)
                    break
            lines.append(f"{short_id} page  {icon_str}{title}")

        elif obj_type == "data_source":
            # API 2025-09-03 returns data_source for databases
            # Register database_id (from parent) since that's what notion_read needs
            parent = item.get("parent", {})
            db_id = parent.get("database_id", obj_id)  # Fallback to data_source_id

            try:
                short_id = _id_registry.register(db_id)
            except ValueError:
                short_id = db_id[:8]

            # Title is directly on data_source object
            title_arr = item.get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_arr)
            lines.append(f"{short_id} db    {icon_str}{title}")

    if not lines:
        return f"No results for '{query}'"

    header = f"Found {len(lines)} result(s) for '{query}':\n"
    return header + "\n".join(lines)


@mcp.tool()
async def notion_apply(
    script: str = "",
    operations: str = "",
    action: str = "",
    dry_run: bool = False,
    allow_partial: bool = False
) -> str:
    """Execute mutations on Notion pages.

    PARAMETER: Use `script` (not `operations`, `action`, or `commands`).

    Args:
        script: Apply script with mutation commands. See examples below.
        operations: Alias for script. Use one or the other, not both.
        action: Alias for script. Use one or the other, not both.
        dry_run: If True, validate script without executing (default False).
        allow_partial: If True, continue on errors (default False).

    Block Commands:
        + parent=ID [after=ID]    Add blocks (indented content follows)
        x ID [ID2 ID3...]         Delete/archive blocks
        m ID -> parent=ID [after=ID]  Move block (parent= always required,
                                       even for same-parent reposition)
        u ID = "text"             Update block text (can change type!)
        t ID = 0|1                Toggle todo checkbox

    Update with Type Change:
        The u command can change block types by including the marker:
        u ID = ">## :yellow[New Title]"   # h3 toggle → h2 yellow toggle
        u ID = "- bullet item"            # paragraph → bullet
        u ID = "[ ] task"                 # paragraph → todo
        When type changes, block is replaced (children preserved, ID changes).
        Output shows old→new ID: "0: OldID→NewID"

    Block Types (indent 2 spaces per nesting level):
        Plain paragraph text
        # Heading 1    ## Heading 2    ### Heading 3
        ># Toggle H1   >## Toggle H2   >### Toggle H3
        - Bullet       1. Numbered
        [ ] Todo       [x] Checked todo
        > Toggle       | Quote
        ! Callout      !gray Gray callout   !red Warning callout
        ---            (divider)
        ```python      (code block, close with ```)

    Callout colors: gray, brown, orange, yellow, green, blue, purple, pink, red

    Inline Formatting (in any text content):
        **bold**  *italic*  ~~strike~~  `code`  :u[underline]
        :red[colored]  :yellow-background[highlighted]
        [link text](https://url)
        $x^2 + y^2$              (equation)
        @user:uuid-here          (user mention)
        @date:2025-01-15         (date mention)
        @date:2025-01-14→2025-01-16  (date range)
        @p:shortID               (page @mention - PREFERRED)
        [custom text](p:shortID) (link to page with custom text)

    Page References:
        @p:shortID - Creates a Notion @mention showing the page's actual title.
                     Title updates automatically if page is renamed. PREFERRED.
        [text](p:shortID) - Creates a hyperlink with YOUR custom text.
                            Use when you need specific link text.

    Escape: \\n (newline) \\t (tab) \\\\ (backslash) \\" (quote)
    Escape block markers: \\# Not a heading  \\- Not a bullet

    Page Commands:
        +page parent=ID title="Title" [icon=📝] [cover=URL]
          Content blocks...
        +page parent=ID after=BLOCK title="Title"   (after specific block)
        +page parent=ID pos=start title="Title"      (at top of parent)
        mpage ID -> parent=ID
        xpage ID
        upage ID [= "New Title"] [icon=📝] [cover=URL]
        cpage ID -> parent=ID [title="Copy Title"]

    Database Schema Commands:
        +db parent=ID title="DB Name" [icon=📊]
          Name(title)
          Status(select: Todo, Doing, Done)
          Due(date)
          Done(checkbox)
          Priority(number: dollar)
          Tags(multi_select: Bug, Feature)
          TaskID(id: TASK)
          Projects(relation: DbId)
          Tasks(relation: DbId, dual: "Back Ref Name")

        +prop db=ID PropName(type)
        +prop db=ID PropName(type: config)
        xprop db=ID PropName
        xprop db=ID "Prop Name With Spaces"
        uprop db=ID OldName -> "New Name"

    Property Types:
        title, text, number, select, multi_select, date,
        checkbox, url, email, phone, id, relation,
        created, created_by, edited, edited_by

    Database Row Commands:
        +row db=ID
          Title=Task name	Status=Todo	Due=2025-01-15→2025-01-20
        urow ID
          Status=Done	Priority=High
        xrow ID

    Row Value Types:
        Title/Text: Name=My Task
        Select: Status=Done
        Multi-select: Tags=urgent,important
        Checkbox: Done=true  (or: 1, yes, x)
        Number: Priority=5
        Date: Due=2025-01-15
        Date range: Period=2025-01-01→2025-01-31
        URL: Link=https://example.com
        Email: Contact=user@example.com

    Returns:
        Compact multiline string with per-line results (0-indexed):
            0: +A1b2 +B2c3    (created IDs)
            2: X1y2→Y2z3      (moved: old→new)
            3: A1b2→B2c3      (updated with type change: old→new)
            4: err MESSAGE    (error)
            7: ok             (success, no IDs)
            ---
            3/4 ok            (summary)

    Example - Complex nested structure with formatting:
        + parent=A1b2
          ## 📋 Project Setup
          !blue Overview
            This project uses **Python 3.11** with `asyncio`.
            See [documentation](https://docs.example.com) for details.
          ># Requirements
            - Python >= 3.11
            - :red[Required]: `httpx` library
            1. Install dependencies
            2. Configure settings
          [ ] Review :u[security] considerations
          [x] Initial setup complete

    Moving Blocks:
        Always use m to move blocks. Do not manually read → delete → add
        — that round-trips through DNN and loses formatting/mentions.
        m A1b2 -> parent=C3d4              Move to end of C3d4
        m A1b2 -> parent=C3d4 after=E5f6   Move after E5f6
        m A1b2 -> parent=SAME after=X      Reposition (same parent)

    Example - Batch operations:
        x Old1 Old2 Old3
        m Source1 -> parent=Target after=Anchor
        u Text1 = "Updated with **bold** and :green[color]"
        t Todo1 = 1

    Example - Database row with date range:
        +row db=TasksDB
          Name=Q1 Planning\tStatus=In Progress\tPeriod=2025-01-01→2025-03-31
    """
    global _id_registry

    # Resolve script/operations/action alias
    provided = [name for name, val in [("script", script), ("operations", operations), ("action", action)] if val]
    if not provided:
        raise ValueError("Missing required parameter: script")
    if len(provided) > 1:
        raise ValueError(f"Use only one of script/operations/action, got: {', '.join(provided)}")
    resolved = script or operations or action

    result = await execute_apply_script(
        resolved,
        _id_registry,
        dry_run=dry_run,
        allow_partial=allow_partial
    )

    # Format result as compact multiline string
    lines = []
    for lr in result.results:
        if lr.ok:
            if lr.created:
                # +ID1 +ID2 for created blocks
                line_text = f"{lr.line}: +" + " +".join(lr.created)
                if lr.pages_at_bottom:
                    # Indicate which pages were created at bottom (no positioning)
                    line_text += f" (at bottom: {', '.join(lr.pages_at_bottom)})"
                lines.append(line_text)
            elif lr.moved:
                # OLD→NEW for moves
                line_text = f"{lr.line}: {lr.moved[0]}→{lr.moved[1]}"
                lines.append(line_text)
            elif lr.replaced:
                # OLD→NEW for update with type change
                line_text = f"{lr.line}: {lr.replaced[0]}→{lr.replaced[1]}"
                lines.append(line_text)
            else:
                lines.append(f"{lr.line}: ok")

            # Append warnings if any (e.g., link_mention conversions)
            if lr.warnings:
                for warning in lr.warnings:
                    lines.append(f"  ⚠ {warning}")
        else:
            lines.append(f"{lr.line}: err {lr.error}")

    # Summary line
    ok_count = sum(1 for lr in result.results if lr.ok)
    total = len(result.results)
    lines.append("---")
    lines.append(f"{ok_count}/{total} ok")

    return "\n".join(lines)


# =============================================================================
# HTTP Endpoints (/health, /kill)
# =============================================================================

async def health_endpoint(request: Request) -> JSONResponse:
    """Health check endpoint for easy testing."""
    token_loaded = _notion_token is not None

    # If token loaded, try a quick auth check
    auth_status = None
    if token_loaded:
        try:
            result = _notion_request("GET", "/users/me")
            bot_info = result.get("bot", {})
            auth_status = bot_info.get("workspace_name", "connected")
        except Exception as e:
            auth_status = f"error: {type(e).__name__}"

    return JSONResponse({
        "status": "ok",
        "token_loaded": token_loaded,
        "workspace": auth_status,
    })


async def kill_endpoint(request: Request) -> JSONResponse:
    """Kill the server for restart. Usage: curl http://localhost:2052/kill"""
    import os
    import threading

    def delayed_exit():
        import time
        time.sleep(0.1)  # Give time for response to be sent
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return JSONResponse({"status": "shutting_down"})


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the Notion MCP server.

    Supports two transport modes:
    - stdio (default): For Claude Code to launch directly
    - http: For standalone server on port 2052

    Usage:
        uv run notion-mcp          # stdio mode (Claude Code launches this)
        uv run notion-mcp --http   # HTTP mode on localhost:2052
    """
    import argparse

    parser = argparse.ArgumentParser(description="Notion MCP Server")
    parser.add_argument(
        "--token-file",
        required=True,
        help="Path to file containing Notion API token"
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run as HTTP server on localhost:2052 instead of stdio"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Load token from file
    global _notion_token
    token_path = Path(args.token_file).expanduser()
    if not token_path.exists():
        raise SystemExit(f"Token file not found: {token_path}")
    _notion_token = token_path.read_text().strip()
    if not _notion_token:
        raise SystemExit(f"Token file is empty: {token_path}")
    logger.info(f"Loaded Notion token from {token_path}")

    if args.http:
        # HTTP mode: standalone server with health/kill endpoints
        import uvicorn

        app = mcp.streamable_http_app()
        app.add_route("/health", health_endpoint, methods=["GET"])
        app.add_route("/kill", kill_endpoint, methods=["GET", "POST"])

        logger.info("Starting Notion MCP server on http://127.0.0.1:2052")
        uvicorn.run(app, host="127.0.0.1", port=2052, log_level="warning")
    else:
        # stdio mode: Claude Code launches this process
        mcp.run()


if __name__ == "__main__":
    main()
