"""Tests for notion_mcp module - ID system and DNN parser."""

import pytest
from notion_mcp import (
    ApplyCommand,
    ApplyOp,
    BASE62_ALPHABET,
    DnnBlock,
    DnnHeader,
    DnnParseError,
    DnnParseResult,
    IdRegistry,
    ParseState,
    RichTextSpan,
    SHORT_ID_PATTERN,
    UUID_PATTERN,
    _build_property_value,
    _build_row_properties,
    _detect_conflicts,
    _fetch_database_schema,
    _parse_content_blocks,
    _text_to_rich_text,
    calculate_indent_level,
    extract_uuid_from_url,
    generate_short_id,
    normalize_uuid,
    parse_apply_script,
    parse_block_type,
    parse_dnn,
    parse_inline_formatting,
    rich_text_spans_to_notion,
    strip_marker_for_block,
)


class TestGenerateShortId:
    """Tests for generate_short_id function."""

    def test_generates_4_char_id(self):
        sid = generate_short_id()
        assert len(sid) == 4

    def test_uses_base62_alphabet(self):
        sid = generate_short_id()
        for char in sid:
            assert char in BASE62_ALPHABET

    def test_avoids_existing_ids(self):
        existing = {"A1b2", "C3d4", "E5f6"}
        for _ in range(100):
            sid = generate_short_id(existing)
            assert sid not in existing

    def test_matches_short_id_pattern(self):
        for _ in range(100):
            sid = generate_short_id()
            assert SHORT_ID_PATTERN.match(sid)


class TestNormalizeUuid:
    """Tests for normalize_uuid function."""

    def test_normalizes_uuid_with_dashes(self):
        uuid = "12345678-1234-1234-1234-123456789abc"
        assert normalize_uuid(uuid) == "12345678-1234-1234-1234-123456789abc"

    def test_normalizes_uuid_without_dashes(self):
        uuid = "12345678123412341234123456789abc"
        assert normalize_uuid(uuid) == "12345678-1234-1234-1234-123456789abc"

    def test_lowercases_uuid(self):
        uuid = "12345678-1234-1234-1234-123456789ABC"
        assert normalize_uuid(uuid) == "12345678-1234-1234-1234-123456789abc"

    def test_raises_on_invalid_length(self):
        with pytest.raises(ValueError):
            normalize_uuid("1234567")

    def test_raises_on_too_long(self):
        with pytest.raises(ValueError):
            normalize_uuid("12345678123412341234123456789abcdef")


class TestExtractUuidFromUrl:
    """Tests for extract_uuid_from_url function."""

    def test_extracts_from_notion_so(self):
        url = "https://notion.so/workspace/Page-Title-12345678123412341234123456789abc"
        uuid = extract_uuid_from_url(url)
        assert uuid == "12345678-1234-1234-1234-123456789abc"

    def test_extracts_from_www_notion_so(self):
        url = "https://www.notion.so/Page-12345678123412341234123456789abc"
        uuid = extract_uuid_from_url(url)
        assert uuid == "12345678-1234-1234-1234-123456789abc"

    def test_extracts_uuid_with_dashes(self):
        url = "https://notion.so/12345678-1234-1234-1234-123456789abc"
        uuid = extract_uuid_from_url(url)
        assert uuid == "12345678-1234-1234-1234-123456789abc"

    def test_returns_none_for_invalid_url(self):
        url = "https://google.com/page"
        assert extract_uuid_from_url(url) is None

    def test_returns_none_for_short_id_only(self):
        # Short IDs in URLs aren't full UUIDs
        url = "https://notion.so/Page-abc123"
        assert extract_uuid_from_url(url) is None


class TestIdRegistry:
    """Tests for IdRegistry class."""

    def test_register_generates_short_id(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert len(sid) == 4
        assert SHORT_ID_PATTERN.match(sid)

    def test_register_returns_same_id_for_same_uuid(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid1 = registry.register(uuid)
        sid2 = registry.register(uuid)
        assert sid1 == sid2

    def test_register_accepts_preferred_short_id(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid, short_id="A1b2")
        assert sid == "A1b2"

    def test_register_ignores_taken_short_id(self):
        registry = IdRegistry()
        uuid1 = "12345678-1234-1234-1234-123456789abc"
        uuid2 = "abcdef12-3456-7890-abcd-ef1234567890"
        sid1 = registry.register(uuid1, short_id="A1b2")
        sid2 = registry.register(uuid2, short_id="A1b2")
        assert sid1 == "A1b2"
        assert sid2 != "A1b2"

    def test_get_uuid_returns_uuid_for_short_id(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert registry.get_uuid(sid) == uuid

    def test_get_uuid_returns_none_for_unknown(self):
        registry = IdRegistry()
        assert registry.get_uuid("A1b2") is None

    def test_get_short_id_returns_short_id_for_uuid(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert registry.get_short_id(uuid) == sid

    def test_get_short_id_returns_none_for_unknown(self):
        registry = IdRegistry()
        assert registry.get_short_id("12345678-1234-1234-1234-123456789abc") is None

    def test_resolve_short_id(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert registry.resolve(sid) == uuid

    def test_resolve_full_uuid(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        assert registry.resolve(uuid) == uuid

    def test_resolve_uuid_without_dashes(self):
        registry = IdRegistry()
        uuid_no_dash = "12345678123412341234123456789abc"
        assert registry.resolve(uuid_no_dash) == "12345678-1234-1234-1234-123456789abc"

    def test_resolve_typed_ref_page(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert registry.resolve(f"p:{sid}") == uuid

    def test_resolve_typed_ref_block(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert registry.resolve(f"b:{sid}") == uuid

    def test_resolve_typed_ref_row(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert registry.resolve(f"r:{sid}") == uuid

    def test_resolve_typed_ref_with_uuid(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        assert registry.resolve(f"p:{uuid}") == uuid

    def test_resolve_notion_url(self):
        registry = IdRegistry()
        url = "https://notion.so/Page-12345678123412341234123456789abc"
        assert registry.resolve(url) == "12345678-1234-1234-1234-123456789abc"

    def test_resolve_returns_none_for_unknown_short_id(self):
        registry = IdRegistry()
        assert registry.resolve("A1b2") is None

    def test_resolve_returns_none_for_invalid(self):
        registry = IdRegistry()
        assert registry.resolve("invalid") is None

    def test_len(self):
        registry = IdRegistry()
        assert len(registry) == 0
        registry.register("12345678-1234-1234-1234-123456789abc")
        assert len(registry) == 1
        registry.register("abcdef12-3456-7890-abcd-ef1234567890")
        assert len(registry) == 2

    def test_contains_short_id(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        assert sid in registry
        assert "zzzz" not in registry

    def test_contains_uuid(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        registry.register(uuid)
        assert uuid in registry
        assert "aaaaaaaa-1111-2222-3333-444444444444" not in registry

    def test_clear(self):
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc")
        registry.register("abcdef12-3456-7890-abcd-ef1234567890")
        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0


class TestIdRegistryRoundtrip:
    """Integration tests for full roundtrip scenarios."""

    def test_multiple_uuids_roundtrip(self):
        registry = IdRegistry()
        uuids = [
            "12345678-1234-1234-1234-123456789abc",
            "abcdef12-3456-7890-abcd-ef1234567890",
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        ]

        short_ids = [registry.register(uuid) for uuid in uuids]

        # All short IDs are unique
        assert len(set(short_ids)) == 3

        # Roundtrip works
        for uuid, sid in zip(uuids, short_ids):
            assert registry.get_uuid(sid) == uuid
            assert registry.get_short_id(uuid) == sid
            assert registry.resolve(sid) == uuid
            assert registry.resolve(uuid) == uuid

    def test_mixed_input_formats(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)

        # All these should resolve to the same UUID
        inputs = [
            sid,
            uuid,
            "12345678123412341234123456789abc",
            f"p:{sid}",
            f"b:{uuid}",
            "https://notion.so/Page-12345678123412341234123456789abc",
        ]

        for inp in inputs:
            assert registry.resolve(inp) == uuid, f"Failed for input: {inp}"


# =============================================================================
# DNN Parser Tests
# =============================================================================


class TestCalculateIndentLevel:
    """Tests for calculate_indent_level function."""

    def test_no_indent(self):
        level, content = calculate_indent_level("Hello")
        assert level == 0
        assert content == "Hello"

    def test_two_space_indent(self):
        level, content = calculate_indent_level("  Hello")
        assert level == 1
        assert content == "Hello"

    def test_four_space_indent(self):
        level, content = calculate_indent_level("    Hello")
        assert level == 2
        assert content == "Hello"

    def test_preserves_content_spaces(self):
        level, content = calculate_indent_level("  Hello World")
        assert level == 1
        assert content == "Hello World"


class TestParseBlockType:
    """Tests for parse_block_type function."""

    def test_paragraph_default(self):
        btype, content, attrs, _ = parse_block_type("Hello world")
        assert btype == "paragraph"
        assert content == "Hello world"
        assert attrs == {}

    def test_heading_1(self):
        btype, content, attrs, _ = parse_block_type("# Heading")
        assert btype == "heading_1"
        assert content == "Heading"
        assert attrs == {"heading_level": 1}

    def test_heading_2(self):
        btype, content, attrs, _ = parse_block_type("## Subheading")
        assert btype == "heading_2"
        assert content == "Subheading"
        assert attrs == {"heading_level": 2}

    def test_heading_3(self):
        btype, content, attrs, _ = parse_block_type("### Sub-subheading")
        assert btype == "heading_3"
        assert content == "Sub-subheading"
        assert attrs == {"heading_level": 3}

    def test_toggle_heading_1(self):
        btype, content, attrs, _ = parse_block_type("># Toggle H1")
        assert btype == "heading_1"
        assert content == "Toggle H1"
        assert attrs["is_toggle"] is True
        assert attrs["heading_level"] == 1

    def test_toggle_heading_2(self):
        btype, content, attrs, _ = parse_block_type(">## Toggle H2")
        assert btype == "heading_2"
        assert attrs["is_toggle"] is True

    def test_bulleted_list(self):
        btype, content, attrs, _ = parse_block_type("- Item")
        assert btype == "bulleted_list_item"
        assert content == "Item"

    def test_numbered_list(self):
        btype, content, attrs, _ = parse_block_type("1. First item")
        assert btype == "numbered_list_item"
        assert content == "First item"

    def test_numbered_list_large_number(self):
        btype, content, attrs, _ = parse_block_type("42. Item forty-two")
        assert btype == "numbered_list_item"
        assert content == "Item forty-two"

    def test_todo_unchecked(self):
        btype, content, attrs, _ = parse_block_type("[ ] Task")
        assert btype == "to_do"
        assert content == "Task"
        assert attrs["checked"] is False

    def test_todo_checked(self):
        btype, content, attrs, _ = parse_block_type("[x] Done task")
        assert btype == "to_do"
        assert content == "Done task"
        assert attrs["checked"] is True

    def test_todo_checked_uppercase(self):
        btype, content, attrs, _ = parse_block_type("[X] Also done")
        assert btype == "to_do"
        assert attrs["checked"] is True

    def test_toggle(self):
        btype, content, attrs, _ = parse_block_type("> Toggle content")
        assert btype == "toggle"
        assert content == "Toggle content"

    def test_quote(self):
        btype, content, attrs, _ = parse_block_type("| Quote text")
        assert btype == "quote"
        assert content == "Quote text"

    def test_callout(self):
        btype, content, attrs, _ = parse_block_type("! Important note")
        assert btype == "callout"
        assert content == "Important note"

    def test_divider(self):
        btype, content, attrs, _ = parse_block_type("---")
        assert btype == "divider"
        assert content == ""

    def test_child_page(self):
        btype, content, attrs, _ = parse_block_type("§ My Subpage")
        assert btype == "child_page"
        assert content == "My Subpage"

    def test_child_database(self):
        btype, content, attrs, _ = parse_block_type("⊞ Tasks Database")
        assert btype == "child_database"
        assert content == "Tasks Database"

    def test_link_to_page(self):
        btype, content, attrs, _ = parse_block_type("→ Linked Page")
        assert btype == "link_to_page"
        assert content == "Linked Page"

    def test_code_block_start(self):
        btype, content, attrs, _ = parse_block_type("```python")
        assert btype == "code"
        assert attrs["language"] == "python"

    def test_code_block_no_language(self):
        btype, content, attrs, _ = parse_block_type("```")
        assert btype == "code"
        assert attrs["language"] is None

    def test_escaped_heading(self):
        btype, content, attrs, _ = parse_block_type("\\# Not a heading")
        assert btype == "paragraph"
        assert content == "# Not a heading"

    def test_escaped_list(self):
        btype, content, attrs, _ = parse_block_type("\\- Not a list")
        assert btype == "paragraph"
        assert content == "- Not a list"

    def test_h4_heading_warning(self):
        """H4 and beyond should warn - Notion only supports h1-h3."""
        btype, content, attrs, warnings = parse_block_type("#### Too deep")
        assert btype == "paragraph"
        assert content == "#### Too deep"
        assert len(warnings) == 1
        assert "Notion only supports h1-h3" in warnings[0]
        assert "Did you mean '###'?" in warnings[0]

    def test_h5_heading_warning(self):
        """H5 should also warn."""
        btype, content, attrs, warnings = parse_block_type("##### Way too deep")
        assert btype == "paragraph"
        assert len(warnings) == 1
        assert "Notion only supports h1-h3" in warnings[0]

    def test_double_delimiter_warning(self):
        """Nested delimiters like '### ##' should warn."""
        btype, content, attrs, warnings = parse_block_type("### ## text")
        assert btype == "heading_3"
        assert content == "## text"
        assert len(warnings) == 1
        assert "looks like nested delimiters" in warnings[0]

    def test_valid_heading_no_warning(self):
        """Normal headings should not produce warnings."""
        btype, content, attrs, warnings = parse_block_type("### Normal heading")
        assert btype == "heading_3"
        assert content == "Normal heading"
        assert warnings == []


class TestStripMarkerForBlock:
    """Tests for strip_marker_for_block function.

    This function prevents double markers (e.g., "[ ] task" in to_do becomes
    "task" not "[ ] task" with checkbox). Used by both ADD and UPDATE paths.
    """

    def test_todo_marker_stripped_from_todo_block(self):
        """[ ] marker should be stripped when target is to_do."""
        clean, attrs, warnings = strip_marker_for_block("[ ] Buy milk", "to_do")
        assert clean == "Buy milk"
        assert attrs.get("checked") is False
        assert warnings == []

    def test_checked_todo_marker_stripped(self):
        """[x] marker should be stripped and attrs set."""
        clean, attrs, warnings = strip_marker_for_block("[x] Done task", "to_do")
        assert clean == "Done task"
        assert attrs.get("checked") is True
        assert warnings == []

    def test_heading_marker_stripped_from_heading(self):
        """# marker should be stripped when target is heading_1."""
        clean, attrs, warnings = strip_marker_for_block("# Section Title", "heading_1")
        assert clean == "Section Title"
        assert warnings == []

    def test_bullet_marker_stripped_from_bulleted_list(self):
        """- marker should be stripped when target is bulleted_list_item."""
        clean, attrs, warnings = strip_marker_for_block("- List item", "bulleted_list_item")
        assert clean == "List item"
        assert warnings == []

    def test_quote_marker_stripped_from_quote(self):
        """| marker should be stripped when target is quote."""
        clean, attrs, warnings = strip_marker_for_block("| Quote text", "quote")
        assert clean == "Quote text"
        assert warnings == []

    def test_toggle_marker_stripped_from_toggle(self):
        """> marker should be stripped when target is toggle."""
        clean, attrs, warnings = strip_marker_for_block("> Toggle content", "toggle")
        assert clean == "Toggle content"
        assert warnings == []

    def test_plain_text_unchanged(self):
        """Plain text without marker should pass through unchanged."""
        clean, attrs, warnings = strip_marker_for_block("Just some text", "paragraph")
        assert clean == "Just some text"
        assert warnings == []

    def test_plain_text_in_todo_unchanged(self):
        """Plain text without [ ] into to_do should pass through."""
        clean, attrs, warnings = strip_marker_for_block("Buy milk", "to_do")
        assert clean == "Buy milk"
        assert warnings == []

    def test_code_block_preserves_markers(self):
        """Code blocks should never strip markers (literal content)."""
        clean, attrs, warnings = strip_marker_for_block("# Not a heading", "code")
        assert clean == "# Not a heading"
        assert warnings == []

    def test_mismatched_marker_stripped_with_warning(self):
        """Mismatched marker should be stripped but produce warning."""
        clean, attrs, warnings = strip_marker_for_block("- bullet text", "to_do")
        assert clean == "bullet text"
        assert len(warnings) == 1
        assert "bulleted_list_item" in warnings[0]
        assert "to_do" in warnings[0]

    def test_heading_marker_into_paragraph_warns(self):
        """# marker into paragraph should strip and warn."""
        clean, attrs, warnings = strip_marker_for_block("# Title", "paragraph")
        assert clean == "Title"
        assert len(warnings) == 1
        assert "heading_1" in warnings[0]


class TestParseContentBlocksWarnings:
    """Tests for _parse_content_blocks with warnings."""

    def test_h4_warning_in_content_blocks(self):
        """H4 heading in content blocks should store warning in DnnBlock."""
        lines = ["#### This is H4"]
        blocks = _parse_content_blocks(lines, [], 1)
        assert len(blocks) == 1
        assert blocks[0].block_type == "paragraph"
        assert blocks[0].content == "#### This is H4"
        assert len(blocks[0].warnings) == 1
        assert "Notion only supports h1-h3" in blocks[0].warnings[0]

    def test_double_delimiter_warning_in_content_blocks(self):
        """Double delimiter should store warning in DnnBlock."""
        lines = ["### ## text"]
        blocks = _parse_content_blocks(lines, [], 1)
        assert len(blocks) == 1
        assert blocks[0].block_type == "heading_3"
        assert blocks[0].content == "## text"
        assert len(blocks[0].warnings) == 1
        assert "looks like nested delimiters" in blocks[0].warnings[0]

    def test_warnings_in_apply_script(self):
        """Warnings should flow through parse_apply_script."""
        registry = IdRegistry()
        # Register a fake parent
        registry.register("12345678-1234-1234-1234-123456789abc", "Prnt")
        script = """+ parent=Prnt
  #### H4 block
  ### ## double"""
        result = parse_apply_script(script, registry)
        assert len(result.operations) == 1
        op = result.operations[0]
        assert op.command == ApplyCommand.ADD
        assert len(op.content_blocks) == 2
        # First block: H4 warning
        assert op.content_blocks[0].block_type == "paragraph"
        assert len(op.content_blocks[0].warnings) == 1
        assert "Notion only supports h1-h3" in op.content_blocks[0].warnings[0]
        # Second block: double delimiter warning
        assert op.content_blocks[1].block_type == "heading_3"
        assert len(op.content_blocks[1].warnings) == 1
        assert "looks like nested delimiters" in op.content_blocks[1].warnings[0]


class TestParseDnn:
    """Tests for parse_dnn function."""

    def test_empty_document(self):
        result = parse_dnn("")
        assert result.header.version == 1
        assert result.blocks == []
        assert result.errors == []

    def test_header_only(self):
        dnn = """@dnn 1
@page abc123
@title My Page"""
        result = parse_dnn(dnn)
        assert result.header.version == 1
        assert result.header.page_id == "abc123"
        assert result.header.title == "My Page"
        assert result.blocks == []

    def test_simple_blocks(self):
        dnn = """A1b2 # Heading
C3d4 Paragraph text"""
        result = parse_dnn(dnn)
        assert len(result.blocks) == 2
        assert result.blocks[0].short_id == "A1b2"
        assert result.blocks[0].block_type == "heading_1"
        assert result.blocks[0].content == "Heading"
        assert result.blocks[1].short_id == "C3d4"
        assert result.blocks[1].block_type == "paragraph"
        assert result.blocks[1].content == "Paragraph text"

    def test_nested_blocks(self):
        dnn = """A1b2 > Toggle
C3d4   Child item
E5f6     Grandchild"""
        result = parse_dnn(dnn)
        assert len(result.blocks) == 3
        assert result.blocks[0].level == 0
        assert result.blocks[1].level == 1
        assert result.blocks[2].level == 2

    def test_full_document(self):
        dnn = """@dnn 1
@page 9a8b7c6d
@title Project Tasks

A1b2 # Phase 1
C3d4 > Research
E5f6   [x] Read docs
G7h8   [ ] Write summary"""
        result = parse_dnn(dnn)
        assert result.header.title == "Project Tasks"
        assert len(result.blocks) == 4
        assert result.blocks[0].block_type == "heading_1"
        assert result.blocks[1].block_type == "toggle"
        assert result.blocks[2].block_type == "to_do"
        assert result.blocks[2].checked is True
        assert result.blocks[3].block_type == "to_do"
        assert result.blocks[3].checked is False

    def test_code_block(self):
        dnn = """A1b2 ```python
def greet(name):
    return f"Hello, {name}!"
```
C3d4 After code"""
        result = parse_dnn(dnn)
        assert len(result.blocks) == 2
        assert result.blocks[0].block_type == "code"
        assert result.blocks[0].language == "python"
        assert len(result.blocks[0].raw_lines) == 2
        assert "def greet" in result.blocks[0].raw_lines[0]
        assert result.blocks[1].block_type == "paragraph"

    def test_database_header(self):
        dnn = """@dnn 1
@db 8x7y6z5w
@ds 7w6v5u4t
@title Tasks
@cols A1b2:Name(title) C3d4:Status(select)"""
        result = parse_dnn(dnn)
        assert result.header.db_id == "8x7y6z5w"
        assert result.header.ds_id == "7w6v5u4t"
        assert len(result.header.columns) == 2
        assert result.header.columns[0]["id"] == "A1b2"
        assert result.header.columns[0]["name"] == "Name"
        assert result.header.columns[0]["type"] == "title"

    def test_error_missing_id_space(self):
        dnn = """A1b2# Heading"""
        result = parse_dnn(dnn)
        assert len(result.errors) == 1
        assert result.errors[0].code == "MISSING_ID_SPACE"
        assert result.errors[0].autofix is not None
        assert result.errors[0].autofix["patched"] == "A1b2 # Heading"

    def test_error_indent_not_multiple_of_2(self):
        dnn = """A1b2 > Toggle
C3d4    Child with 3 spaces"""
        result = parse_dnn(dnn)
        # Should still parse but with error
        assert len(result.errors) == 1
        assert result.errors[0].code == "INDENT_ERROR"

    def test_error_unterminated_code_block(self):
        dnn = """A1b2 ```python
def incomplete():
    pass"""
        result = parse_dnn(dnn)
        assert len(result.errors) == 1
        assert result.errors[0].code == "CODE_BLOCK_UNTERMINATED"
        # Code block should still be added
        assert len(result.blocks) == 1
        assert result.blocks[0].block_type == "code"

    def test_divider(self):
        dnn = """A1b2 Paragraph
C3d4 ---
E5f6 After divider"""
        result = parse_dnn(dnn)
        assert len(result.blocks) == 3
        assert result.blocks[1].block_type == "divider"

    def test_empty_lines_ignored(self):
        dnn = """A1b2 First

C3d4 Second"""
        result = parse_dnn(dnn)
        assert len(result.blocks) == 2


class TestParseBlockMarkerPrecedence:
    """Test that marker precedence is correct."""

    def test_toggle_heading_before_toggle(self):
        # >## should be toggle heading, not toggle with "## text"
        btype, content, attrs, _ = parse_block_type(">## Toggle Heading 2")
        assert btype == "heading_2"
        assert attrs.get("is_toggle") is True

    def test_regular_heading_after_toggle_heading(self):
        # ## should be heading_2, not caught by toggle heading patterns
        btype, content, attrs, _ = parse_block_type("## Regular Heading")
        assert btype == "heading_2"
        assert attrs.get("is_toggle") is not True

    def test_divider_exact_match(self):
        # --- with extra text should be paragraph, not divider
        btype, content, attrs, _ = parse_block_type("--- extra")
        assert btype == "paragraph"  # Not divider because it has extra text


# =============================================================================
# Inline Formatting Tests
# =============================================================================


class TestParseInlineFormatting:
    """Tests for parse_inline_formatting function."""

    def test_plain_text(self):
        spans = parse_inline_formatting("Hello world")
        assert len(spans) == 1
        assert spans[0].text == "Hello world"
        assert spans[0].bold is False

    def test_bold(self):
        spans = parse_inline_formatting("**bold text**")
        assert len(spans) == 1
        assert spans[0].text == "bold text"
        assert spans[0].bold is True

    def test_italic(self):
        spans = parse_inline_formatting("*italic text*")
        assert len(spans) == 1
        assert spans[0].text == "italic text"
        assert spans[0].italic is True

    def test_strikethrough(self):
        spans = parse_inline_formatting("~~struck~~")
        assert len(spans) == 1
        assert spans[0].text == "struck"
        assert spans[0].strikethrough is True

    def test_code(self):
        spans = parse_inline_formatting("`code`")
        assert len(spans) == 1
        assert spans[0].text == "code"
        assert spans[0].code is True

    def test_underline(self):
        spans = parse_inline_formatting(":u[underlined]")
        assert len(spans) == 1
        assert spans[0].text == "underlined"
        assert spans[0].underline is True

    def test_link(self):
        spans = parse_inline_formatting("[click here](https://example.com)")
        assert len(spans) == 1
        assert spans[0].text == "click here"
        assert spans[0].link == "https://example.com"

    def test_link_with_page_ref(self):
        """Test that [text](p:shortID) creates a link with custom text."""
        spans = parse_inline_formatting("[my page](p:A1b2)")
        assert len(spans) == 1
        assert spans[0].span_type == "text"
        assert spans[0].text == "my page"
        assert spans[0].link == "p:A1b2"  # Stored as p:ref, resolved later

    def test_page_mention_syntax(self):
        """Test that @p:shortID creates a page @mention."""
        spans = parse_inline_formatting("See @p:A1b2 for details")
        assert len(spans) == 3
        assert spans[0].text == "See "
        assert spans[1].span_type == "mention_page"
        assert spans[1].page_id == "A1b2"
        assert spans[2].text == " for details"

    def test_color_red(self):
        spans = parse_inline_formatting(":red[warning]")
        assert len(spans) == 1
        assert spans[0].text == "warning"
        assert spans[0].color == "red"

    def test_color_background(self):
        spans = parse_inline_formatting(":blue-background[info]")
        assert len(spans) == 1
        assert spans[0].text == "info"
        assert spans[0].color == "blue_background"

    def test_equation(self):
        spans = parse_inline_formatting("$E=mc^2$")
        assert len(spans) == 1
        assert spans[0].span_type == "equation"
        assert spans[0].expression == "E=mc^2"

    def test_user_mention(self):
        spans = parse_inline_formatting("@user:12345678-1234-1234-1234-123456789abc")
        assert len(spans) == 1
        assert spans[0].span_type == "mention_user"
        assert spans[0].user_id == "12345678-1234-1234-1234-123456789abc"

    def test_date_mention(self):
        spans = parse_inline_formatting("@date:2024-01-15")
        assert len(spans) == 1
        assert spans[0].span_type == "mention_date"
        assert spans[0].date == "2024-01-15"

    def test_date_range_mention(self):
        spans = parse_inline_formatting("@date:2024-01-15→2024-01-20")
        assert len(spans) == 1
        assert spans[0].span_type == "mention_date"
        assert spans[0].date == "2024-01-15"
        assert spans[0].end_date == "2024-01-20"

    def test_date_range_normalization_unicode_arrow(self):
        """Two separate @date mentions with → arrow normalized to single range."""
        spans = parse_inline_formatting("@date:2024-01-15 → @date:2024-01-20")
        assert len(spans) == 1
        assert spans[0].span_type == "mention_date"
        assert spans[0].date == "2024-01-15"
        assert spans[0].end_date == "2024-01-20"

    def test_date_range_normalization_ascii_arrow(self):
        """Two separate @date mentions with -> arrow normalized to single range."""
        spans = parse_inline_formatting("@date:2024-01-15 -> @date:2024-01-20")
        assert len(spans) == 1
        assert spans[0].span_type == "mention_date"
        assert spans[0].date == "2024-01-15"
        assert spans[0].end_date == "2024-01-20"

    def test_date_range_normalization_no_spaces(self):
        """Two separate @date mentions without spaces normalized to single range."""
        spans = parse_inline_formatting("@date:2024-01-15→@date:2024-01-20")
        assert len(spans) == 1
        assert spans[0].span_type == "mention_date"
        assert spans[0].date == "2024-01-15"
        assert spans[0].end_date == "2024-01-20"

    def test_escaped_asterisk(self):
        spans = parse_inline_formatting("\\*not bold\\*")
        assert len(spans) == 1
        assert spans[0].text == "*not bold*"

    def test_escaped_dollar(self):
        spans = parse_inline_formatting("Price: \\$10")
        assert len(spans) == 1
        assert spans[0].text == "Price: $10"

    def test_mixed_formatting(self):
        spans = parse_inline_formatting("Normal **bold** and *italic*")
        assert len(spans) == 4
        assert spans[0].text == "Normal "
        assert spans[1].text == "bold"
        assert spans[1].bold is True
        assert spans[2].text == " and "
        assert spans[3].text == "italic"
        assert spans[3].italic is True

    def test_nested_color_and_bold(self):
        spans = parse_inline_formatting(":red[**important**]")
        assert len(spans) == 1
        assert spans[0].text == "important"
        assert spans[0].color == "red"
        assert spans[0].bold is True

    def test_link_with_bold_text(self):
        spans = parse_inline_formatting("[**bold link**](https://example.com)")
        assert len(spans) == 1
        assert spans[0].text == "bold link"
        assert spans[0].bold is True
        assert spans[0].link == "https://example.com"

    def test_complex_example_from_spec(self):
        # From design doc: **Bold**, *italic*, ~~struck~~, :u[underline], `code`.
        spans = parse_inline_formatting(
            "**Bold**, *italic*, ~~struck~~, :u[underline], `code`."
        )
        # Should have: bold, comma, italic, comma, struck, comma, underline, comma, code, period
        bold_spans = [s for s in spans if s.bold]
        italic_spans = [s for s in spans if s.italic]
        strike_spans = [s for s in spans if s.strikethrough]
        underline_spans = [s for s in spans if s.underline]
        code_spans = [s for s in spans if s.code]

        assert len(bold_spans) == 1
        assert bold_spans[0].text == "Bold"
        assert len(italic_spans) == 1
        assert italic_spans[0].text == "italic"
        assert len(strike_spans) == 1
        assert strike_spans[0].text == "struck"
        assert len(underline_spans) == 1
        assert underline_spans[0].text == "underline"
        assert len(code_spans) == 1
        assert code_spans[0].text == "code"

    def test_empty_string(self):
        spans = parse_inline_formatting("")
        assert spans == []

    def test_nested_colors_preserved(self):
        """Inner color should NOT be overwritten by outer color."""
        spans = parse_inline_formatting(":red[:green[inner] outer]")
        # Find the spans with actual text
        text_spans = [s for s in spans if s.text.strip()]
        assert len(text_spans) == 2
        # Inner should be green, outer should be red
        inner = next(s for s in text_spans if "inner" in s.text)
        outer = next(s for s in text_spans if "outer" in s.text)
        assert inner.color == "green"
        assert outer.color == "red"

    def test_escaped_bracket_literal(self):
        """Escaped brackets should become literal characters."""
        spans = parse_inline_formatting(r"text \[ bracket \]")
        # Should merge into single span with literal brackets
        full_text = "".join(s.text for s in spans)
        assert "[" in full_text
        assert "]" in full_text

    def test_escaped_asterisk_in_text(self):
        """Escaped asterisks should not trigger bold/italic."""
        spans = parse_inline_formatting(r"price is \*10\* dollars")
        full_text = "".join(s.text for s in spans)
        assert "*10*" in full_text
        # Should not be bold
        assert not any(s.bold for s in spans)

    def test_deeply_nested_formatting(self):
        """Multiple levels of nesting should all work."""
        spans = parse_inline_formatting(":red[**:blue[nested]**]")
        # Find the nested text
        nested_span = next(s for s in spans if "nested" in s.text)
        assert nested_span.color == "blue"  # Inner color preserved
        assert nested_span.bold is True  # Bold applied


class TestRichTextSpansToNotion:
    """Tests for rich_text_spans_to_notion function."""

    def test_plain_text(self):
        spans = [RichTextSpan(text="Hello")]
        result = rich_text_spans_to_notion(spans)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"]["content"] == "Hello"
        assert "annotations" not in result[0]

    def test_bold_text(self):
        spans = [RichTextSpan(text="Bold", bold=True)]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["annotations"]["bold"] is True

    def test_link(self):
        spans = [RichTextSpan(text="Click", link="https://example.com")]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["text"]["link"]["url"] == "https://example.com"

    def test_color(self):
        spans = [RichTextSpan(text="Red", color="red")]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["annotations"]["color"] == "red"

    def test_equation(self):
        spans = [RichTextSpan(text="", span_type="equation", expression="x^2")]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["type"] == "equation"
        assert result[0]["equation"]["expression"] == "x^2"

    def test_user_mention(self):
        spans = [RichTextSpan(
            text="",
            span_type="mention_user",
            user_id="abc-123"
        )]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["type"] == "mention"
        assert result[0]["mention"]["type"] == "user"
        assert result[0]["mention"]["user"]["id"] == "abc-123"

    def test_date_mention(self):
        spans = [RichTextSpan(
            text="",
            span_type="mention_date",
            date="2024-01-15"
        )]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["type"] == "mention"
        assert result[0]["mention"]["type"] == "date"
        assert result[0]["mention"]["date"]["start"] == "2024-01-15"

    def test_date_range_mention(self):
        spans = [RichTextSpan(
            text="",
            span_type="mention_date",
            date="2024-01-15",
            end_date="2024-01-20"
        )]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["type"] == "mention"
        assert result[0]["mention"]["type"] == "date"
        assert result[0]["mention"]["date"]["start"] == "2024-01-15"
        assert result[0]["mention"]["date"]["end"] == "2024-01-20"

    def test_multiple_annotations(self):
        spans = [RichTextSpan(text="Complex", bold=True, italic=True, color="blue")]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["annotations"]["bold"] is True
        assert result[0]["annotations"]["italic"] is True
        assert result[0]["annotations"]["color"] == "blue"

    def test_page_mention_with_registry(self):
        """Page mention with registry should resolve short ID to UUID."""
        registry = IdRegistry()
        full_uuid = "12345678-1234-1234-1234-123456789abc"
        short_id = registry.register(full_uuid)

        spans = [RichTextSpan(
            text="My Page",
            span_type="mention_page",
            page_id=short_id
        )]
        result = rich_text_spans_to_notion(spans, registry=registry)
        assert result[0]["type"] == "mention"
        assert result[0]["mention"]["type"] == "page"
        assert result[0]["mention"]["page"]["id"] == full_uuid

    def test_page_mention_without_registry(self):
        """Page mention without registry should use page_id directly."""
        spans = [RichTextSpan(
            text="My Page",
            span_type="mention_page",
            page_id="12345678-1234-1234-1234-123456789abc"
        )]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["type"] == "mention"
        assert result[0]["mention"]["type"] == "page"
        assert result[0]["mention"]["page"]["id"] == "12345678-1234-1234-1234-123456789abc"

    def test_page_mention_unknown_short_id_fallback(self):
        """Unknown short ID without registry falls back to plain text."""
        spans = [RichTextSpan(
            text="My Page",
            span_type="mention_page",
            page_id=""  # Empty ID
        )]
        result = rich_text_spans_to_notion(spans)
        # Should fall back to plain text since page_id is empty
        assert result[0]["type"] == "text"
        assert result[0]["text"]["content"] == "My Page"

    def test_page_link_resolves_to_notion_url(self):
        """[text](p:shortID) should create a link with Notion URL."""
        registry = IdRegistry()
        full_uuid = "12345678-1234-1234-1234-123456789abc"
        short_id = registry.register(full_uuid)

        spans = [RichTextSpan(text="my page", link=f"p:{short_id}")]
        result = rich_text_spans_to_notion(spans, registry=registry)
        assert result[0]["type"] == "text"
        assert result[0]["text"]["content"] == "my page"
        # UUID dashes removed for URL format
        assert result[0]["text"]["link"]["url"] == "https://notion.so/12345678123412341234123456789abc"

    def test_page_link_without_registry_uses_ref_directly(self):
        """[text](p:ref) without registry uses ref directly in URL."""
        spans = [RichTextSpan(text="my page", link="p:some-uuid-here")]
        result = rich_text_spans_to_notion(spans)
        assert result[0]["type"] == "text"
        assert result[0]["text"]["link"]["url"] == "https://notion.so/someuuidhere"


class TestDetectConflicts:
    """Tests for _detect_conflicts function - parallel execution validation."""

    def test_no_conflicts_independent_ops(self):
        """Independent operations should not conflict."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.UPDATE, target="A1b2"),
            ApplyOp(line_num=2, command=ApplyCommand.UPDATE, target="C3d4"),
            ApplyOp(line_num=3, command=ApplyCommand.DELETE, targets=["E5f6"]),
        ]
        errors = _detect_conflicts(ops)
        assert errors == []

    def test_duplicate_target_detected(self):
        """Same block targeted by multiple ops should error."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.UPDATE, target="A1b2"),
            ApplyOp(line_num=2, command=ApplyCommand.UPDATE, target="A1b2"),
        ]
        errors = _detect_conflicts(ops)
        assert len(errors) == 1
        assert "A1b2" in errors[0]
        assert "lines [1, 2]" in errors[0]

    def test_after_references_moved_block(self):
        """Using after=X where X is being moved should error."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.MOVE,
                    source="A1b2", dest_parent="P1p2"),
            ApplyOp(line_num=2, command=ApplyCommand.ADD,
                    parent="P1p2", after="A1b2", content_blocks=[]),
        ]
        errors = _detect_conflicts(ops)
        assert len(errors) == 1
        assert "Line 2" in errors[0]
        assert "after=A1b2" in errors[0]
        assert "moved on line 1" in errors[0]
        assert "ID will change" in errors[0]

    def test_after_references_deleted_block(self):
        """Using after=X where X is being deleted should error."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.DELETE, targets=["A1b2"]),
            ApplyOp(line_num=2, command=ApplyCommand.ADD,
                    parent="P1p2", after="A1b2", content_blocks=[]),
        ]
        errors = _detect_conflicts(ops)
        assert len(errors) == 1
        assert "Line 2" in errors[0]
        assert "after=A1b2" in errors[0]
        assert "deleted on line 1" in errors[0]

    def test_move_dest_after_references_moved_block(self):
        """Move with dest_after=X where X is being moved should error."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.MOVE,
                    source="A1b2", dest_parent="P1p2"),
            ApplyOp(line_num=2, command=ApplyCommand.MOVE,
                    source="B2c3", dest_parent="P1p2", dest_after="A1b2"),
        ]
        errors = _detect_conflicts(ops)
        assert len(errors) == 1
        assert "after=A1b2" in errors[0]
        assert "ID will change" in errors[0]

    def test_same_parent_is_allowed(self):
        """Multiple ops adding to same parent should not conflict."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.ADD,
                    parent="P1p2", content_blocks=[]),
            ApplyOp(line_num=2, command=ApplyCommand.ADD,
                    parent="P1p2", content_blocks=[]),
        ]
        errors = _detect_conflicts(ops)
        assert errors == []

    def test_page_move_does_not_invalidate_id(self):
        """Page moves don't change IDs, so after refs should be OK."""
        ops = [
            ApplyOp(line_num=1, command=ApplyCommand.MOVE_PAGE,
                    source="A1b2", dest_parent="P1p2"),
            ApplyOp(line_num=2, command=ApplyCommand.ADD,
                    parent="P1p2", after="A1b2", content_blocks=[]),
        ]
        errors = _detect_conflicts(ops)
        assert errors == []


# =============================================================================
# Apply Script Error Propagation Tests
# =============================================================================


class TestApplyScriptErrorPropagation:
    """Tests for error handling in apply script parsing."""

    def test_empty_after_parameter_add(self):
        """after= with empty value gives helpful error for ADD command."""
        script = "+ parent=ABC1 after="
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 1
        assert result.errors[0].code == "EMPTY_AFTER_PARAM"
        assert "Notion API does not support" in result.errors[0].message
        assert "inserting at the beginning" in result.errors[0].message
        assert result.errors[0].line == 0

    def test_empty_after_with_whitespace_add(self):
        """after= followed by whitespace only gives same error for ADD."""
        script = "+ parent=ABC1 after=   "
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 1
        assert result.errors[0].code == "EMPTY_AFTER_PARAM"

    def test_empty_after_parameter_move(self):
        """after= with empty value gives helpful error for MOVE command."""
        script = "m SRC1 -> parent=DST1 after="
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 1
        assert result.errors[0].code == "EMPTY_AFTER_PARAM"
        assert "Notion API does not support" in result.errors[0].message
        assert "moving to the beginning" in result.errors[0].message
        assert result.errors[0].line == 0

    def test_empty_after_with_whitespace_move(self):
        """after= followed by whitespace only gives same error for MOVE."""
        script = "m SRC1 -> parent=DST1 after=   "
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 1
        assert result.errors[0].code == "EMPTY_AFTER_PARAM"

    def test_error_line_attribute_accessible(self):
        """DnnParseError.line is accessible (regression test for .line_num bug)."""
        err = DnnParseError(
            code="TEST",
            message="test error",
            line=5,
            excerpt="5|test"
        )
        # This should not raise AttributeError
        assert err.line == 5
        # Verify line_num doesn't exist (to prevent regression)
        assert not hasattr(err, 'line_num')

    def test_valid_after_parameter_still_works(self):
        """Valid after=BLOCK_ID still parses correctly."""
        script = "+ parent=ABC1 after=XYZ9\n  - Some content"
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].command == ApplyCommand.ADD
        assert result.operations[0].after == "XYZ9"

    def test_omitted_after_still_works(self):
        """Omitting after= entirely still works (appends to end)."""
        script = "+ parent=ABC1\n  - Some content"
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].command == ApplyCommand.ADD
        assert result.operations[0].after is None


# =============================================================================
# Property Value Builder Tests
# =============================================================================


class TestBuildPropertyValue:
    """Tests for _build_property_value helper function."""

    def test_title_property(self):
        result = _build_property_value("title", "My Title")
        assert result == {"title": [{"type": "text", "text": {"content": "My Title"}}]}

    def test_rich_text_property(self):
        result = _build_property_value("rich_text", "Some text")
        assert result == {"rich_text": [{"type": "text", "text": {"content": "Some text"}}]}

    def test_select_property(self):
        result = _build_property_value("select", "Option A")
        assert result == {"select": {"name": "Option A"}}

    def test_multi_select_property(self):
        result = _build_property_value("multi_select", "tag1, tag2, tag3")
        assert result == {"multi_select": [
            {"name": "tag1"},
            {"name": "tag2"},
            {"name": "tag3"}
        ]}

    def test_multi_select_trims_whitespace(self):
        result = _build_property_value("multi_select", "  a  ,  b  ")
        assert result == {"multi_select": [{"name": "a"}, {"name": "b"}]}

    def test_checkbox_true_values(self):
        for val in ("true", "1", "yes", "x", "TRUE", "Yes", "X"):
            result = _build_property_value("checkbox", val)
            assert result == {"checkbox": True}, f"Failed for: {val}"

    def test_checkbox_false_values(self):
        for val in ("false", "0", "no", "", "anything"):
            result = _build_property_value("checkbox", val)
            assert result == {"checkbox": False}, f"Failed for: {val}"

    def test_number_property_integer(self):
        result = _build_property_value("number", "42")
        assert result == {"number": 42.0}

    def test_number_property_float(self):
        result = _build_property_value("number", "3.14")
        assert result == {"number": 3.14}

    def test_number_property_invalid(self):
        result = _build_property_value("number", "not a number")
        assert result is None

    def test_date_property_single(self):
        result = _build_property_value("date", "2025-01-15")
        assert result == {"date": {"start": "2025-01-15"}}

    def test_date_property_range(self):
        result = _build_property_value("date", "2025-01-15→2025-01-20")
        assert result == {"date": {"start": "2025-01-15", "end": "2025-01-20"}}

    def test_date_property_range_trims_whitespace(self):
        result = _build_property_value("date", "2025-01-15 → 2025-01-20")
        assert result == {"date": {"start": "2025-01-15", "end": "2025-01-20"}}

    def test_url_property(self):
        result = _build_property_value("url", "https://example.com")
        assert result == {"url": "https://example.com"}

    def test_email_property(self):
        result = _build_property_value("email", "user@example.com")
        assert result == {"email": "user@example.com"}

    def test_phone_number_property(self):
        result = _build_property_value("phone_number", "+1-555-0123")
        assert result == {"phone_number": "+1-555-0123"}

    def test_unknown_property_type(self):
        result = _build_property_value("unknown_type", "value")
        assert result is None


class TestBuildRowProperties:
    """Tests for _build_row_properties helper function."""

    def test_builds_multiple_properties(self):
        row_values = {
            "Name": "Task 1",
            "Status": "Done"
        }
        db_props = {
            "Name": {"type": "title"},
            "Status": {"type": "select"}
        }
        result = _build_row_properties(row_values, db_props)
        assert result == {
            "Name": {"title": [{"type": "text", "text": {"content": "Task 1"}}]},
            "Status": {"select": {"name": "Done"}}
        }

    def test_case_insensitive_property_lookup(self):
        row_values = {"name": "Task"}  # lowercase
        db_props = {"Name": {"type": "title"}}  # Capitalized
        result = _build_row_properties(row_values, db_props)
        assert "Name" in result  # Uses canonical name from db_props

    def test_trailing_whitespace_property_lookup(self):
        """Notion property names may have trailing spaces; DNN input is stripped."""
        row_values = {"Paid on": "2026-01-19"}  # No trailing space (stripped by parser)
        db_props = {"Paid on ": {"type": "date"}}  # Trailing space in Notion
        result = _build_row_properties(row_values, db_props)
        assert "Paid on " in result  # Uses canonical name from db_props (with space)

    def test_skips_unknown_properties(self):
        row_values = {"Name": "Task", "Unknown": "Value"}
        db_props = {"Name": {"type": "title"}}
        result = _build_row_properties(row_values, db_props)
        assert "Name" in result
        assert "Unknown" not in result

    def test_skips_invalid_number_conversion(self):
        row_values = {"Count": "not a number"}
        db_props = {"Count": {"type": "number"}}
        result = _build_row_properties(row_values, db_props)
        assert "Count" not in result  # Invalid conversion returns None

    def test_empty_row_values(self):
        result = _build_row_properties({}, {"Name": {"type": "title"}})
        assert result == {}

    def test_empty_db_properties(self):
        result = _build_row_properties({"Name": "Value"}, {})
        assert result == {}


# =============================================================================
# @mentions in Titles Tests
# =============================================================================


class TestTextToRichText:
    """Tests for _text_to_rich_text helper function."""

    def test_plain_text(self):
        result = _text_to_rich_text("Hello")
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"]["content"] == "Hello"

    def test_date_mention(self):
        result = _text_to_rich_text("Due @date:2025-02-01")
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"]["content"] == "Due "
        assert result[1]["type"] == "mention"
        assert result[1]["mention"]["type"] == "date"
        assert result[1]["mention"]["date"]["start"] == "2025-02-01"

    def test_page_mention_with_registry(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        result = _text_to_rich_text(f"See @p:{sid}", registry)
        mentions = [r for r in result if r.get("type") == "mention"]
        assert len(mentions) == 1
        assert mentions[0]["mention"]["page"]["id"] == uuid

    def test_empty_string(self):
        assert _text_to_rich_text("") == []

    def test_none_returns_empty(self):
        # Empty string should return empty list
        assert _text_to_rich_text("") == []

    def test_user_mention(self):
        result = _text_to_rich_text("@user:12345678-1234-1234-1234-123456789abc")
        mentions = [r for r in result if r.get("type") == "mention"]
        assert len(mentions) == 1
        assert mentions[0]["mention"]["type"] == "user"
        assert mentions[0]["mention"]["user"]["id"] == "12345678-1234-1234-1234-123456789abc"

    def test_date_range(self):
        result = _text_to_rich_text("@date:2025-01-15→2025-01-20")
        mentions = [r for r in result if r.get("type") == "mention"]
        assert len(mentions) == 1
        assert mentions[0]["mention"]["date"]["start"] == "2025-01-15"
        assert mentions[0]["mention"]["date"]["end"] == "2025-01-20"

    def test_bold_in_title(self):
        result = _text_to_rich_text("**Important** Meeting")
        bold_spans = [r for r in result if r.get("annotations", {}).get("bold")]
        assert len(bold_spans) == 1
        assert bold_spans[0]["text"]["content"] == "Important"


class TestBuildPropertyValueWithRegistry:
    """Tests for _build_property_value with registry support."""

    def test_title_with_date_mention(self):
        result = _build_property_value("title", "Task due @date:2025-02-01")
        rich_text = result["title"]
        assert len(rich_text) == 2
        assert rich_text[1]["type"] == "mention"
        assert rich_text[1]["mention"]["type"] == "date"

    def test_rich_text_with_page_mention(self):
        registry = IdRegistry()
        uuid = "12345678-1234-1234-1234-123456789abc"
        sid = registry.register(uuid)
        result = _build_property_value(
            "rich_text",
            f"See @p:{sid}",
            registry=registry
        )
        mentions = [r for r in result["rich_text"]
                   if r.get("type") == "mention"]
        assert len(mentions) == 1
        assert mentions[0]["mention"]["page"]["id"] == uuid

    def test_title_plain_text_still_works(self):
        """Plain text without @mentions should still work."""
        result = _build_property_value("title", "Simple Title")
        # Verify structure matches rich text format
        assert "title" in result
        assert len(result["title"]) == 1
        assert result["title"][0]["type"] == "text"
        assert result["title"][0]["text"]["content"] == "Simple Title"

    def test_rich_text_with_formatting(self):
        result = _build_property_value("rich_text", "**bold** text")
        rich_text = result["rich_text"]
        bold_spans = [r for r in rich_text if r.get("annotations", {}).get("bold")]
        assert len(bold_spans) == 1


class TestAddRowIcon:
    """Tests for +row icon= parameter parsing."""

    def test_add_row_with_emoji_icon(self):
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId icon=🔥\n  Name=Task"
        result = parse_apply_script(script, registry)
        assert len(result.operations) == 1
        assert result.operations[0].icon == "🔥"

    def test_add_row_with_url_icon(self):
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId icon=https://example.com/icon.png\n  Name=Task"
        result = parse_apply_script(script, registry)
        assert result.operations[0].icon == "https://example.com/icon.png"

    def test_add_row_without_icon(self):
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId\n  Name=Task"
        result = parse_apply_script(script, registry)
        assert result.operations[0].icon is None

    def test_add_row_icon_before_values(self):
        """Icon parameter must come on same line as +row."""
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId icon=📌\n  Name=Test Task\tStatus=Done"
        result = parse_apply_script(script, registry)
        assert len(result.operations) == 1
        assert result.operations[0].command == ApplyCommand.ADD_ROW
        assert result.operations[0].icon == "📌"
        assert result.operations[0].database == "DbId"


class TestTabIndentation:
    """Tests for tab indentation in +row and urow commands."""

    def test_add_row_with_tab_indented_values(self):
        """Values line can use tab instead of spaces."""
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId\n\tName=Task\tStatus=Done"
        result = parse_apply_script(script, registry)
        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].command == ApplyCommand.ADD_ROW
        assert result.operations[0].row_values == {"Name": "Task", "Status": "Done"}

    def test_add_row_with_space_indented_values(self):
        """Values line with two-space indent still works."""
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId\n  Name=Task\tStatus=Done"
        result = parse_apply_script(script, registry)
        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].row_values == {"Name": "Task", "Status": "Done"}

    def test_urow_with_tab_indented_values(self):
        """urow values line can use tab instead of spaces."""
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "RowId")
        script = "urow RowId\n\tStatus=Done\tPriority=High"
        result = parse_apply_script(script, registry)
        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].command == ApplyCommand.UPDATE_ROW
        assert result.operations[0].row_values == {"Status": "Done", "Priority": "High"}

    def test_add_row_with_icon_and_tab_values(self):
        """Icon parameter works with tab-indented values."""
        registry = IdRegistry()
        registry.register("12345678-1234-1234-1234-123456789abc", "DbId")
        script = "+row db=DbId icon=💪\n\tName=Test\tDate=2026-02-01"
        result = parse_apply_script(script, registry)
        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].icon == "💪"
        assert result.operations[0].row_values == {"Name": "Test", "Date": "2026-02-01"}
