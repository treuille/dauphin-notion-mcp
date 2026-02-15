"""Tests for notion_mcp module - ID system and DNN parser."""

import asyncio

import pytest
from notion_mcp import (
    ApplyCommand,
    ApplyOp,
    BASE62_ALPHABET,
    DnnBlock,
    DnnHeader,
    DnnParseError,
    DnnParseResult,
    FILE_BEARING_BLOCK_TYPES,
    FilterAtom,
    FilterCompound,
    FilterParseError,
    IdRegistry,
    ParseState,
    RichTextSpan,
    SHORT_ID_PATTERN,
    UNMOVABLE_BLOCK_TYPES,
    UUID_PATTERN,
    _build_property_value,
    _build_row_properties,
    _check_movability,
    _coerce_value,
    _detect_conflicts,
    _identify_move_chains,
    _fetch_database_schema,
    _find_property,
    _notion_request,
    _parse_content_blocks,
    _read_impl,
    _reupload_notion_file,
    _text_to_rich_text,
    _tokenize_filter,
    calculate_indent_level,
    compile_filter,
    extract_uuid_from_url,
    fetch_database_async,
    fetch_data_source_async,
    generate_short_id,
    normalize_uuid,
    notion_rich_text_to_dnn,
    parse_apply_script,
    parse_block_type,
    parse_columns_dsl,
    parse_dnn,
    parse_filter_dsl,
    parse_inline_formatting,
    parse_sort_dsl,
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


# =============================================================================
# Filter DSL Tests
# =============================================================================

# Helper: minimal schema for testing
def _make_schema(**kwargs):
    """Build a minimal schema dict. kwargs: name=type, e.g. Status="select"."""
    properties = {}
    for name, ptype in kwargs.items():
        # Replace underscores with spaces for property names like "Last_Contact"
        display_name = name.replace("_", " ") if "_" in name else name
        properties[display_name] = {"type": ptype}
    return {"properties": properties}


SAMPLE_SCHEMA = _make_schema(
    Title="title", Status="select", Due="date", Done="checkbox",
    Priority="number", Tags="multi_select", Notes="rich_text",
    Last_Contact="date", Last_Platform="select",
)


class TestFilterTokenizer:
    """Tests for _tokenize_filter function."""

    def test_simple_equality(self):
        tokens = _tokenize_filter("Status = Done")
        types = [t[0] for t in tokens]
        assert types == ["WORD", "OP", "WORD", "EOF"]

    def test_quoted_value(self):
        tokens = _tokenize_filter('Status = "In Progress"')
        assert tokens[2] == ("QUOTED", "In Progress")

    def test_quoted_property(self):
        tokens = _tokenize_filter('"Last Contact" >= 2026-02-01')
        assert tokens[0] == ("QUOTED", "Last Contact")
        assert tokens[1] == ("OP", ">=")

    def test_and_or(self):
        tokens = _tokenize_filter("A = 1 & B = 2 | C = 3")
        types = [t[0] for t in tokens]
        assert types == ["WORD", "OP", "WORD", "AND",
                         "WORD", "OP", "WORD", "OR",
                         "WORD", "OP", "WORD", "EOF"]

    def test_parentheses(self):
        tokens = _tokenize_filter("(A = 1 | B = 2) & C = 3")
        types = [t[0] for t in tokens]
        assert types == ["LPAREN", "WORD", "OP", "WORD", "OR",
                         "WORD", "OP", "WORD", "RPAREN", "AND",
                         "WORD", "OP", "WORD", "EOF"]

    def test_unary_operators(self):
        tokens = _tokenize_filter("Title ? & Notes !?")
        ops = [t for t in tokens if t[0] == "OP"]
        assert ops == [("OP", "?"), ("OP", "!?")]

    def test_all_comparison_ops(self):
        for op in ("=", "!=", "~", "!~", "<", ">", "<=", ">="):
            tokens = _tokenize_filter(f"X {op} Y")
            assert tokens[1] == ("OP", op), f"Failed for {op}"

    def test_unterminated_quote_raises(self):
        with pytest.raises(FilterParseError, match="Unterminated"):
            _tokenize_filter('Status = "In Progress')

    def test_escaped_quote_in_value(self):
        tokens = _tokenize_filter(r'Title = "say \"hello\""')
        assert tokens[2] == ("QUOTED", 'say "hello"')

    def test_empty_string(self):
        tokens = _tokenize_filter("")
        assert tokens == [("EOF", "")]


class TestFilterParser:
    """Tests for parse_filter_dsl function."""

    def test_simple_atom(self):
        node = parse_filter_dsl("Status = Done")
        assert isinstance(node, FilterAtom)
        assert node.property == "Status"
        assert node.operator == "="
        assert node.value == "Done"

    def test_quoted_property_and_value(self):
        node = parse_filter_dsl('"Last Contact" = "In Progress"')
        assert isinstance(node, FilterAtom)
        assert node.property == "Last Contact"
        assert node.value == "In Progress"

    def test_and_chain(self):
        node = parse_filter_dsl("A = 1 & B = 2 & C = 3")
        assert isinstance(node, FilterCompound)
        assert node.op == "&"
        assert len(node.children) == 3

    def test_or_chain(self):
        node = parse_filter_dsl("A = 1 | B = 2 | C = 3")
        assert isinstance(node, FilterCompound)
        assert node.op == "|"
        assert len(node.children) == 3

    def test_and_binds_tighter_than_or(self):
        # A = 1 | B = 2 & C = 3  →  A=1 | (B=2 & C=3)
        node = parse_filter_dsl("A = 1 | B = 2 & C = 3")
        assert isinstance(node, FilterCompound)
        assert node.op == "|"
        assert len(node.children) == 2
        assert isinstance(node.children[0], FilterAtom)  # A = 1
        assert isinstance(node.children[1], FilterCompound)  # B=2 & C=3
        assert node.children[1].op == "&"

    def test_parentheses_override_precedence(self):
        # (A = 1 | B = 2) & C = 3
        node = parse_filter_dsl("(A = 1 | B = 2) & C = 3")
        assert isinstance(node, FilterCompound)
        assert node.op == "&"
        assert isinstance(node.children[0], FilterCompound)  # A=1 | B=2
        assert node.children[0].op == "|"

    def test_unary_is_empty(self):
        node = parse_filter_dsl("Title ?")
        assert isinstance(node, FilterAtom)
        assert node.operator == "?"
        assert node.value == ""

    def test_unary_is_not_empty(self):
        node = parse_filter_dsl("Title !?")
        assert isinstance(node, FilterAtom)
        assert node.operator == "!?"

    def test_error_missing_operator(self):
        with pytest.raises(FilterParseError, match="Expected operator"):
            parse_filter_dsl("Status Done")

    def test_error_missing_value(self):
        with pytest.raises(FilterParseError, match="Expected value"):
            parse_filter_dsl("Status = ")

    def test_error_mismatched_paren(self):
        with pytest.raises(FilterParseError):
            parse_filter_dsl("(A = 1 | B = 2")

    def test_error_empty_filter(self):
        with pytest.raises(FilterParseError, match="Expected property"):
            parse_filter_dsl("&")


class TestFilterCompiler:
    """Tests for compile_filter function."""

    def test_text_contains(self):
        node = parse_filter_dsl("Title ~ bug")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Title", "title": {"contains": "bug"}}

    def test_select_equals(self):
        node = parse_filter_dsl("Status = Done")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Status", "select": {"equals": "Done"}}

    def test_date_before(self):
        node = parse_filter_dsl("Due < 2024-01-15")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Due", "date": {"before": "2024-01-15"}}

    def test_date_on_or_after(self):
        node = parse_filter_dsl("Due >= 2024-01-15")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Due", "date": {"on_or_after": "2024-01-15"}}

    def test_number_greater_than(self):
        node = parse_filter_dsl("Priority > 3")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Priority", "number": {"greater_than": 3}}

    def test_number_float(self):
        node = parse_filter_dsl("Priority = 3.5")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Priority", "number": {"equals": 3.5}}

    def test_checkbox_true(self):
        node = parse_filter_dsl("Done = 1")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Done", "checkbox": {"equals": True}}

    def test_checkbox_false(self):
        node = parse_filter_dsl("Done = false")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Done", "checkbox": {"equals": False}}

    def test_multi_select_contains(self):
        node = parse_filter_dsl("Tags ~ urgent")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Tags", "multi_select": {"contains": "urgent"}}

    def test_is_empty(self):
        node = parse_filter_dsl("Notes ?")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Notes", "rich_text": {"is_empty": True}}

    def test_is_not_empty(self):
        node = parse_filter_dsl("Notes !?")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Notes", "rich_text": {"is_not_empty": True}}

    def test_and_logic(self):
        node = parse_filter_dsl("Status = Done & Due < 2024-01-15")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert "and" in result
        assert len(result["and"]) == 2

    def test_or_logic(self):
        node = parse_filter_dsl("Status = Done | Status = Todo")
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert "or" in result
        assert len(result["or"]) == 2

    def test_case_insensitive_property(self):
        node = parse_filter_dsl("status = Done")
        result = compile_filter(node, SAMPLE_SCHEMA)
        # Should resolve to canonical "Status"
        assert result["property"] == "Status"

    def test_quoted_property_with_space(self):
        node = parse_filter_dsl('"Last Contact" >= 2026-02-01')
        result = compile_filter(node, SAMPLE_SCHEMA)
        assert result == {"property": "Last Contact", "date": {"on_or_after": "2026-02-01"}}

    def test_error_unknown_property(self):
        node = parse_filter_dsl("Unknown = foo")
        with pytest.raises(FilterParseError, match="Unknown property"):
            compile_filter(node, SAMPLE_SCHEMA)

    def test_error_unsupported_operator(self):
        # select doesn't support ~
        node = parse_filter_dsl("Status ~ Done")
        with pytest.raises(FilterParseError, match="not valid"):
            compile_filter(node, SAMPLE_SCHEMA)

    def test_error_bad_checkbox_value(self):
        with pytest.raises(FilterParseError, match="Invalid checkbox"):
            _coerce_value("maybe", "checkbox")

    def test_error_bad_number_value(self):
        with pytest.raises(FilterParseError, match="Invalid number"):
            _coerce_value("abc", "number")

    def test_relative_date_days(self):
        from datetime import date, timedelta
        result = _coerce_value("-14d", "date")
        expected = (date.today() - timedelta(days=14)).isoformat()
        assert result == expected

    def test_relative_date_in_filter(self):
        from datetime import date, timedelta
        node = parse_filter_dsl('"Last Contact" >= -7d')
        result = compile_filter(node, SAMPLE_SCHEMA)
        expected_date = (date.today() - timedelta(days=7)).isoformat()
        assert result == {"property": "Last Contact", "date": {"on_or_after": expected_date}}

    def test_relative_date_not_applied_to_text(self):
        result = _coerce_value("-14d", "rich_text")
        assert result == "-14d"

    def test_iso_date_accepted(self):
        assert _coerce_value("2026-01-25", "date") == "2026-01-25"

    def test_iso_datetime_accepted(self):
        assert _coerce_value("2026-01-25T14:30", "date") == "2026-01-25T14:30"

    def test_iso_datetime_with_seconds_accepted(self):
        assert _coerce_value("2026-01-25T14:30:00", "date") == "2026-01-25T14:30:00"

    def test_invalid_date_rejected(self):
        with pytest.raises(FilterParseError, match="Invalid date value"):
            _coerce_value("yesterday", "date")

    def test_garbage_date_rejected(self):
        with pytest.raises(FilterParseError, match="Invalid date value"):
            _coerce_value("not-a-date", "date")

    def test_invalid_date_rejected_for_created_time(self):
        with pytest.raises(FilterParseError, match="Invalid date value"):
            _coerce_value("last week", "created_time")

    def test_invalid_date_rejected_for_last_edited_time(self):
        with pytest.raises(FilterParseError, match="Invalid date value"):
            _coerce_value("abc", "last_edited_time")


class TestSortParser:
    """Tests for parse_sort_dsl function."""

    def test_single_sort_desc(self):
        result = parse_sort_dsl("Due desc", SAMPLE_SCHEMA)
        assert result == [{"property": "Due", "direction": "descending"}]

    def test_single_sort_asc(self):
        result = parse_sort_dsl("Due asc", SAMPLE_SCHEMA)
        assert result == [{"property": "Due", "direction": "ascending"}]

    def test_default_direction(self):
        result = parse_sort_dsl("Due", SAMPLE_SCHEMA)
        assert result == [{"property": "Due", "direction": "ascending"}]

    def test_multiple_sorts(self):
        result = parse_sort_dsl("Due desc, Status asc", SAMPLE_SCHEMA)
        assert len(result) == 2
        assert result[0]["property"] == "Due"
        assert result[0]["direction"] == "descending"
        assert result[1]["property"] == "Status"
        assert result[1]["direction"] == "ascending"

    def test_quoted_property(self):
        result = parse_sort_dsl('"Last Contact" desc', SAMPLE_SCHEMA)
        assert result == [{"property": "Last Contact", "direction": "descending"}]

    def test_case_insensitive(self):
        result = parse_sort_dsl("due desc", SAMPLE_SCHEMA)
        assert result[0]["property"] == "Due"

    def test_empty_string(self):
        result = parse_sort_dsl("", SAMPLE_SCHEMA)
        assert result == []

    def test_error_unknown_property(self):
        with pytest.raises(FilterParseError, match="Unknown property"):
            parse_sort_dsl("Unknown desc", SAMPLE_SCHEMA)

    def test_error_invalid_direction(self):
        with pytest.raises(FilterParseError, match="Invalid sort direction"):
            parse_sort_dsl("Due sideways", SAMPLE_SCHEMA)


class TestColumnsParser:
    """Tests for parse_columns_dsl function."""

    def test_single_column(self):
        result = parse_columns_dsl("Title", SAMPLE_SCHEMA)
        assert result == ["Title"]

    def test_multiple_columns(self):
        result = parse_columns_dsl("Title, Status, Due", SAMPLE_SCHEMA)
        assert result == ["Title", "Status", "Due"]

    def test_case_insensitive(self):
        result = parse_columns_dsl("title, status", SAMPLE_SCHEMA)
        assert result == ["Title", "Status"]

    def test_with_spaces_in_name(self):
        result = parse_columns_dsl("Title, Last Contact, Last Platform", SAMPLE_SCHEMA)
        assert result == ["Title", "Last Contact", "Last Platform"]

    def test_deduplication(self):
        result = parse_columns_dsl("Title, Status, Title", SAMPLE_SCHEMA)
        assert result == ["Title", "Status"]

    def test_empty_string(self):
        result = parse_columns_dsl("", SAMPLE_SCHEMA)
        assert result == []

    def test_whitespace_trimming(self):
        result = parse_columns_dsl("  Title ,  Status  ", SAMPLE_SCHEMA)
        assert result == ["Title", "Status"]

    def test_error_unknown_column(self):
        with pytest.raises(FilterParseError, match="Unknown property"):
            parse_columns_dsl("Title, Nonexistent", SAMPLE_SCHEMA)


# =============================================================================
# Integration Tests — Live Notion API
# =============================================================================

# Test Playground page UUID (shared with integration)
TEST_PLAYGROUND_ID = "2e7b94fa-2460-80e4-80f3-cba1f6fe0866"


def _notion_available() -> bool:
    """Check if Notion API is available (token exists and works)."""
    try:
        _notion_request("GET", "/users/me")
        return True
    except Exception:
        return False


def _create_test_database(parent_page_id: str, title: str) -> dict:
    """Create a database under the test playground for integration tests.

    In API 2025-09-03, properties are added via data_source PATCH,
    not in the database creation call.

    Returns the created database object (with data_sources).
    """
    # Step 1: Create database container with just the title property
    db = _notion_request("POST", "/databases", json_body={
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": title}}],
        "properties": {"Name": {"title": {}}},
    })

    # Step 2: Add remaining properties via data_source PATCH
    ds_id = db["data_sources"][0]["id"]
    _notion_request("PATCH", f"/data_sources/{ds_id}", json_body={
        "properties": {
            "Status": {
                "select": {
                    "options": [
                        {"name": "Todo", "color": "gray"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Done", "color": "green"},
                    ]
                }
            },
            "Priority": {"number": {}},
            "Due": {"date": {}},
            "Done": {"checkbox": {}},
            "Notes": {"rich_text": {}},
        },
    })

    return db


def _add_test_row(database_id: str, name: str, **props) -> dict:
    """Add a row to the test database."""
    page_props: dict = {
        "Name": {"title": [{"text": {"content": name}}]},
    }
    if "status" in props:
        page_props["Status"] = {"select": {"name": props["status"]}}
    if "priority" in props:
        page_props["Priority"] = {"number": props["priority"]}
    if "due" in props:
        page_props["Due"] = {"date": {"start": props["due"]}}
    if "done" in props:
        page_props["Done"] = {"checkbox": props["done"]}
    if "notes" in props:
        page_props["Notes"] = {"rich_text": [{"text": {"content": props["notes"]}}]}

    return _notion_request("POST", "/pages", json_body={
        "parent": {"type": "database_id", "database_id": database_id},
        "properties": page_props,
    })


def _archive_database(database_id: str) -> None:
    """Archive (delete) a database."""
    try:
        _notion_request("PATCH", f"/blocks/{database_id}", json_body={"archived": True})
    except Exception:
        pass  # Best-effort cleanup


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop shared across all integration tests.

    Using event_loop.run_until_complete() closes the loop after each call, which breaks
    httpx async connections in subsequent tests. A shared loop avoids this.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def test_db():
    """Create a test database with seed data, yield it, then archive."""
    if not _notion_available():
        pytest.skip("Notion API not available (no token or network)")

    import time
    title = f"FilterDSL Test {int(time.time())}"
    db = _create_test_database(TEST_PLAYGROUND_ID, title)
    db_id = db["id"]

    # Seed rows
    _add_test_row(db_id, "Fix login bug", status="Done", priority=3,
                  due="2024-01-10", done=True, notes="Critical fix")
    _add_test_row(db_id, "Add dark mode", status="In Progress", priority=2,
                  due="2024-01-20", done=False, notes="Design ready")
    _add_test_row(db_id, "Write docs", status="Todo", priority=1,
                  due="2024-02-01", done=False)
    _add_test_row(db_id, "Fix search bug", status="Done", priority=5,
                  due="2024-01-05", done=True, notes="Quick fix")
    _add_test_row(db_id, "Deploy v2", status="In Progress", priority=4,
                  due="2024-01-15", done=False)

    # Small delay for Notion to index
    time.sleep(1)

    yield {"id": db_id, "title": title}

    # Teardown
    _archive_database(db_id)


@pytest.mark.integration
class TestNotionReadDbE2E:
    """End-to-end tests for notion_read with filter/sort/columns on live Notion API."""

    def test_read_unfiltered(self, test_db, event_loop):
        """Basic database read without filters returns all rows."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50)
        )
        assert "@dnn 1" in result
        assert "@db " in result
        assert "@cols " in result
        assert "@rows 5" in result

    def test_filter_select_equals(self, test_db, event_loop):
        """Filter by select property."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Status = Done")
        )
        assert "@rows 2" in result
        # Should contain both "Done" rows
        assert "Fix login bug" in result
        assert "Fix search bug" in result
        # Should not contain non-Done rows
        assert "Add dark mode" not in result

    def test_filter_number_greater_than(self, test_db, event_loop):
        """Filter by number comparison."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Priority > 3")
        )
        # Priority 4 and 5
        assert "Fix search bug" in result
        assert "Deploy v2" in result

    def test_filter_date_before(self, test_db, event_loop):
        """Filter by date before."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Due < 2024-01-15")
        )
        # Due before Jan 15: "Fix login bug" (Jan 10), "Fix search bug" (Jan 5)
        assert "Fix login bug" in result
        assert "Fix search bug" in result
        assert "Add dark mode" not in result

    def test_filter_checkbox(self, test_db, event_loop):
        """Filter by checkbox."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Done = 1")
        )
        assert "@rows 2" in result

    def test_filter_text_contains(self, test_db, event_loop):
        """Filter by text contains."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Name ~ bug")
        )
        assert "Fix login bug" in result
        assert "Fix search bug" in result
        assert "Add dark mode" not in result

    def test_filter_compound_and(self, test_db, event_loop):
        """Compound AND filter."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Status = Done & Priority > 3")
        )
        # Only "Fix search bug" (Done, priority 5)
        assert "@rows 1" in result
        assert "Fix search bug" in result

    def test_filter_compound_or(self, test_db, event_loop):
        """Compound OR filter."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl='Status = Done | Status = "In Progress"')
        )
        # 2 Done + 2 In Progress = 4
        assert "@rows 4" in result

    def test_sort_desc(self, test_db, event_loop):
        """Sort by priority descending."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        sort_dsl="Priority desc")
        )
        lines = result.strip().split("\n")
        # Find data rows (after blank line following headers)
        data_start = None
        for i, line in enumerate(lines):
            if line == "" and i > 0:
                data_start = i + 1
        assert data_start is not None
        data_lines = [l for l in lines[data_start:] if l.strip()]
        # First data row should have highest priority (5 = Fix search bug)
        assert "Fix search bug" in data_lines[0]

    def test_columns_selection(self, test_db, event_loop):
        """Columns parameter limits which columns are shown."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        columns_dsl="Name, Status")
        )
        assert "@cols " in result
        # @cols should only have Name and Status
        cols_line = [l for l in result.split("\n") if l.startswith("@cols ")][0]
        assert "Name" in cols_line
        assert "Status" in cols_line
        assert "Priority" not in cols_line
        assert "Due" not in cols_line

    def test_filter_and_sort_combined(self, test_db, event_loop):
        """Filter + sort together."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Done = 0", sort_dsl="Priority desc")
        )
        assert "@rows 3" in result
        lines = result.strip().split("\n")
        data_start = None
        for i, line in enumerate(lines):
            if line == "" and i > 0:
                data_start = i + 1
        data_lines = [l for l in lines[data_start:] if l.strip()]
        # First should be "Deploy v2" (priority 4, not done)
        assert "Deploy v2" in data_lines[0]

    def test_filter_is_empty(self, test_db, event_loop):
        """Filter by is_empty on Notes."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Notes ?")
        )
        # "Write docs" and "Deploy v2" have no notes
        assert "Write docs" in result
        assert "Deploy v2" in result

    def test_filter_error_unknown_property(self, test_db, event_loop):
        """Unknown property returns error."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Nonexistent = foo")
        )
        assert "error: FILTER_ERROR" in result
        assert "Unknown property" in result

    def test_filter_error_bad_operator(self, test_db, event_loop):
        """Invalid operator for type returns error."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "edit", depth=10, limit=50,
                        filter_dsl="Status ~ Done")
        )
        assert "error: FILTER_ERROR" in result
        assert "not valid" in result

    def test_view_mode(self, test_db, event_loop):
        """View mode works with filters (no short IDs in output)."""
        result = event_loop.run_until_complete(
            _read_impl(test_db["id"], "view", depth=10, limit=50,
                        filter_dsl="Status = Done")
        )
        assert "@rows 2" in result
        # In view mode, rows don't start with 4-char IDs
        lines = result.strip().split("\n")
        data_start = None
        for i, line in enumerate(lines):
            if line == "" and i > 0:
                data_start = i + 1
        if data_start:
            data_lines = [l for l in lines[data_start:] if l.strip()]
            if data_lines:
                # First cell is icon (may be empty), second is name
                cells = data_lines[0].split(",")
                assert cells[1].startswith("Fix")


# =============================================================================
# DNN Mention Rendering Tests
# =============================================================================


class TestNotionRichTextToDnn:
    """Tests for notion_rich_text_to_dnn — especially new mention types."""

    def test_user_mention(self):
        rich_text = [{"type": "mention", "mention": {
            "type": "user", "user": {"id": "abc-123"}
        }, "annotations": {"color": "default"}, "plain_text": "John"}]
        assert notion_rich_text_to_dnn(rich_text) == "@user:abc-123"

    def test_date_mention(self):
        rich_text = [{"type": "mention", "mention": {
            "type": "date", "date": {"start": "2025-06-15"}
        }, "annotations": {"color": "default"}, "plain_text": "2025-06-15"}]
        assert notion_rich_text_to_dnn(rich_text) == "@date:2025-06-15"

    def test_date_range_mention(self):
        rich_text = [{"type": "mention", "mention": {
            "type": "date", "date": {"start": "2025-01-01", "end": "2025-12-31"}
        }, "annotations": {"color": "default"}, "plain_text": "Jan-Dec"}]
        assert notion_rich_text_to_dnn(rich_text) == "@date:2025-01-01→2025-12-31"

    def test_page_mention(self):
        rich_text = [{"type": "mention", "mention": {
            "type": "page", "page": {"id": "abc-def-123"}
        }, "annotations": {"color": "default"}, "plain_text": "My Page"}]
        assert notion_rich_text_to_dnn(rich_text) == "[My Page](p:abc-def-123)"

    def test_database_mention(self):
        """Database mentions render as page links (databases are pages)."""
        rich_text = [{"type": "mention", "mention": {
            "type": "database", "database": {"id": "db-uuid-456"}
        }, "annotations": {"color": "default"}, "plain_text": "Task DB"}]
        assert notion_rich_text_to_dnn(rich_text) == "[Task DB](p:db-uuid-456)"

    def test_link_mention(self):
        """link_mention renders as standard link (preview card lost)."""
        rich_text = [{"type": "mention", "mention": {
            "type": "link_mention",
            "link_mention": {"href": "https://example.com/article"}
        }, "annotations": {"color": "default"},
            "plain_text": "Example Article"}]
        result = notion_rich_text_to_dnn(rich_text)
        assert result == "[Example Article](https://example.com/article)"

    def test_link_preview_mention(self):
        """link_preview renders as standard link."""
        rich_text = [{"type": "mention", "mention": {
            "type": "link_preview",
            "link_preview": {"url": "https://github.com/org/repo/pull/1"}
        }, "annotations": {"color": "default"},
            "plain_text": "PR #1"}]
        result = notion_rich_text_to_dnn(rich_text)
        assert result == "[PR #1](https://github.com/org/repo/pull/1)"

    def test_template_mention(self):
        """template_mention renders as plain text."""
        rich_text = [{"type": "mention", "mention": {
            "type": "template_mention",
            "template_mention": {"type": "template_mention_date",
                                 "template_mention_date": "today"}
        }, "annotations": {"color": "default"}, "plain_text": "today"}]
        assert notion_rich_text_to_dnn(rich_text) == "today"

    def test_unknown_mention_type(self):
        """Unknown mention types fall back to plain_text."""
        rich_text = [{"type": "mention", "mention": {
            "type": "future_type", "future_type": {"data": "something"}
        }, "annotations": {"color": "default"}, "plain_text": "some text"}]
        assert notion_rich_text_to_dnn(rich_text) == "some text"

    def test_mention_with_bold_annotation(self):
        """Annotations on mentions are preserved."""
        rich_text = [{"type": "mention", "mention": {
            "type": "page", "page": {"id": "abc-123"}
        }, "annotations": {"bold": True, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "My Page"}]
        assert notion_rich_text_to_dnn(rich_text) == "**[My Page](p:abc-123)**"

    def test_mention_with_color_annotation(self):
        """Color annotations on mentions are preserved."""
        rich_text = [{"type": "mention", "mention": {
            "type": "user", "user": {"id": "abc-123"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "red"},
            "plain_text": "John"}]
        assert notion_rich_text_to_dnn(rich_text) == ":red[@user:abc-123]"

    def test_mention_with_code_annotation(self):
        """Code annotation on mentions wraps in backticks."""
        rich_text = [{"type": "mention", "mention": {
            "type": "date", "date": {"start": "2025-01-01"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": True, "color": "default"},
            "plain_text": "2025-01-01"}]
        assert notion_rich_text_to_dnn(rich_text) == "`@date:2025-01-01`"

    def test_mention_with_background_color(self):
        """Background color annotation uses dash format."""
        rich_text = [{"type": "mention", "mention": {
            "type": "page", "page": {"id": "abc-123"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "blue_background"},
            "plain_text": "My Page"}]
        result = notion_rich_text_to_dnn(rich_text)
        assert result == ":blue-background[[My Page](p:abc-123)]"

    def test_mention_with_multiple_annotations(self):
        """Multiple annotations stack correctly on mentions."""
        rich_text = [{"type": "mention", "mention": {
            "type": "user", "user": {"id": "abc-123"}
        }, "annotations": {"bold": True, "italic": True,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "red"},
            "plain_text": "John"}]
        result = notion_rich_text_to_dnn(rich_text)
        assert result == ":red[***@user:abc-123***]"

    def test_equation(self):
        rich_text = [{"type": "equation", "equation": {
            "expression": "E = mc^2"
        }}]
        assert notion_rich_text_to_dnn(rich_text) == "$E = mc^2$"

    def test_empty_rich_text(self):
        assert notion_rich_text_to_dnn([]) == ""


# =============================================================================
# MOVE_MISSING_PARENT Error Detection Tests
# =============================================================================


class TestMoveMissingParentError:
    """Tests for MOVE_MISSING_PARENT error detection."""

    def test_move_missing_parent_detected(self):
        """m X -> after=Y (no parent=) produces MOVE_MISSING_PARENT error."""
        script = "m SRC1 -> after=AFT1"
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 1
        assert result.errors[0].code == "MOVE_MISSING_PARENT"
        assert "parent=" in result.errors[0].message
        assert "reposition" in result.errors[0].message
        assert result.errors[0].line == 0
        assert len(result.errors[0].suggestions) == 1
        assert "parent=???" in result.errors[0].suggestions[0]

    def test_valid_move_not_caught(self):
        """Valid move command with parent= is not caught by this check."""
        script = "m SRC1 -> parent=DST1 after=AFT1"
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].command == ApplyCommand.MOVE

    def test_move_without_after_not_caught(self):
        """Move command without after= is not caught by missing-parent check."""
        script = "m SRC1 -> parent=DST1"
        registry = IdRegistry()
        result = parse_apply_script(script, registry)

        assert len(result.errors) == 0
        assert len(result.operations) == 1


# =============================================================================
# Movability Pre-Check Tests
# =============================================================================


class TestCheckMovability:
    """Tests for _check_movability function."""

    def test_paragraph_is_movable(self):
        """Paragraph blocks move faithfully — no warnings, no errors."""
        block = {"type": "paragraph", "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": "hello"}}]
        }}
        warnings, error = _check_movability(block)
        assert error is None
        assert warnings == []

    def test_synced_block_is_unmovable(self):
        """synced_block is refused with clear error."""
        block = {"type": "synced_block", "synced_block": {}}
        warnings, error = _check_movability(block)
        assert error is not None
        assert "sync relationship" in error

    def test_link_preview_block_is_unmovable(self):
        """link_preview block is refused."""
        block = {"type": "link_preview", "link_preview": {
            "url": "https://example.com"
        }}
        warnings, error = _check_movability(block)
        assert error is not None
        assert "integration-specific" in error

    def test_table_is_unmovable(self):
        """table block is refused."""
        block = {"type": "table", "table": {}}
        warnings, error = _check_movability(block)
        assert error is not None
        assert "table structure" in error

    def test_table_of_contents_is_unmovable(self):
        block = {"type": "table_of_contents", "table_of_contents": {}}
        warnings, error = _check_movability(block)
        assert error is not None
        assert "positional" in error

    def test_breadcrumb_is_unmovable(self):
        block = {"type": "breadcrumb", "breadcrumb": {}}
        warnings, error = _check_movability(block)
        assert error is not None
        assert "positional" in error

    def test_unsupported_is_unmovable(self):
        block = {"type": "unsupported", "unsupported": {}}
        warnings, error = _check_movability(block)
        assert error is not None
        assert "unsupported" in error

    def test_image_with_hosted_file_warns(self):
        """Image block with Notion-hosted file gets info warning."""
        block = {"type": "image", "image": {
            "type": "file",
            "file": {"url": "https://s3.aws.example.com/img.png"}
        }}
        warnings, error = _check_movability(block)
        assert error is None
        assert len(warnings) == 1
        assert "re-uploaded" in warnings[0]

    def test_image_with_external_url_no_warning(self):
        """Image block with external URL — no warning needed."""
        block = {"type": "image", "image": {
            "type": "external",
            "external": {"url": "https://example.com/img.png"}
        }}
        warnings, error = _check_movability(block)
        assert error is None
        assert warnings == []

    def test_block_with_link_mention_warns(self):
        """Block containing link_mention in rich_text gets warning."""
        block = {"type": "paragraph", "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": "See "}},
                {"type": "mention", "mention": {
                    "type": "link_mention",
                    "link_mention": {"href": "https://example.com"}
                }, "plain_text": "Example"}
            ]
        }}
        warnings, error = _check_movability(block)
        assert error is None
        assert len(warnings) == 1
        assert "rich URL embeds" in warnings[0]

    def test_all_unmovable_types_covered(self):
        """All UNMOVABLE_BLOCK_TYPES return errors."""
        for block_type in UNMOVABLE_BLOCK_TYPES:
            block = {"type": block_type, block_type: {}}
            _, error = _check_movability(block)
            assert error is not None, f"{block_type} should be unmovable"

    def test_all_file_bearing_types_exist(self):
        """Verify FILE_BEARING_BLOCK_TYPES covers expected types."""
        assert FILE_BEARING_BLOCK_TYPES == {"image", "file", "video", "pdf"}


# =============================================================================
# File Re-upload Tests
# =============================================================================


class TestReuploadNotionFile:
    """Tests for _reupload_notion_file with mocked API."""

    def test_external_file_returned_as_is(self):
        """External files don't need re-upload."""
        file_obj = {"type": "external", "external": {
            "url": "https://example.com/img.png"
        }}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result == file_obj
        assert warning is None

    def test_file_upload_returned_as_is(self):
        """Already-uploaded files don't need re-upload."""
        file_obj = {"type": "file_upload", "file_upload": {"id": "existing-id"}}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result == file_obj
        assert warning is None

    def test_unknown_type_returned_as_is(self):
        """Unknown file types are returned as-is."""
        file_obj = {"type": "something_new", "something_new": {}}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result == file_obj
        assert warning is None

    def test_file_without_url_returns_warning(self):
        """File with no URL returns a warning."""
        file_obj = {"type": "file", "file": {}}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result == file_obj
        assert "no URL" in warning

    def test_successful_reupload(self, monkeypatch):
        """Successful re-upload returns file_upload reference."""
        import notion_mcp

        poll_count = 0

        async def mock_request(method, endpoint, json_body=None):
            nonlocal poll_count
            if method == "POST" and endpoint == "/file_uploads":
                return {
                    "id": "new-upload-id",
                    "upload_url": "https://api.notion.com/v1/file_uploads/new-upload-id/send",
                    "status": "pending",
                }
            if method == "GET" and "/file_uploads/" in endpoint:
                poll_count += 1
                if poll_count >= 2:
                    return {"id": "new-upload-id", "status": "uploaded"}
                return {"id": "new-upload-id", "status": "pending"}
            return {}

        # Mock the download (httpx client.get) and upload (client.post)
        class MockResponse:
            status_code = 200
            content = b"fake-image-bytes"
            def raise_for_status(self):
                pass

        class MockClient:
            async def get(self, url, **kwargs):
                return MockResponse()
            async def post(self, url, **kwargs):
                return MockResponse()

        monkeypatch.setattr(notion_mcp, "_notion_request_async", mock_request)
        async def mock_get_client():
            return MockClient()
        monkeypatch.setattr(notion_mcp, "_get_async_client", mock_get_client)
        monkeypatch.setattr(notion_mcp, "_get_token", lambda: "fake-token")

        file_obj = {"type": "file", "file": {
            "url": "https://s3.amazonaws.com/some/image.png?sig=abc",
            "expiry_time": "2025-01-01T00:00:00.000Z"
        }}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result["type"] == "file_upload"
        assert result["file_upload"]["id"] == "new-upload-id"
        assert warning is None

    def test_failed_reupload_falls_back(self, monkeypatch):
        """Failed re-upload falls back to original with warning."""
        import notion_mcp

        async def mock_request(method, endpoint, json_body=None):
            if method == "POST" and endpoint == "/file_uploads":
                return {
                    "id": "upload-id",
                    "upload_url": "https://api.notion.com/v1/file_uploads/upload-id/send",
                    "status": "pending",
                }
            if method == "GET" and "/file_uploads/" in endpoint:
                return {"id": "upload-id", "status": "failed"}
            return {}

        class MockResponse:
            status_code = 200
            content = b"fake-image-bytes"
            def raise_for_status(self):
                pass

        class MockClient:
            async def get(self, url, **kwargs):
                return MockResponse()
            async def post(self, url, **kwargs):
                return MockResponse()

        monkeypatch.setattr(notion_mcp, "_notion_request_async", mock_request)
        async def mock_get_client():
            return MockClient()
        monkeypatch.setattr(notion_mcp, "_get_async_client", mock_get_client)
        monkeypatch.setattr(notion_mcp, "_get_token", lambda: "fake-token")

        file_obj = {"type": "file", "file": {
            "url": "https://s3.amazonaws.com/some/image.png",
            "expiry_time": "2025-01-01T00:00:00.000Z"
        }}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result == file_obj
        assert "failed" in warning

    def test_api_error_falls_back(self, monkeypatch):
        """API exception on download falls back to original with warning."""
        import notion_mcp

        class MockClient:
            async def get(self, url, **kwargs):
                raise RuntimeError("Download failed")

        async def mock_get_client():
            return MockClient()
        monkeypatch.setattr(notion_mcp, "_get_async_client", mock_get_client)

        file_obj = {"type": "file", "file": {
            "url": "https://s3.amazonaws.com/some/image.png",
            "expiry_time": "2025-01-01T00:00:00.000Z"
        }}
        result, warning = asyncio.run(
            _reupload_notion_file(file_obj)
        )
        assert result == file_obj
        assert "failed" in warning


# =============================================================================
# DNN Round-Trip Tests (Mention Types)
# =============================================================================


class TestDnnMentionRoundTrip:
    """Test that DNN render → parse → build preserves mention types."""

    def test_user_mention_round_trip(self):
        """User mention survives DNN round-trip."""
        user_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        rich_text = [{"type": "mention", "mention": {
            "type": "user", "user": {"id": user_uuid}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "John"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert f"@user:{user_uuid}" in dnn

        # Parse back
        spans = parse_inline_formatting(dnn)
        assert any(s.span_type == "mention_user" and s.user_id == user_uuid
                    for s in spans)

    def test_date_mention_round_trip(self):
        """Date mention survives DNN round-trip."""
        rich_text = [{"type": "mention", "mention": {
            "type": "date", "date": {"start": "2025-06-15"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "June 15"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert "@date:2025-06-15" in dnn

        spans = parse_inline_formatting(dnn)
        assert any(s.span_type == "mention_date" and s.date == "2025-06-15"
                    for s in spans)

    def test_date_range_mention_round_trip(self):
        """Date range mention survives DNN round-trip."""
        rich_text = [{"type": "mention", "mention": {
            "type": "date", "date": {"start": "2025-01-01", "end": "2025-12-31"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "2025"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert "@date:2025-01-01→2025-12-31" in dnn

        spans = parse_inline_formatting(dnn)
        assert any(s.span_type == "mention_date" and s.date == "2025-01-01"
                    and s.end_date == "2025-12-31" for s in spans)

    def test_page_mention_round_trip(self):
        """Page mention survives DNN round-trip as page link."""
        page_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        rich_text = [{"type": "mention", "mention": {
            "type": "page", "page": {"id": page_uuid}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "My Page"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert f"[My Page](p:{page_uuid})" in dnn

        # Parse back — should create a link span with p: prefix
        spans = parse_inline_formatting(dnn)
        assert any(s.link and s.link.startswith("p:") for s in spans)

    def test_database_mention_round_trip(self):
        """Database mention renders as page link, survives round-trip."""
        db_uuid = "db123456-e5f6-7890-abcd-ef1234567890"
        rich_text = [{"type": "mention", "mention": {
            "type": "database", "database": {"id": db_uuid}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "Task DB"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert f"[Task DB](p:{db_uuid})" in dnn

    def test_link_mention_round_trip(self):
        """link_mention renders as standard link, parseable on round-trip."""
        rich_text = [{"type": "mention", "mention": {
            "type": "link_mention",
            "link_mention": {"href": "https://example.com/article"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "Example Article"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert "[Example Article](https://example.com/article)" in dnn

        # Parse back — should create a link span
        spans = parse_inline_formatting(dnn)
        assert any(s.link == "https://example.com/article" for s in spans)

    def test_template_mention_round_trip(self):
        """template_mention renders as plain text."""
        rich_text = [{"type": "mention", "mention": {
            "type": "template_mention",
            "template_mention": {"type": "template_mention_date",
                                 "template_mention_date": "today"}
        }, "annotations": {"bold": False, "italic": False,
                           "strikethrough": False, "underline": False,
                           "code": False, "color": "default"},
            "plain_text": "today"}]

        dnn = notion_rich_text_to_dnn(rich_text)
        assert dnn == "today"


class TestIdentifyMoveChains:
    """Tests for _identify_move_chains — detecting consecutive same-parent moves."""

    def _move_op(self, source, dest_parent, dest_after=None, line_num=0):
        return ApplyOp(
            command=ApplyCommand.MOVE,
            line_num=line_num,
            source=source,
            dest_parent=dest_parent,
            dest_after=dest_after,
        )

    def _add_op(self, parent, line_num=0):
        return ApplyOp(
            command=ApplyCommand.ADD,
            line_num=line_num,
            parent=parent,
        )

    def test_empty(self):
        assert _identify_move_chains([]) == []

    def test_single_move(self):
        """A single move is not a chain."""
        ops = [self._move_op("A", "X")]
        assert _identify_move_chains(ops) == []

    def test_two_moves_same_parent(self):
        """Two consecutive moves to the same parent form a chain."""
        ops = [
            self._move_op("A", "X"),
            self._move_op("B", "X"),
        ]
        assert _identify_move_chains(ops) == [[0, 1]]

    def test_three_moves_same_parent(self):
        ops = [
            self._move_op("A", "X"),
            self._move_op("B", "X"),
            self._move_op("C", "X"),
        ]
        assert _identify_move_chains(ops) == [[0, 1, 2]]

    def test_two_moves_different_parents(self):
        """Moves to different parents don't chain."""
        ops = [
            self._move_op("A", "X"),
            self._move_op("B", "Y"),
        ]
        assert _identify_move_chains(ops) == []

    def test_explicit_after_breaks_chain(self):
        """A move with explicit after= is not chainable."""
        ops = [
            self._move_op("A", "X"),
            self._move_op("B", "X", dest_after="Z"),
            self._move_op("C", "X"),
        ]
        # A is alone (chain of 1), B has explicit after (standalone),
        # C is alone (chain of 1). No chains of length >= 2.
        assert _identify_move_chains(ops) == []

    def test_non_move_breaks_chain(self):
        """A non-move op between moves breaks the chain."""
        ops = [
            self._move_op("A", "X"),
            self._add_op("Y"),
            self._move_op("B", "X"),
        ]
        assert _identify_move_chains(ops) == []

    def test_two_separate_chains(self):
        """Different parent groups form separate chains."""
        ops = [
            self._move_op("A", "X"),
            self._move_op("B", "X"),
            self._move_op("C", "Y"),
            self._move_op("D", "Y"),
        ]
        assert _identify_move_chains(ops) == [[0, 1], [2, 3]]

    def test_chain_then_single(self):
        """Chain followed by a lone move to a different parent."""
        ops = [
            self._move_op("A", "X"),
            self._move_op("B", "X"),
            self._move_op("C", "Y"),
        ]
        assert _identify_move_chains(ops) == [[0, 1]]

    def test_mixed_ops_with_chain(self):
        """Chain surrounded by non-move ops."""
        ops = [
            self._add_op("Z"),
            self._move_op("A", "X"),
            self._move_op("B", "X"),
            self._move_op("C", "X"),
            self._add_op("Z"),
        ]
        assert _identify_move_chains(ops) == [[1, 2, 3]]
