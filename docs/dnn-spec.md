# DNN/1 — Dauphin Notion Notation, version 1

Ultra-compact Notion page representation optimized for LLM manipulation.

## Problem: Notion API Verbosity

A simple paragraph "Hello World" returns ~400 chars of JSON.
**Target**: 11 chars. **Overhead: 97%**.

---

## Solution: DNN Format (Dauphin Notion Notation)

### Design Principles

1. **Markdown-inspired** - familiar syntax for headings, lists, etc.
2. **Indentation for nesting** - 2 spaces = 1 level
3. **Fixed 4-char IDs** - left-justified, session-scoped mapping
4. **Per-span colors** - `:red[text]` [Streamlit-style](https://docs.streamlit.io/develop/api-reference/text/st.markdown) inline
5. **Read-only mode** - omit IDs entirely for max compression
6. **Opaque blocks** - unsupported types shown as `!type~` placeholders
7. **Structural blocks** - container types (tables, columns) use `!type N` notation

---

## Parsing Contexts

DNN parsing uses a **state machine** with three contexts:

### 1. Header Context

Lines before the first block line are **header/metadata**.
Header lines use `@` prefix:

```
@dnn 1
@page 9a8b7c6d
@title My Todo List
```

A line is a **block line** iff it matches `^[A-Za-z0-9]{4} `.
All lines before the first block line are header context.

### 2. Block Context

Normal parsing: extract ID, indent, marker, content.

### 3. Code Block Context

When a block line contains ` ``` `:
- Code block **starts** at `XXXX ```lang` (3+ backticks)
- Subsequent lines are **raw content** (no ID, no marker parsing)
- Code block **ends** at a line with backticks matching the opening length at same indent
- The fence length auto-adjusts to avoid collision with backtick runs in content

````
A1b2 ```python
def greet(name):
    return f"Hello, {name}!"
```
C3d4 Next block after code
````

**Error**: `CODE_BLOCK_UNTERMINATED` if EOF reached without closing.

---

## ID System

### Format

- **Length**: Fixed 4 characters (e.g., `A1b2`, `Xy9Z`)
- **Alphabet**: Full base62 (a-z, A-Z, 0-9), case-sensitive
- **Position**: IDs are always left-justified (column 0)
- **Layout**: `XXXX<space><indent><content>`

### Left-Justified Layout

IDs stay at column 0; indentation follows the space separator:

```
A1b2 # Heading
C3d4 > Toggle block
E5f6   Child of toggle (2-space indent)
G7h8   Another child
I9j0     Grandchild (4-space indent)
K1l2 Next top-level block
```

### Parsing Rule

1. Extract chars 0-3 = ID
2. Char 4 = space (skip)
3. Count leading spaces in remainder (÷2 = nesting level)
4. Parse remaining content

### Session-Scoped Mapping

Server maintains bidirectional map per session:
- `short_id` → `notion_uuid`
- `notion_uuid` → `short_id`

### Accepting Full UUIDs

All tools accept any of:
- Short ID: `A1b2`
- Full UUID: `12345678-1234-1234-1234-123456789abc`
- UUID without dashes: `12345678123412341234123456789abc`
- Notion URL: `https://notion.so/workspace/Page-abc123`
- Typed ref: `p:A1b2` (page), `b:C3d4` (block), `r:E5f6` (row)

---

## Block Type Markers

**Rule**: Markers are recognized at line start, followed by space.
To use marker chars as literal text, escape with backslash.

| Marker | Block Type | Notes |
|:-------|:-----------|:------|
| (none) | paragraph | Default |
| `# ` | heading_1 | Space required |
| `## ` | heading_2 | |
| `### ` | heading_3 | |
| `># ` | toggle heading_1 | Collapsible |
| `>## ` | toggle heading_2 | |
| `>### ` | toggle heading_3 | |
| `- ` | bulleted_list | |
| `N. ` | numbered_list | N = any integer |
| `[ ] ` | to_do unchecked | |
| `[x] ` | to_do checked | |
| `> ` | toggle | Single `>` + space |
| `\| ` | quote | Pipe + space |
| `! ` | callout | Default color |
| `!color ` | callout | gray/blue/red/etc. |
| `---` | divider | Exactly `---` |
| `§ ` | child_page | Page-backed (see below) |
| `⊞ ` | child_database | Page-backed (see below) |
| `→ ` | link_to_page | Read-only |
| ` ``` ` | code block start | |
| `!cols N` | column_list | Structural (N = column count) |
| `!col N` | column | Structural (N = position, 1-indexed) |
| `!table N` | table | Structural (N = column count) |
| `\| c1 \| c2 \|` | table_row | Pipe-delimited cells |

---

## Structural Blocks

Structural blocks use the `!type N` notation for container blocks that
have editable structure (columns and tables). Unlike opaque blocks (`!type~`),
structural blocks are fully read/write.

| Marker | Notion Type | N Meaning |
|:-------|:------------|:----------|
| `!cols N` | column_list | Number of columns |
| `!col N` | column | Column position (1-indexed) |
| `!table N` | table | Number of columns (table_width) |

**Bang (`!`) prefix disambiguation order** (first match wins):
1. Colored callout: `!color text` (gray, blue, red, etc.)
2. Opaque block: `!type~` or `!type~ (caption)`
3. Structural block: `!type N` (where type is `cols`, `col`, or `table`)
4. Default callout: `! text`

### Tables

Tables render as a `!table N` parent with pipe-delimited `table_row` children:

```
A1b2 !table 3
C3d4   | Name | Status | Notes |
E5f6   | Task one | Done | Shipped |
G7h8   | Task two | In progress | On track |
```

**Auto-grouping**: When adding rows, consecutive pipe-delimited lines
without an explicit `!table` parent are automatically grouped into a
table. The column count is inferred from the first row:

```
+ parent=A1b2
  | Name | Status | Notes |
  | Task one | Done | Shipped |
  | Task two | In progress | On track |
```

Markdown separator rows (`|---|---|`) are silently dropped.

Pipes within cell content must be escaped: `\|`.

Table rows support full rich text (bold, links, colors, etc.) in cells.

### Column Layouts

Column layouts render as a `!cols N` parent with `!col N` children:

```
A1b2 !cols 2
C3d4   !col 1
E5f6     First column content
G7h8   !col 2
I9j0     Second column content
```

---

## Escape Sequences

### Block-Level Escapes (Line Start)

When content starts with a marker character, escape it:

| Input | Output | Block Type |
|:------|:-------|:-----------|
| `\# hashtag` | `# hashtag` | paragraph |
| `\- not a list` | `- not a list` | paragraph |
| `\> quoted text` | `> quoted text` | paragraph |
| `\| pipe symbol` | `| pipe symbol` | paragraph |
| `\! exclamation` | `! exclamation` | paragraph |
| `\1. not numbered` | `1. not numbered` | paragraph |
| `\[ ] brackets` | `[ ] brackets` | paragraph |
| `\--- dashes` | `--- dashes` | paragraph |
| `\→ arrow text` | `→ arrow text` | paragraph |

### Inline Escapes

| Escape | Literal | Use Case |
|:-------|:--------|:---------|
| `\\` | `\` | Backslash |
| `\*` | `*` | Not bold |
| `\~` | `~` | Not strikethrough |
| `` \` `` | `` ` `` | Not code |
| `\[` | `[` | Not link start |
| `\]` | `]` | Not link end |
| `\:` | `:` | Not color/equation directive |
| `\→` | `→` | Arrow in link text |

### Arrows Mid-Line

Arrows need escaping only at line start or in `[link](url)` text:
- `A1b2 Click here → next step` — works fine
- `A1b2 \→ Starts with arrow` — escaped at line start

---

## Inline Formatting

| Syntax | Renders As | Notes |
|:-------|:-----------|:------|
| `**text**` | **bold** | |
| `*text*` | *italic* | |
| `~~text~~` | ~~strikethrough~~ | |
| `:u[text]` | underline | Directive (avoids `__` ambiguity) |
| `` `text` `` | `code` | |
| `:eq[expr]` | equation | Directive syntax |
| `[text](url)` | link | |
| `@p:A1b2` | page @mention | **USE THIS** |
| `[text](p:A1b2)` | page link | Custom text only |
| `@user:UUID` | user mention | Full Notion user UUID |
| `@date:2024-01-15` | date mention | ISO format |
| `@date:2024-01-15→2024-01-20` | date range | Start→End |

### Page References

**Always use `@p:shortID` for page references.** This creates a Notion
@mention that displays the page's actual title and auto-updates if renamed.

Only use `[custom text](p:shortID)` when you specifically need different
link text than the page title (rare).

```
# GOOD - @mention shows real title, stays in sync
See @p:A1b2 for details.

# AVOID - custom text can become stale
See [the docs](p:A1b2) for details.
```

### Other Mention Types

These Notion mention types render as standard links in DNN output:
- **Database mentions** → `[db title](p:db_id)` (same syntax as page links)
- **Link mentions** (rich URL embeds) → `[title](url)`
- **Link previews** (integration embeds) → `[title](url)`
- **Template mentions** (dynamic values like "today") → plain text

### Inline Equations

- Inline equation: `:eq[x = 5]` → renders as LaTeX
- Dollar signs are literal text (no escaping needed): `$10` → "$10"

Example: `The formula :eq[E=mc^2] is famous.`

---

## Colors (Streamlit-Style)

Per-span color directives allow mixed colors within a line:

```
A1b2 This has :red[warning] and :green[success] text.
```

| Directive | Notion Color |
|:----------|:-------------|
| `:gray[text]` | gray |
| `:brown[text]` | brown |
| `:orange[text]` | orange |
| `:yellow[text]` | yellow |
| `:green[text]` | green |
| `:blue[text]` | blue |
| `:purple[text]` | purple |
| `:pink[text]` | pink |
| `:red[text]` | red |
| `:gray-background[text]` | gray_background |
| `:*-background[text]` | other backgrounds |

---

## Move Strategies

Notion's API has different capabilities for different object types.
We classify by **move strategy**:

### 1. Clone-Movable Blocks

Most blocks: paragraph, heading, list, toggle, quote, callout, etc.

**How it works**: Read subtree → recreate under new parent → archive
original. Short IDs are **preserved** (reassigned to the new block UUIDs)
so references remain valid after move.

### 2. Page-Movable Objects

Child pages (`§`) and child databases (`⊞`) are **page-backed**.

**How it works**: Use Notion's Move Page endpoint. The underlying
page/database retains its identity; no ID change.

Commands:
- `m` on `§` or `⊞` auto-routes to page move
- `u` on `§` → Update Page (changes title)
- `u` on `⊞` → Update Database (changes title)

### 3. Lossy-Clone Blocks

Some blocks may lose data when cloned:
- `!image~` — hosted file URLs may expire
- `!file~` — same issue
- `!bookmark~` — preview data may be lost
- `!link_preview~` — embed state may be lost

The `~` suffix warns: "move may be lossy."

### 4. Complex-Structure Blocks

Some blocks have creation constraints:
- `column_list` (`!cols N`) — must have ≥2 columns on create
- `table` (`!table N`) — must have ≥1 row on create
- `synced_block` — has `synced_from` reference model

These are clonable but require special handling.

---

## Opaque Blocks

Unsupported block types rendered as `!type~` placeholders:

- `image`, `video`, `file`, `pdf`, `bookmark`, `embed`
- `link_preview`, `synced_block`, `link_to_page`
- `table_of_contents`, `breadcrumb`
- `equation`, `template`, `unsupported`

Note: `table` and `table_row` are **not** opaque — they use structural
notation with full read/write support (see Structural Blocks above).

```
E5f6 !image~ (screenshot.png)
G7h8 !synced_block* (from Template)
K1l2 § My Subpage
M3n4 ⊞ Tasks Database
```

### Suffix Meanings

| Suffix | Meaning |
|:-------|:--------|
| `*` | Has hidden children |
| `~` | Clone may be lossy (file URLs expire) |

Note: `§` (child_page) and `⊞` (child_database) are page-backed.
Column layouts use `!cols N` and `!col N` structural markers (not opaque).
Tables use `!table N` structural markers with pipe-delimited rows (not opaque).

### Opaque Read Modes *(not yet implemented)*

`opaque` parameter on reads:
- `collapse` (default): one-line placeholder
- `expand`: show children if types are supported
- `raw`: debug mode, minimal JSON

---

## API Design

### Philosophy

Five tools: read, apply (write), search, get_url, and check_auth.

### Tool: notion_check_auth

Verify authentication and return workspace info.

| Param | Type | Default | Description |
|:------|:-----|:--------|:------------|
| (none) | | | No parameters |

**Returns:** String with auth status and workspace name, or error.

### Tool: notion_search

Search Notion by title.

| Param | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `query` | string | required | Search query |
| `filter_type` | enum | `all` | `page`, `database`, or `all` |
| `limit` | int | 20 | Max results (max 100) |

**Returns:** Compact search results with short IDs for use with notion_read.

### Tool: notion_get_url

Convert any reference to a Notion URL.

| Param | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `ref` | string | required | Short ID, UUID, or URL |

**Returns:** `https://notion.so/<uuid>` or error if unresolvable.

### Tool: notion_read

Read pages or databases. Accepts a list of refs with per-ref overrides.

| Param | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `pages` | list[dict] | required | List of page/DB specs (see below) |
| `depth` | int | 10 | Default nesting depth for pages |
| `limit` | int | 50 | Default DB row limit |
| `filter` | string | `""` | Default DB filter (see Filter DSL) |
| `sort` | string | `""` | Default DB sort (see Sort DSL) |
| `columns` | string | `""` | Default DB column selection |

Each entry in `pages` is a dict:

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `ref` | string | required | Short ID, UUID, URL, or typed ref |
| `mode` | enum | `"edit"` | `edit` (with IDs) or `view` (no IDs) |
| `depth` | int | (global) | Override nesting depth for this ref |
| `limit` | int | (global) | Override row limit for this ref |
| `filter` | string | (global) | Override filter for this ref |
| `sort` | string | (global) | Override sort for this ref |
| `columns` | string | (global) | Override columns for this ref |

**Returns for page:**
```
@dnn 1
@page 9a8b7c6d
@icon 📋
@title My Todo List

A1b2 # Today
C3d4 > Morning
E5f6   [x] Standup
G7h8   [ ] Review PRs
I9j0 > Afternoon
K1l2   [ ] Code review
```

**Returns for database:**
```
@dnn 1
@db 8x7y6z5w
@ds 7w6v5u4t
@title Tasks
@cols Name(title),Status(select),Due(date),Done(checkbox)

I9j0,Fix bug,In Progress,2024-01-12,0
K1l2,Ship it,Done,2024-01-13,1
```

Notes:
- `@icon` = page icon (emoji or external URL), omitted if none
- `@ds` = resolved data_source_id (required for queries)
- `@cols` = column definitions for compact updates
- Comma-separated values (CSV) with quote escaping
- Row IDs are 4-char like block IDs

#### Filter DSL

Filter databases using a compact expression syntax:

```
Status = Done
Title ~ bug
Due < 2024-01-15
Priority >= 3
"Last Contact" >= -14d
Status = Done & Due < 2024-01-15
(Status = Done | Status = "In Progress") & Priority > 3
Tags ~ urgent
Notes ?
Notes !?
```

**Operators:**

| Op | Meaning | Applies To |
|:---|:--------|:-----------|
| `=` | equals | title, text, number, select, status, date, checkbox, url, email, phone |
| `!=` | not equals | title, text, number, select, status, date, url, email, phone |
| `~` | contains | title, text, multi_select, url, email, phone |
| `!~` | does not contain | title, text, multi_select, url, email, phone |
| `<` | less than | number, date, created_time, last_edited_time |
| `>` | greater than | number, date, created_time, last_edited_time |
| `<=` | less than or equal | number, date, created_time, last_edited_time |
| `>=` | greater than or equal | number, date, created_time, last_edited_time |
| `?` | is empty (unary) | most types |
| `!?` | is not empty (unary) | most types |

**Logic:** `&` (AND, binds tighter) and `|` (OR), with parentheses.

**Values:**
- Unquoted words: `Done`, `123`, `2024-01-15`
- Quoted strings: `"In Progress"`, `"Has \"quotes\""`
- Relative dates: `-7d` (7 days ago), `-14d` (14 days ago)
- Checkbox: `1`/`0`, `true`/`false`, `yes`/`no`

**Property names** with spaces must be quoted: `"Last Contact" >= -14d`

**Grammar:**
```
expr     → and_expr ('|' and_expr)*
and_expr → atom ('&' atom)*
atom     → '(' expr ')' | predicate
predicate→ PROPERTY OP VALUE
         | PROPERTY '?' | PROPERTY '!?'
```

#### Sort DSL

Comma-separated sort specs. Default direction is ascending.

```
Due desc
Due desc, Status asc
"Last Contact" asc
Priority desc, Name
```

#### Columns DSL

Comma-separated property names. Selects which columns to return.

```
Name, Status, Due
"Last Contact", Priority
```

Property names are matched case-insensitively against the schema.

### Tool: notion_apply

All mutations in one script.

| Param | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `script` | string | required | Edit commands |
| `dry_run` | bool | false | Validate only |

#### Script Commands: Blocks

```
# Add blocks under parent
+ parent=A1b2
  [ ] New task
  [ ] Another task

# Add after specific sibling
+ parent=A1b2 after=C3d4
  - Inserted after C3d4

# Delete/archive blocks
x E5f6
x G7h8 I9j0 K1l2

# Move block (clone+archive, short IDs preserved)
m M3n4 -> parent=O5p6
m Q7r8 -> parent=S9t0 after=U1v2

# Update block text
u W3x4 = "Updated text here"
u Y5z6 = "Text with :red[color] and **bold**"

# Toggle todo checkbox
t A7b8 = 1
t C9d0 = 0
```

#### Script Commands: Pages

```
# Create page with optional icon and cover
+page parent=A1b2 title="New Page"
  # Initial content
  Some text here.

+page parent=A1b2 title="New Page" icon=📝
  Content with emoji icon.

+page parent=A1b2 title="New Page" icon=📝 cover=https://example.com/img.jpg
  Content with icon and cover image.

# Position: insert after a specific block
+page parent=A1b2 after=C3d4 title="After Page"

# Position: insert at the start
+page parent=A1b2 pos=start title="First Page"

# Create page in database
+page parent=db:X1y2 title="New Task"
  props: Status="Todo", Priority="High"
  [ ] First subtask

# Move page (works for § and ⊞ blocks too)
mpage P1q2 -> parent=R3s4

# Delete/archive page
xpage T5u6

# Update page title
upage P1q2 = "New Title"

# Update page icon (emoji or external URL)
upage P1q2 icon=🎯
upage P1q2 icon=https://example.com/icon.png

# Update page cover (external URL only)
upage P1q2 cover=https://example.com/cover.jpg

# Clear icon or cover
upage P1q2 icon=none
upage P1q2 cover=none

# Copy page to new parent (deep copy of content)
cpage SOURCE -> parent=TARGET
cpage SOURCE -> parent=TARGET title="Custom Title"
```

**Copy limitations**: Only pages can be copied. Databases, synced blocks,
and certain embedded content cannot be deep-copied. See error codes below.

#### Script Commands: Database Rows

```
# Add row (TSV format: Name=value separated by tabs)
+row db=X1y2
  Title=Task name	Status=Todo	Due=2024-01-15	Done=0

# Add row with date range
+row db=X1y2
  Title=Q1 Planning	Status=In Progress	Period=2025-01-01→2025-03-31

# Update row properties
urow I9j0
  Status=Done	Done=1

# Delete row
xrow K1l2
```

#### Script Commands: Database Schema

```
# Create database with typed properties
+db parent=A1b2 title="Tasks"
  Name(title)
  Status(select: Todo|In Progress|Done)
  Priority(select: High|Medium|Low)
  Due(date)
  Done(checkbox)
  Notes(rich_text)
  Score(number: number)

# Create database with icon
+db parent=A1b2 title="Tracker" icon=📊
  Name(title)
  Status(status)

# Add property to existing database
+prop db=X1y2 Priority(select: High|Medium|Low)
+prop db=X1y2 Due(date)

# Delete property
xprop db=X1y2 Priority
xprop db=X1y2 "Property With Spaces"

# Rename property
uprop db=X1y2 OldName -> NewName
uprop db=X1y2 "Old Name" -> "New Name"
```

**Property definition syntax:** `Name(type)` or `Name(type: config)`

Supported property types: `title`, `rich_text`, `number`, `select`,
`multi_select`, `status`, `date`, `checkbox`, `url`, `email`,
`phone_number`, `created_time`, `last_edited_time`, `relation`.

Select/multi_select options: `Status(select: Open|Closed|Blocked)`

Number format: `Score(number: percent)`, `Price(number: dollar)`

Note: Consecutive schema operations (`+prop`, `xprop`, `uprop`) on the
same database execute sequentially to avoid race conditions.

**Returns:** Compact multiline string (0-indexed line numbers):
```
0: +Z1a2 +Z3b4    (created IDs)
1: ok             (delete succeeded)
2: M3n4→N5o6      (moved: old→new)
3: ok             (update succeeded)
4: ok             (toggle succeeded)
---
5/5 ok            (summary)
```

Errors appear as `N: err MESSAGE`. Moves/type-changes show `OLD→NEW`.

**Important**: Moves are clone+archive, but short IDs are preserved
(reassigned to the new block's UUID).

---

## Insertion Limitations

Notion's Append Block Children endpoint:
- Appends to **end** by default
- Can insert **after** a specific block via `after` parameter
- **Cannot** insert before first child directly

---

## Execution Semantics

### Parallel Execution Model

All operations in a script run **concurrently** using `asyncio.gather()`.
This provides significant speedup over sequential execution.

**Key assumptions:**

1. **All IDs pre-exist** — Every ID referenced in a script must already
   exist in the session registry (from a prior `notion_read` or search).
   You cannot reference an ID created by an earlier op in the same script.

2. **No duplicate targets** — Each block ID can appear as a target in
   only ONE operation per script. Attempting to target the same block
   multiple times returns `CONFLICT_DETECTED` error.

   Valid:
   ```
   u A1b2 = "text"
   u C3d4 = "other"
   ```

   Invalid (same block):
   ```
   u A1b2 = "first"
   u A1b2 = "second"   # Error: A1b2 targeted by lines 1, 2
   ```

3. **Parent/after are not conflicts** — Multiple ops can add to the same
   parent or use the same `after` anchor. Only direct targets conflict.

4. **Move preserves short IDs** — The `m` command clones and archives,
   then reassigns the short ID to the new block UUID. References to the
   moved block remain valid.

### Rate Limiting

A shared semaphore limits concurrent Notion API requests to 50. Notion's
documented limit is 3 req/sec, but we use 50 since rate limiting hasn't
been an issue in practice. Exponential backoff handles any 429 responses.

### Result Ordering

Results are returned **by script line number**, not completion order.
If line 3 finishes before line 1, the output still shows line 1 first.

### Error Handling (Map-Style)

All ops run regardless of individual failures. Each line reports its
own status. Summary shows `N/M ok`. Use separate `notion_apply` calls
if you need sequential dependency between operations.

---

## Error Handling: Self-Healing

### Error Response Shape

```json
{
  "ok": false,
  "error": {
    "code": "INDENT_ERROR",
    "message": "Indent not multiple of 2",
    "line": 5,
    "excerpt": "5|    [ ] Task",
    "expected": ["0, 2, 4, or 6 spaces"],
    "suggestions": ["Round to nearest: 4 spaces"],
    "autofix": {
      "safe": true,
      "patched": "    [ ] Task"
    }
  }
}
```

### Error Codes

| Code | Meaning | Autofix? |
|:-----|:--------|:---------|
| `PARSE_ERROR` | Script syntax error | Sometimes |
| `UNKNOWN_ID` | Short ID not found | No |
| `REF_GONE` | Block deleted/archived | No |
| `INDENT_ERROR` | Not multiple of 2 | Yes |
| `TABS_IN_INDENT` | Tabs instead of spaces | Yes |
| `MISSING_ID_SPACE` | `A1b2#` not `A1b2 #` | Yes |
| `CODE_BLOCK_UNTERMINATED` | Missing ``` | Yes |
| `MULTI_DATASOURCE` | DB has multiple sources | No |
| `MISSING_CAPABILITY` | Integration lacks permission | No |
| `COPY_NOT_SUPPORTED` | Target type cannot be copied | No |
| `COPY_DATABASE` | Cannot copy databases via API | No |
| `COPY_SYNCED_BLOCK` | Cannot copy synced blocks | No |
| `COPY_CONTAINS_UNSUPPORTED` | Page contains uncopyable content | No |
| `INVALID_COVER_URL` | Cover must be external URL | No |
| `MOVE_DATABASE` | Cannot move databases via API | No |
| `CONFLICT_DETECTED` | Same block targeted by multiple ops | No |
| `FILTER_ERROR` | Invalid filter DSL syntax | No |
| `SORT_ERROR` | Invalid sort DSL syntax | No |
| `COLUMNS_ERROR` | Invalid column selection | No |
| `DB_PARAMS_ON_PAGE` | filter/sort/columns used on non-database ref | No |

### High-Value Autofixes

| Error | Autofix |
|:------|:--------|
| Indent 3 spaces | → 2 or 4 (nearest) |
| Tab characters | → 2 spaces each |
| `A1b2#` | → `A1b2 #` |
| `- [X]` | → `- [x]` |
| Missing closing ``` | → Insert at correct indent |

### Capability Errors

Notion returns 403 for missing integration capabilities.
Surface as:

```json
{
  "code": "MISSING_CAPABILITY",
  "message": "Integration lacks 'Insert content' capability",
  "suggestions": [
    "Share page with integration",
    "Enable 'Insert content' in integration settings"
  ]
}
```

---

## Complex Examples

### Example 1: Full Page with Header

```
@dnn 1
@page 9a8b7c6d
@title Project Tasks

A1b2 # Phase 1
C3d4 > Research
E5f6   [x] Read docs
G7h8   [ ] Write summary
I9j0     [ ] Draft outline
K1l2     [ ] Get feedback
M3n4 # Phase 2
O5p6 > Implementation
Q7r8   [ ] Setup
S9t0   [ ] Code
```

### Example 2: Code Block (Properly Closed)

````
A1b2 ## Code Example
C3d4 ```python
def greet(name):
    # This # is not a heading
    return f"Hello, {name}!"

print("→ arrow in string")
```
E5f6 The function above greets users.
````

### Example 3: Mixed Inline Formatting

```
A1b2 **Bold**, *italic*, ~~struck~~, :u[underline], `code`.
C3d4 Colors: :red[error], :green[success], :blue[info].
E5f6 Math: The formula :eq[E=mc^2] — and it costs $10.
G7h8 Links: [docs](https://x.com) and [page](p:I9j0).
I9j0 Mentions: @user:abc123 and @date:2024-01-15.
```

### Example 4: Page-Backed Blocks

```
A1b2 # My Workspace
C3d4 § Project Notes
E5f6 § Meeting Minutes
G7h8 ⊞ Task Tracker
I9j0 Regular paragraph here.
```

To update "Project Notes" title: `u C3d4 = "Project Documentation"`
(Routes to Update Page endpoint internally.)

### Example 5: Table

```
A1b2 !table 3
C3d4   | Name | Status | Notes |
E5f6   | Fix login bug | Done | Shipped v2.1 |
G7h8   | Add dark mode | In progress | Design ready |
```

### Example 6: Column Layout

```
A1b2 !cols 2
C3d4   !col 1
E5f6     # Left Column
G7h8     Some content on the left.
I9j0   !col 2
K1l2     # Right Column
M3n4     Some content on the right.
```

### Example 7: Auto-Grouped Table (in notion_apply)

```
+ parent=A1b2
  | Name | Status | Notes |
  | Task one | Done | Shipped |
  | Task two | In progress | On track |
```

Consecutive pipe rows are auto-wrapped in a `!table N` parent.
Markdown separator rows (`|---|---|---`) are silently dropped.

### Example 8: Database with TSV

```
@dnn 1
@db 8x7y6z5w
@ds 7w6v5u4t
@title Bug Tracker
@cols Title(title),Status(select),Priority(select)

G7h8,Login crash,Open,High
I9j0,Typo in header,Closed,Low
K1l2,Slow query,Open,Medium
```

---

## Token Efficiency

| Scenario | API JSON | DNN | Savings |
|:---------|:---------|:----|:--------|
| 10-block page (edit) | ~2,400 | ~320 | **87%** |
| 10-block page (view) | ~2,400 | ~180 | **92%** |
| 10-row database | ~4,800 | ~400 | **92%** |

---

## Implementation Notes

### Parsing State Machine

````
START → HEADER
HEADER: line matches ^@\w+ → stay HEADER
HEADER: line matches ^[A-Za-z0-9]{4}\s → BLOCK
BLOCK: line contains ``` → CODE
BLOCK: normal line → stay BLOCK
CODE: line is ``` only → BLOCK
CODE: any other line → stay CODE (raw)
````

### Read-Only Mode

In view mode (no IDs), indentation starts at column 0:

```
# Project Tasks
> Phase 1: Research
  [x] Read documentation
  [ ] Write summary
    [ ] Draft outline
> Phase 2: Implementation
  [ ] Set up environment
```

### Marker Precedence

Check in order (first match wins):
1. `>### ` → toggle heading_3
2. `>## ` → toggle heading_2
3. `># ` → toggle heading_1
4. `### ` → heading_3
5. `## ` → heading_2
6. `# ` → heading_1
7. `---` → divider (exact)
8. `[ ] ` or `[x] ` → to_do
9. `N. ` → numbered_list
10. `- ` → bulleted_list
11. `> ` → toggle
12. `| ... |` → table_row (pipe-delimited with trailing `|`)
13. `| ` → quote (pipe, no trailing `|`)
14. `!color ` → colored callout
15. `!type~` → opaque block
16. `!type N` → structural block (cols, col, table)
17. `! ` → default callout
18. `§ ` → child_page
19. `⊞ ` → child_database
20. `→ ` → link_to_page
21. ` ``` ` → code block
22. (else) → paragraph

---

## References

- [Notion API](https://developers.notion.com/reference)
- [Append Block Children](https://developers.notion.com/reference/patch-block-children)
- [Move Page](https://developers.notion.com/reference/post-page-move)
- [Update Page](https://developers.notion.com/reference/patch-page)
- [Query Data Source](https://developers.notion.com/reference/post-database-query)
- [TOON Format](https://github.com/toon-format/spec)
