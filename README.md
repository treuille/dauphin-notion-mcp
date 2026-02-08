# dauphin-notion-mcp

This MCP server gives Claude full read/write access to your Notion workspace.

- **Token-efficient** — a typical Notion page is 50-100KB of JSON from the API. This server compresses that to 2-5KB of readable text (87-92% reduction), so Claude can work with large workspaces without blowing through context.
- **Parallel read/write** — mutations execute concurrently with async rate limiting. Batch edits to a page happen in parallel, not one block at a time.
- **Broad block type coverage** — 16 block types, inline formatting, and database CRUD. Not universal yet (tables, synced blocks, and media are read-only).

### Why Notion for agents?

Notion is one of the most flexible tools for structuring personal information — projects, contacts, preferences, notes. That same flexibility makes it equally powerful as agent memory:

- **User preferences & context** — grounding data agents can reference across sessions
- **Contacts & CRM** — structured databases agents can query and update
- **Projects & tasks** — planning surfaces agents can read and write to
- **Knowledge bases** — rich documents agents can search and synthesize

Everything Notion does well for humans, it does just as well for agents.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (or another MCP client)
- A [Notion](https://www.notion.so/) account with an API integration
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

> **Note:** Only tested on Ubuntu so far.

## Setup

### 1. Get a Notion API token

Create an internal integration at [notion.so/profile/integrations](https://www.notion.so/profile/integrations). It needs **Read**, **Update**, and **Insert content** capabilities. You'll get a token starting with `ntn_`.

Then share the pages/databases you want accessible: open each top-level page in Notion, go to **...** → **Connections**, and add your integration. Child pages inherit access.

### 2. Save the token to a file

```bash
mkdir -p ~/.config/dauphin-notion-mcp
echo "ntn_YOUR_TOKEN_HERE" > ~/.config/dauphin-notion-mcp/token
```

### 3. Add to Claude Code

Add this to `~/.claude/settings.json` (or a project `.mcp.json`):

```json
{
  "mcpServers": {
    "notion-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/treuille/dauphin-notion-mcp",
        "dauphin-notion-mcp",
        "--token-file",
        "~/.config/dauphin-notion-mcp/token"
      ]
    }
  }
}
```

Restart Claude Code. You should see five new tools: `notion_read`, `notion_apply`, `notion_search`, `notion_check_auth`, and `notion_get_url`. Ask Claude to run `notion_check_auth` to verify the connection.

## How it works

The server uses **DNN (Dauphin Notion Notation)**, a compact text format designed for LLM consumption. Instead of passing Notion's raw JSON to Claude, DNN uses markdown-like syntax that's both human-readable and token-efficient.

### Supported block types

| Type | DNN syntax |
|:-----|:-----------|
| Paragraph | plain text |
| Heading 1/2/3 | `#` `##` `###` |
| Toggle heading | `>#` `>##` `>###` |
| Bulleted list | `- item` |
| Numbered list | `1. item` |
| To-do | `[ ] task` / `[x] done` |
| Toggle | `> content` |
| Quote | `\| content` |
| Callout | `! content` |
| Divider | `---` |
| Code block | ` ``` lang ` |
| Child page | `§ Title` |
| Child database | `⊞ Title` |

### Inline formatting

`**bold**`, `*italic*`, `~~strike~~`, `` `code` ``,
`:u[underline]`, `:red[colored text]`,
`:yellow-background[highlighted]`,
`[link text](url)`, `@p:ID` page mentions,
`@date:2025-01-15` date mentions, `$E=mc^2$` equations.

### Databases

Databases render as compact TSV tables with typed columns.
Supports reading rows (with filter, sort, limit), creating
rows (`+row`), updating properties (`urow`), and deleting
rows (`xrow`).

## Not yet supported

- **Tables** — Notion tables (not databases) are read-only
- **Synced blocks** — displayed as placeholder, can't create
- **Images/video/files** — displayed as `!image` etc.; can't
  be created via the API
- **Column layouts** — read-only
- **Block equations** — read-only (inline `$math$` works)
- **Database schema changes** — can't add/rename properties

## DNN format specification

See [docs/dnn-spec.md](docs/dnn-spec.md) for the full format
specification including all block types, inline formatting,
mutation commands, error codes, and examples.

## License

Apache 2.0
