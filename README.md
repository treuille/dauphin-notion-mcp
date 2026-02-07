# dauphin-notion-mcp

An MCP server for reading and writing Notion pages and databases from Claude Code.

Uses **DNN (Dauphin Notion Notation)**, a compact text format that compresses Notion's JSON API by 87–92%, so Claude can read and edit large Notion workspaces without blowing through the context window.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

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
        "git+https://github.com/adrienbrault/dauphin-notion-mcp",
        "dauphin-notion-mcp",
        "--token-file",
        "~/.config/dauphin-notion-mcp/token"
      ]
    }
  }
}
```

Restart Claude Code. You should see five new tools: `notion_read`, `notion_apply`, `notion_search`, `notion_check_auth`, and `notion_get_url`. Ask Claude to run `notion_check_auth` to verify the connection.

## What makes this different

**Token efficiency.** Notion's API returns deeply nested JSON — a typical page is 50–100KB. DNN compresses that to 2–5KB of readable text. This means Claude can work with much larger pages and databases without context window issues.

**Parallel operations.** Mutations (add, delete, move, update) execute concurrently with async rate limiting. Batch edits to a page with 20 blocks happen in parallel, not sequentially.

**Full read/write cycle.** Read a page in DNN, edit the text, apply changes back. The format round-trips cleanly.

## Supported block types

| Type | DNN syntax | Read | Write |
|:-----|:-----------|:-----|:------|
| Paragraph | plain text | yes | yes |
| Heading 1/2/3 | `#` `##` `###` | yes | yes |
| Toggle heading | `>#` `>##` `>###` | yes | yes |
| Bulleted list | `- item` | yes | yes |
| Numbered list | `1. item` | yes | yes |
| To-do | `[ ] task` / `[x] done` | yes | yes |
| Toggle | `> content` | yes | yes |
| Quote | `\| content` | yes | yes |
| Callout | `! content` | yes | yes |
| Divider | `---` | yes | yes |
| Code block | ` ``` lang ` | yes | yes |
| Child page | `§ Title` | yes | yes |
| Child database | `⊞ Title` | yes | yes |

### Inline formatting

`**bold**`, `*italic*`, `~~strike~~`, `` `code` ``,
`:u[underline]`, `:red[colored text]`,
`:yellow-background[highlighted]`,
`[link text](url)`, `@p:ID` page mentions,
`@date:2025-01-15` date mentions, `$E=mc^2$` equations.

### Databases

Databases render as compact TSV (tab-separated) tables
with typed columns. Supports:
- Reading rows with optional filter, sort, and limit
- Creating rows (`+row`)
- Updating row properties (`urow`)
- Deleting rows (`xrow`)

## Not yet supported

- **Tables** — Notion tables (not databases) are read-only
- **Synced blocks** — displayed as placeholder, can't create
- **Images/video/files** — displayed as `!image`, `!video`,
  etc.; can't be created via the API
- **Column layouts** — read-only
- **Block equations** — read-only (inline `$math$` works)
- **Database schema changes** — can't add/rename properties

## DNN format specification

See [docs/dnn-spec.md](docs/dnn-spec.md) for the full format
specification including all block types, inline formatting,
mutation commands, error codes, and examples.

## License

Apache 2.0
