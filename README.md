# dauphin-notion-mcp

An MCP server for reading and writing Notion pages and databases from Claude Code.

Uses **DNN (Dauphin Notion Notation)**, a compact text format that compresses Notion's JSON API by 87–92%, so Claude can read and edit large Notion workspaces without blowing through the context window.

## Setup

### 1. Create a Notion integration

1. Go to [notion.so/profile/integrations](https://www.notion.so/profile/integrations)
2. Click **New integration**
3. Give it a name, select your workspace, and set capabilities to **Read + Update + Insert content**
4. Copy the token (starts with `ntn_`)

### 2. Share pages with the integration

In Notion, open each page or database you want accessible, click **...** → **Connections** → add your integration. Child pages inherit access.

### 3. Add to Claude Code

Add this to your Claude Code MCP config (`~/.claude/settings.json` or project `.mcp.json`):

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
        "--token", "ntn_YOUR_TOKEN_HERE"
      ]
    }
  }
}
```

### 4. Enable tools

In Claude Code settings, enable the `notion-mcp` tools:
- `notion_read` — read pages and databases
- `notion_apply` — create, update, move, and delete content
- `notion_search` — find pages and databases by title
- `notion_check_auth` — verify the connection works
- `notion_get_url` — get the Notion URL for a page or block

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
