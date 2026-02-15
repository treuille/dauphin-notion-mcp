# dauphin-notion-mcp

This MCP server gives Claude full read/write access to your Notion workspace.

- **Token-efficient** — a typical Notion page is 50-100KB of JSON from the API. This server compresses that to 2-5KB of readable text (87-92% reduction), so Claude can work with large workspaces without blowing through context.
- **Parallel read/write** — mutations execute concurrently with async rate limiting. Batch edits to a page happen in parallel, not one block at a time.
- **Broad block type coverage** — 16 block types, inline formatting, and database CRUD. Not universal yet (tables, synced blocks, and media are read-only).

## How it works

Doesn't Notion basically feel like fancy markdown? Well, that's the entire idea here. When you read a Notion page through this server, you get back something that looks like this:

```
R4kQ # Potential names for my cat
t9Xm > Serious contenders
pL3n   - **Gerald**
vB8s     - Very distinguished
J2wE   - **Margaret** :red[(vetoed @date:2026-02-14)]
hA3z > Backups
dR5v   - **Dr. Philip Hoffmann III**
wU6j # Potential names for my newborn
bT1y - :blue[Mr. Beans] (current frontrunner)
xF4p   - Pros: already responds to it
gN9s   - Cons: none
kD2r - Mittens
eP5m   - *Does this sound too much like a cat?*
gI7N   - See @p:R4kQ
```

There are many small differences from markdown — `>` is a toggle (foldable section), `@p:R4kQ` is a page mention, and [following Streamlit](https://docs.streamlit.io/develop/api-reference/text/st.markdown), `:red[...]` is colored text — but the format is designed to be immediately readable without learning anything new. For the full specification covering all block types, inline formatting, mutation commands, and error codes, see [docs/dnn-spec.md](docs/dnn-spec.md).

One big departure from markdown is those four-character codes on the left (`R4kQ`, `t9Xm`, ...). Each one is a token-efficient reference to a Notion block. The MCP server manages these IDs across reads and writes so that edits target the correct blocks.

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
    "dauphin-notion-mcp": {
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

## Not yet supported

- **Tables** — Notion tables (not databases) are read-only
- **Synced blocks** — displayed as placeholder, can't create
- **Images/video/files** — displayed as `!image` etc.; can't
  be created via the API
- **Column layouts** — read-only
- **Block equations** — read-only (inline `$math$` works)
- **Database schema changes** — can't add/rename properties

## Changelog

### 2026-02-14

Major new features:

- **Better searching and filtering over databases** — `notion_read` now accepts `filter`, `sort`, and `columns` parameters. The filter DSL supports comparison operators (`=`, `!=`, `~`, `!~`, `<`, `>`, `<=`, `>=`), unary checks (`?` empty, `!?` not empty), boolean logic (`&`, `|`), parenthesized grouping, quoted property names, and relative dates (`-14d`). Sort takes compact specs like `Due desc, Status asc`. Columns selects which properties to return (`Name, Status, Due`). Schema-aware compilation validates property names and coerces types.
- **Parallel reads** — each entry in the `pages` list can now override `depth`, `limit`, `filter`, `sort`, and `columns` independently, enabling batch reads that mix pages and filtered databases in a single call.
- **More block type coverage** — database mentions, `link_mention` (rich URL embeds), `link_preview` (integration embeds), and `template_mention` types are now rendered instead of silently dropped. Annotations (bold, italic, color, etc.) are applied to mention output. Page and row icons (emoji or external URL) are included in DNN output.
- **Better move semantics** — moving blocks that contain Notion-hosted files (images, videos, PDFs) now re-uploads the file via the File Upload API to get a permanent reference. Move commands pre-flight check whether a block can be moved, returning clear errors for synced blocks, tables, breadcrumbs, and other non-portable types. Consecutive moves to the same parent execute sequentially with auto-chaining to preserve script order. `m X -> after=Y` without `parent=` now returns a helpful error instead of silently failing.
- **Renamed MCP config key** — the recommended key in `.mcp.json` / `settings.json` is now `dauphin-notion-mcp` (was `notion-mcp`), to avoid conflicts with other Notion MCP servers.

## License

Apache 2.0
