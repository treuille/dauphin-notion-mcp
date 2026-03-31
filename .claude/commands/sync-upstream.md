Sync notion_mcp.py from the upstream repo at ../dauphin into this project.

## Background

This repo is a standalone Notion MCP server extracted from the ../dauphin monorepo. The only functional difference is **secrets management**: upstream discovers a `secrets/notion_token` file by walking directories, while this project takes `--token-file <path>` as a required CLI argument. Everything else should match upstream.

## Steps

### 1. Copy source and test files

Copy both files from upstream, overwriting the local versions:
- `../dauphin/src/notion_mcp.py` → `src/notion_mcp.py`
- `../dauphin/tests/test_notion_mcp.py` → `tests/test_notion_mcp.py`

### 2. Restore --token-file secrets management

After copying, the file will have upstream's secrets-dir approach. Replace it with this project's `--token-file` approach. There are four specific changes:

**a) Module docstring** — Change:
```
Credentials: Reads token from secrets/notion_token file.
```
to:
```
Token: Passed via --token-file <path> CLI argument at startup.
```

**b) Replace `_SERVER_VERSION` block** — The upstream version uses `subprocess` + `git rev-parse` to get a git hash. Replace the entire `_SERVER_VERSION` block (and remove the `import subprocess` line) with:
```python
# Server version from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import version as _pkg_version
    _SERVER_VERSION = f"v{_pkg_version('dauphin-notion-mcp')}"
except Exception:
    _SERVER_VERSION = "unknown"
```

**c) Remove `_find_secrets_dir()` and `_load_notion_token()`** — Delete both functions entirely (they're between the `_notion_token` global and `_get_token()`).

**d) Replace `_get_token()`** — Replace the upstream version (which calls `_load_notion_token()`) with:
```python
def _get_token() -> str:
    """Get the Notion token (set via --token-file CLI arg)."""
    if _notion_token is None:
        raise RuntimeError(
            "No Notion token. Pass --token-file <path> on the command line."
        )
    return _notion_token
```

**e) Replace token loading in `main()`** — In the `main()` function, add `--token-file` as a required argparse argument, and replace the `_load_notion_token()` call with file-reading logic:
```python
parser.add_argument(
    "--token-file",
    required=True,
    help="Path to file containing Notion API token"
)
```
And for the token loading block (replacing the try/except that calls `_load_notion_token()`):
```python
global _notion_token
token_path = Path(args.token_file).expanduser()
if not token_path.exists():
    raise SystemExit(f"Token file not found: {token_path}")
_notion_token = token_path.read_text().strip()
if not _notion_token:
    raise SystemExit(f"Token file is empty: {token_path}")
logger.info(f"Loaded Notion token from {token_path}")
```

### 3. Verify test file

Check that `tests/test_notion_mcp.py` does not import `_find_secrets_dir` or `_load_notion_token` (which were removed in step 2). If it does, remove those imports and any tests that use them.

### 4. Bump version

Bump the patch version in `pyproject.toml` (e.g., `0.1.0` → `0.1.1`). This version is used by `_SERVER_VERSION` at runtime and shown in `notion_check_auth` output.

### 5. Run tests

```
uv run python -m pytest tests/test_notion_mcp.py -x -q
```
All tests should pass. Integration tests being skipped is expected.

### 6. Update the changelog

Add a new date-indexed entry to the Changelog section of README.md. Look at `git diff` to understand what changed. Keep the same style as existing entries. Do NOT reference the upstream repo or where changes came from — just describe what changed.

### 7. Commit and push

Stage changed files (README.md, src/notion_mcp.py, tests/test_notion_mcp.py — and pyproject.toml only if it changed) and commit. Do NOT include any Co-Authored-By lines, attribution, or other ads in commit messages. Do NOT reference the upstream repo in commit messages — just describe the changes. Then push to origin.
