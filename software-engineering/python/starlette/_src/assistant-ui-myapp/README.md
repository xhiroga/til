# assistant-ui with MCP hosted by FastAPI

## NOT WORKS!!! /api/chat does not work because of hosting static files

```sh
pnpm build
uv run main.py
```

## Debug

```sh
uv run main.py
curl http://localhost:8000/mcp # MUST return {"jsonrpc": "2.0", ...}
npx @modelcontextprotocol/inspector@latest
```
