# Binary Resource MCP Server

This is a simple MCP server that listens for connections and sends a binary resource to the client.

## Usage

### Inspector

```sh
pnpm build
npx @modelcontextprotocol/inspector@latest node dist/index.js
```

### Claude

```json
{
  "mcpServers": {
    "binary-resource": {
      "command": "node",
      "args": ["C:/Users/hiroga/GitHub/til/software-engineering/mcp/_src/binary-resource/dist/index.js"]
    }
  }
}
```
