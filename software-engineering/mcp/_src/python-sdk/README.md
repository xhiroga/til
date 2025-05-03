# MCP Server with Python SDK

NOTE: [create-mcp-server](https://github.com/modelcontextprotocol/create-python-server) はすでに非推奨になっている。

したがって、こちらを参照した。

- https://zenn.dev/taku_sid/articles/20250331_mcp_python?redirected=1

## Debug

```sh
mcp dev server.py
```

## Installation

## Claude Desktop

WSLではなくClaudeを実行しているWindows上で実行する必要がある。

```sh
uv run mcp install server.py --name "Demo by Python SDK"
```

すると、次のように登録される。

```json
 "Demo by Python SDK": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "C:\\Users\\hiroga\\GitHub\\til\\software-engineering\\mcp\\_src\\python-sdk\\server.py"
      ]
    }
```

## 実験

Python SDKを用いたMCPサーバーのインストールは、`mcp[cli]`を用いる方法と`uvx`で起動する方法の2つありそうだ。では、そのままClaude Desktopに設定できるのだろうか？また、`--from` オプションを指定する場合や、PEP508記法でサブディレクトリを指定する方法は有効なのか？

結論から言うと、そのままClaude Desktopに指定する場合は動く。

```json
{
    "mcpServers": {
        "fetch": {
            "command": "uvx",
            "args": [
                "mcp-server-fetch"
            ]
        }
    }
}
```

MCPサーバーが素直にプロジェクトルートにある場合は次のように指定できる。  
（なお、`fetch`は異なる）

```json
{
  "mcpServers": {
    "MiniMax": {
      "command": "uvx",
       "args": [
         "--refresh",
         "--from",
         "git+https://github.com/MiniMax-AI/MiniMax-MCP",
         "minimax-mcp"
      ]
    }
  }
}
```

しかし、次の通り`--from`でサブディレクトリを指定するとエラーになる。

```json
{
    "mcpServers": {
        "fetch": {
            "command": "uvx",
            "args": [
                "--refresh",
                "--from",
                "git+https://github.com/modelcontextprotocol/servers#subdirectory=src/fetch",
                "mcp-server-fetch"
            ]
        }
    }
}
```

エラーの内容は次の通り。

```log
2025-04-24T04:38:51.231Z [fetch] [info] Initializing server...
2025-04-24T04:38:51.286Z [fetch] [info] Server started and connected successfully
2025-04-24T04:38:51.378Z [fetch] [info] Message from client: {"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"claude-ai","version":"0.1.0"}},"jsonrpc":"2.0","id":0}
  × Invalid `--with` requirement
  ╰─▶ Git operation failed
```
