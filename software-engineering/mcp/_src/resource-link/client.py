import asyncio
from fastmcp import Client

async def main():
    """メイン"""

    # MCPクライアントの準備
    client = Client("http://127.0.0.1:8000/mcp")

    async with client:
        # MCPサーバで提供されているツール一覧の取得
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        # addツールを呼び出す
        result = await client.call_tool("add", {"a": 3, "b": 5})
        print(f"add(3,5) = {result}")

if __name__ == "__main__":
    asyncio.run(main())
