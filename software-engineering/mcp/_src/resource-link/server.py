from fastmcp import FastMCP


mcp = FastMCP(name="ResourceLink")

@mcp.tool
def add(a: int, b: int) -> int:
    """2つの整数を足し合わせるツール"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
