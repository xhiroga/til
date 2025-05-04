import click
import uvicorn
from mcp.server.fastmcp import FastMCP


@click.command()
@click.option("--port", default=8000, help="Port to listen on")
def main(port: int) -> None:
    mcp = FastMCP("Echo", port=port, host="localhost")

    @mcp.resource("echo://{message}")
    def echo_resource(message: str) -> str:
        """Echo a message as a resource"""
        return f"Resource echo: {message}"

    @mcp.tool()
    def echo_tool(message: str) -> str:
        """Echo a message as a tool"""
        return f"Tool echo: {message}"

    @mcp.prompt()
    def echo_prompt(message: str) -> str:
        """Create an echo prompt"""
        return f"Please process this message: {message}"

    uvicorn.run(mcp.sse_app(), host="localhost", port=port)


if __name__ == "__main__":
    main()
