"""
Week16 演示用 MCP Server（stdio 模式，供 client 通过子进程连接）.
运行: python -m tests.week16.mcp_server_demo
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Week16Demo", json_response=True)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a / b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    # 默认 stdio，供 StdioServerParameters 子进程连接
    try:
        mcp.run(transport="stdio")
    except TypeError:
        mcp.run()
