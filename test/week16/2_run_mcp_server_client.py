"""
Week16: MCP Server 与 MCP Client — 直接运行脚本（可调试）.
运行: 在项目根目录执行  python tests/week16/run_mcp_server_client.py
需要: pip install "mcp[cli]"
"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# 本包内 server 脚本，stdio 子进程用
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))


def _stdio_server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.week16.mcp_server_demo"],
        env=os.environ.copy(),
        cwd=PROJECT_ROOT,
    )


async def run_server_logic():
    """1. 直接调用 server 里的工具函数（不启动子进程）."""
    from tests.week16.mcp_server_demo import add, multiply, get_greeting
    print("=== 1. MCP Server 工具逻辑（直接调用）===")
    print("add(2, 3) =", add(2, 3))
    print("multiply(3, 4) =", multiply(3, 4))
    print("get_greeting('World') =", get_greeting("World"))
    print()


async def run_client_list_tools():
    """2. Client 连接 Server，列出 tools."""
    server_params = _stdio_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
    names = [t.name for t in tools_resp.tools]
    print("=== 2. MCP Client list_tools ===")
    print("Tools:", names)
    print()


async def run_client_call_tool_add():
    """3. Client 调用 add 工具."""
    server_params = _stdio_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", arguments={"a": 7, "b": 5})
    first = result.content[0]
    text = first.text if isinstance(first, TextContent) else str(first)
    print("=== 3. MCP Client call_tool add(7, 5) ===")
    print("Result:", text)
    print()


async def run_client_call_tool_multiply():
    """4. Client 调用 multiply 工具."""
    server_params = _stdio_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("multiply", arguments={"a": 6, "b": 7})
    first = result.content[0]
    text = first.text if isinstance(first, TextContent) else str(first)
    print("=== 4. MCP Client call_tool multiply(6, 7) ===")
    print("Result:", text)
    print()


async def run_client_list_resources():
    """5. Client 列出 resources."""
    server_params = _stdio_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resources_resp = await session.list_resources()
    print("=== 5. MCP Client list_resources ===")
    print("Resources:", [r.uri for r in resources_resp.resources])
    print()


if __name__ == "__main__":
    async def main():
        await run_server_logic()
        await run_client_list_tools()
        await run_client_call_tool_add()
        await run_client_call_tool_multiply()
        await run_client_list_resources()

    asyncio.run(main())
