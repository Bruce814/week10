"""
Week16: 模型与 MCP 联动 — 直接运行脚本（可调试）.
将 MCP Server 的 tools 转成 OpenAI 格式给 Silicon，再根据 model 的 tool_calls 回调 MCP 执行.
运行: 在项目根目录执行  python tests/week16/run_silicon_mcp_integration.py
需要: mcp, openai，环境变量 SILICONFLOW_API_KEY
"""
import asyncio
import json
import os
import sys

os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import OpenAI

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
TOOL_CALLING_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _stdio_server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.week16.mcp_server_demo"],
        env=os.environ.copy(),
        cwd=PROJECT_ROOT,
    )


def mcp_tool_to_openai_tool(mcp_tool) -> dict:
    name = mcp_tool.name
    description = mcp_tool.description or ""
    params = getattr(mcp_tool, "inputSchema", None) or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


async def run_fetch_mcp_tools_and_convert():
    """从 MCP Server 拉取 tools 并转成 OpenAI 格式."""
    async with stdio_client(_stdio_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
    openai_tools = [mcp_tool_to_openai_tool(t) for t in tools_resp.tools]
    names = [t["function"]["name"] for t in openai_tools]
    print("=== 1. MCP tools 转 OpenAI 格式 ===")
    print("OpenAI tools names:", names)
    return openai_tools


async def run_llm_uses_mcp_add():
    """用户问加法，模型选 MCP 的 add，在 MCP 上执行."""
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not key:
        print("请设置 SILICONFLOW_API_KEY")
        return

    async with stdio_client(_stdio_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            openai_tools = [mcp_tool_to_openai_tool(t) for t in tools_resp.tools]

    client = OpenAI(api_key=key, base_url=SILICON_BASE_URL)
    messages = [{"role": "user", "content": "请用你手头的工具计算 15 加 27 等于多少？"}]
    response = client.chat.completions.create(
        model=TOOL_CALLING_MODEL,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message
    if not getattr(msg, "tool_calls", None) or not msg.tool_calls:
        print("模型未返回 tool_calls")
        return
    tc = msg.tool_calls[0]
    name, args_str = tc.function.name, tc.function.arguments
    args = json.loads(args_str)
    print("=== 2. 模型与 MCP 联动：LLM 选择工具并在 MCP 执行 ===")
    print(f"  LLM 选择: {name}({args})")

    async with stdio_client(_stdio_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=args)
    content = result.content[0]
    text = content.text if isinstance(content, TextContent) else str(content)
    print("  MCP 执行结果:", text)


if __name__ == "__main__":
    async def main():
        await run_fetch_mcp_tools_and_convert()
        print()
        await run_llm_uses_mcp_add()

    asyncio.run(main())
