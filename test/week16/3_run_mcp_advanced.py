"""
Week16: 进阶 MCP Server 的 Client 演示 — 直接运行（可调试）.
连接 mcp_server_advanced，演示复杂 tools / resources / prompts 的调用.
运行: 在项目根目录执行  python tests/week16/run_mcp_advanced.py
需要: pip install "mcp[cli]"
"""
import asyncio
import json
import os
import sys
os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))


def _stdio_server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.week16.mcp_server_advanced"],
        env=os.environ.copy(),
        cwd=PROJECT_ROOT,
    )


def _text(result) -> str:
    if not result.content:
        return ""
    first = result.content[0]
    return first.text if isinstance(first, TextContent) else str(first)


# ---------- 1. 列出所有 tools / resources / prompts ----------

async def run_list_all(session: ClientSession):
    print("=== 1. 列出 Tools / Resources / Prompts ===")
    tools_resp = await session.list_tools()
    print("Tools:", [t.name for t in tools_resp.tools])
    res_resp = await session.list_resources()
    print("Resources:", [r.uri for r in res_resp.resources])
    try:
        prompts_resp = await session.list_prompts()
        print("Prompts:", [p.name for p in prompts_resp.prompts])
    except Exception:
        print("Prompts: (list_prompts 可能不可用)")
    print()


# ---------- 2. 复杂 Tools 调用 ----------

async def run_tool_eval_math(session: ClientSession):
    print("=== 2.1 Tool: eval_math ===")
    r = await session.call_tool("eval_math", arguments={"expression": "2 + 3 * 4"})
    print("  eval_math('2 + 3 * 4') =>", _text(r))
    r = await session.call_tool("eval_math", arguments={"expression": "(10 - 2) / 2"})
    print("  eval_math('(10 - 2) / 2') =>", _text(r))
    print()


async def run_tool_search_docs(session: ClientSession):
    print("=== 2.2 Tool: search_docs ===")
    r = await session.call_tool("search_docs", arguments={"query": "MCP", "limit": 2})
    print("  search_docs('MCP', limit=2) =>")
    print(_text(r)[:400] + "..." if len(_text(r)) > 400 else _text(r))
    print()


async def run_tool_get_weather(session: ClientSession):
    print("=== 2.3 Tool: get_weather ===")
    r = await session.call_tool("get_weather", arguments={"city": "北京", "unit": "celsius"})
    print("  get_weather('北京') =>", _text(r))
    r = await session.call_tool("get_weather", arguments={"city": "Shanghai", "unit": "fahrenheit"})
    print("  get_weather('Shanghai', unit='fahrenheit') =>", _text(r))
    print()


async def run_tool_kv_store(session: ClientSession):
    print("=== 2.4 Tool: kv_store ===")
    r = await session.call_tool("kv_store", arguments={"key": "user", "value": "alice"})
    print("  kv_store('user', 'alice') =>", _text(r))
    r = await session.call_tool("kv_store", arguments={"key": "user"})
    print("  kv_store('user') =>", _text(r))
    print()


async def run_tool_summarize_text(session: ClientSession):
    print("=== 2.5 Tool: summarize_text ===")
    long_text = "这是第一句。这是第二句。这是第三句。这是第四句。这是第五句。"
    r = await session.call_tool("summarize_text", arguments={"text": long_text, "max_sentences": 2})
    print("  summarize_text(..., max_sentences=2) =>", _text(r))
    print()


async def run_tool_batch_calculate(session: ClientSession):
    print("=== 2.6 Tool: batch_calculate ===")
    ops = [{"op": "add", "a": 1, "b": 2}, {"op": "mul", "a": 3, "b": 4}, {"op": "sub", "a": 10, "b": 3}]
    r = await session.call_tool("batch_calculate", arguments={"operations_json": json.dumps(ops)})
    print("  batch_calculate([add(1,2), mul(3,4), sub(10,3)]) =>", _text(r))
    print()


# ---------- 3. Resources 读取 ----------

async def run_resource_doc(session: ClientSession):
    print("=== 3.1 Resource: doc://doc2 ===")
    try:
        from pydantic import AnyUrl
        r = await session.read_resource(AnyUrl("doc://doc2"))
    except Exception as e:
        try:
            r = await session.read_resource("doc://doc2")
        except Exception:
            print("  (read_resource 需要 pydantic.AnyUrl 或当前 SDK 不支持 doc://)", e)
            return
    if r.contents:
        c = r.contents[0]
        text = c.text if isinstance(c, TextContent) else str(c)
        print("  content:", text[:200] + "..." if len(text) > 200 else text)
    print()


async def run_resource_config(session: ClientSession):
    print("=== 3.2 Resource: config://version ===")
    try:
        from pydantic import AnyUrl
        r = await session.read_resource(AnyUrl("config://version"))
    except Exception:
        try:
            r = await session.read_resource("config://version")
        except Exception as e:
            print("  (read_resource 不可用):", e)
            return
    if r.contents:
        c = r.contents[0]
        text = c.text if isinstance(c, TextContent) else str(c)
        print("  content:", text)
    print()


# ---------- 4. Prompt 获取 ----------

async def run_prompt_make_task(session: ClientSession):
    print("=== 4. Prompt: make_task_prompt ===")
    try:
        p = await session.get_prompt("make_task_prompt", arguments={"topic": "什么是 MCP", "style": "简洁"})
        if p.messages:
            msg = p.messages[0]
            content = getattr(msg, "content", None) or str(msg)
            print("  prompt message:", content[:150] if len(str(content)) > 150 else content)
    except Exception as e:
        print("  (get_prompt 可能不可用):", e)
    print()


if __name__ == "__main__":
    async def main():
        async with stdio_client(_stdio_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await run_list_all(session)
                await run_tool_eval_math(session)
                await run_tool_search_docs(session)
                await run_tool_get_weather(session)
                await run_tool_kv_store(session)
                await run_tool_summarize_text(session)
                await run_tool_batch_calculate(session)
                await run_resource_doc(session)
                await run_resource_config(session)
                await run_prompt_make_task(session)

    asyncio.run(main())
