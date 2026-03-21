"""
Week16: Plan-and-Execute Agent — 连接进阶 MCP Server，先规划再执行.
规划阶段产出步骤列表（每步指定 tool + arguments），执行阶段依次调用 MCP 工具并收集结果.
支持：预定义场景、以及可选的使用 LLM 生成规划.
运行: 在项目根目录执行  python tests/week16/plan_execute_agent.py
需要: pip install "mcp[cli]"，可选 SILICONFLOW_API_KEY（用于 LLM 规划）.
"""
import asyncio
import json
import os
import re
import sys

os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
PLACEHOLDER_PREV = "__PREV__"


def _stdio_server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.week16.mcp_server_advanced"],
        env=os.environ.copy(),
        cwd=PROJECT_ROOT,
    )


def _result_text(call_result) -> str:
    if not call_result.content:
        return ""
    first = call_result.content[0]
    return first.text if isinstance(first, TextContent) else str(first)


def _inject_prev(arguments: dict, prev_result: str) -> dict:
    out = {}
    for k, v in arguments.items():
        if v == PLACEHOLDER_PREV or (isinstance(v, str) and v.strip() == PLACEHOLDER_PREV):
            out[k] = prev_result
        elif isinstance(v, dict):
            out[k] = _inject_prev(v, prev_result)
        else:
            out[k] = v
    return out


async def run_plan(session: ClientSession, steps: list[dict]) -> list[str]:
    """
    按顺序执行规划步骤，每步调用 MCP tool；arguments 中可用 PLACEHOLDER_PREV 表示使用上一步结果.
    返回每步的输出字符串列表.
    """
    results: list[str] = []
    prev = ""
    for i, step in enumerate(steps):
        tool_name = step.get("tool")
        if not tool_name:
            results.append(f"[skip step {i+1}: no tool]")
            continue
            ### human in loop
        arguments = step.get("arguments", {})
        arguments = _inject_prev(arguments, prev)
        try:
            r = await session.call_tool(tool_name, arguments=arguments)
            prev = _result_text(r)
            results.append(prev)
        except Exception as e:
            prev = f"Error: {e}"
            results.append(prev)
    return results


# ---------- 预定义场景（Plan 步骤列表）----------

SCENARIO_WEATHER_SAVE = {
    "name": "查北京天气并存入 KV",
    "description": "调用 get_weather 查北京天气，再把结果用 kv_store 存到 key bj_weather",
    "steps": [
        {"tool": "get_weather", "arguments": {"city": "北京", "unit": "celsius"}},
        {"tool": "kv_store", "arguments": {"key": "bj_weather", "value": PLACEHOLDER_PREV}},
    ],
}

SCENARIO_CALC_CHAIN = {
    "name": "链式计算并存结果",
    "description": "先算 (10+2)*3，再与 5 相加，最后把结果存到 kv calc_result",
    "steps": [
        {"tool": "eval_math", "arguments": {"expression": "(10+2)*3"}},
        {"tool": "batch_calculate", "arguments": {"operations_json": json.dumps([{"op": "add", "a": PLACEHOLDER_PREV, "b": 5}])}},
    ],
}

SCENARIO_CALC_CHAIN_V2 = {
    "name": "链式计算并存结果",
    "description": "计算 (10+2)*3+5 得到 41，结果存 kv calc_result",
    "steps": [
        {"tool": "eval_math", "arguments": {"expression": "(10+2)*3+5"}},
        {"tool": "kv_store", "arguments": {"key": "calc_result", "value": PLACEHOLDER_PREV}},
    ],
}

SCENARIO_SEARCH_SUMMARIZE_SAVE = {
    "name": "搜索文档并摘要存储",
    "description": "搜索关键词 MCP，取前 2 条；对结果做最多 2 句摘要，存到 kv mcp_summary",
    "steps": [
        {"tool": "search_docs", "arguments": {"query": "MCP", "limit": 2}},
        {"tool": "summarize_text", "arguments": {"text": PLACEHOLDER_PREV, "max_sentences": 2}},
        {"tool": "kv_store", "arguments": {"key": "mcp_summary", "value": PLACEHOLDER_PREV}},
    ],
}

SCENARIO_MULTI_CITY_WEATHER = {
    "name": "多城市天气并汇总存储",
    "description": "查北京、上海天气，再摘要后存 kv multi_weather",
    "steps": [
        {"tool": "get_weather", "arguments": {"city": "北京", "unit": "celsius"}},
        {"tool": "get_weather", "arguments": {"city": "上海", "unit": "celsius"}},
        # 上一步会覆盖 prev，这里只保存了上海；若要保存两段需合并。简化为：只把上海结果存起来，或先合并再存
        {"tool": "kv_store", "arguments": {"key": "shanghai_weather", "value": PLACEHOLDER_PREV}},
    ],
}

SCENARIO_READ_DOC_SUMMARIZE = {
    "name": "搜索并摘要",
    "description": "搜索关键词 API，对结果做 1 句摘要",
    "steps": [
        {"tool": "search_docs", "arguments": {"query": "API", "limit": 1}},
        {"tool": "summarize_text", "arguments": {"text": PLACEHOLDER_PREV, "max_sentences": 1}},
    ],
}

# 所有预定义场景
PREDEFINED_SCENARIOS = [
    SCENARIO_WEATHER_SAVE,
    SCENARIO_CALC_CHAIN_V2,
    SCENARIO_SEARCH_SUMMARIZE_SAVE,
    SCENARIO_MULTI_CITY_WEATHER,
    SCENARIO_READ_DOC_SUMMARIZE,
]


async def run_scenario(session: ClientSession, scenario: dict) -> list[str]:
    """执行一个预定义场景，返回每步结果."""
    steps = scenario["steps"]
    return await run_plan(session, steps)


def _parse_llm_plan(text: str) -> list[dict] | None:
    """从 LLM 返回的文本中解析出步骤列表（JSON 数组）. 兼容 ```json ... ``` 包裹."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    steps = []
    for item in data:
        if isinstance(item, dict) and item.get("tool"):
            steps.append({"tool": item["tool"], "arguments": item.get("arguments", {})})
    return steps if steps else None


async def run_with_llm_plan(session: ClientSession, user_query: str, api_key: str) -> tuple[list[dict] | None, list[str]]:
    """
    使用 LLM 根据用户 query 生成规划步骤，再执行.
    返回 (plan_steps, step_results). 若解析失败则 plan_steps 为 None.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None, ["Error: 需要 openai 包"]
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
    tools_desc = """
- get_weather(city, unit?) 查询城市天气
- kv_store(key, value?) 读/写键值，value 为空则读
- eval_math(expression) 计算数学表达式
- search_docs(query, limit?) 搜索文档
- summarize_text(text, max_sentences?) 文本摘要
- batch_calculate(operations_json) 批量计算，JSON 数组每项 {"op":"add"|"mul"|"sub","a":n,"b":n}
请根据用户需求，输出一个 JSON 数组，每项为 {"tool": "工具名", "arguments": {...}}。
若某步要用上一步的结果作为参数，该参数值写 "__PREV__"。只输出 JSON，不要其他解释。
"""
    prompt = f"{tools_desc}\n\n用户需求：{user_query}"
    try:
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return None, [f"LLM 调用失败: {e}"]
    plan = _parse_llm_plan(content)
    print(f'模型plan的结果为：{plan}')
    if not plan:
        return None, [f"无法解析规划，LLM 返回: {content[:200]}..."]
    results = await run_plan(session, plan)
    return plan, results


if __name__ == "__main__":
    async def main():
        async with stdio_client(_stdio_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print("========== Plan-and-Execute Agent（连接 MCP Server）==========\n")

                # ---------- 预定义场景 ----------
                for scenario in PREDEFINED_SCENARIOS:
                    print(f"--- 场景: {scenario['name']} ---")
                    print(f"描述: {scenario['description']}")
                    results = await run_scenario(session, scenario)
                    for i, r in enumerate(results):
                        print(f"  步骤 {i+1} 结果: {r[:120]}{'...' if len(r) > 120 else ''}")
                    print()

                # ---------- 可选：LLM 规划 ----------
                key = (os.environ.get("SILICONFLOW_API_KEY") or "").strip()
                if key:
                    print("--- 场景: LLM 规划 — 「查上海天气并存到 kv 的 sh_weather」---")
                    plan, results = await run_with_llm_plan(
                        session,
                        "查上海天气，然后把查询结果存到 kv_store，key 为 sh_weather。",
                        key,
                    )
                    if plan:
                        print("  规划步骤:", json.dumps(plan, ensure_ascii=False, indent=2))
                        for i, r in enumerate(results):
                            print(f"  步骤 {i+1} 结果: {r[:120]}{'...' if len(r) > 120 else ''}")
                    else:
                        print("  结果:", results[0] if results else "")
                    print()
                else:
                    print("--- 跳过 LLM 规划场景（需设置 SILICONFLOW_API_KEY）---\n")

    asyncio.run(main())
