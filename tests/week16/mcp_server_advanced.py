"""
Week16 进阶 MCP Server（FastMCP）— 更复杂的 tools / resources / prompts.
包含：数学求值、模拟文档搜索、天气、键值存储、文本摘要、批量计算、文档/配置资源、提示词模板.
运行: python -m tests.week16.mcp_server_advanced
"""
import json
import re,os
from typing import Optional
os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Week16Advanced", json_response=True)

# ---------- 内存键值存储（供 kv_store 使用）----------
_KV_STORE: dict[str, str] = {}

# ---------- 模拟文档库（供 search_docs / doc 资源使用）----------
_DOCS = [
    {"id": "doc1", "title": "Python 入门", "content": "Python 是一种解释型、面向对象的高级编程语言。语法简洁，适合初学者。"},
    {"id": "doc2", "title": "MCP 协议简介", "content": "Model Context Protocol (MCP) 用于为 LLM 提供标准化上下文，支持 tools、resources、prompts。"},
    {"id": "doc3", "title": "FastMCP 使用", "content": "FastMCP 通过装饰器 @mcp.tool、@mcp.resource、@mcp.prompt 快速暴露能力。"},
    {"id": "doc4", "title": "API 设计规范", "content": "RESTful API 设计应遵循资源导向、统一接口、无状态等原则。"},
]

# ---------- 模拟配置（供 config 资源使用）----------
_CONFIG = {
    "app_name": "Week16 MCP Demo",
    "version": "1.0",
    "features": "tools,resources,prompts",
}


# ========== Tools（复杂工具）==========

@mcp.tool()
def eval_math(expression: str) -> str:
    """
    安全地计算数学表达式（仅允许数字与 + - * / ( ) 及空格）。
    例如: "2 + 3 * 4" -> 14, "(1+2)*3" -> 9.
    """
    expr = expression.strip()
    if not re.match(r"^[\d\s+\-*/().]+$", expr):
        return "Error: 表达式仅允许数字与 + - * / ( )"
    try:
        result = eval(expr)
        return str(result) if isinstance(result, (int, float)) else str(result)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def search_docs(query: str, limit: int = 5) -> str:
    """
    在模拟文档库中按关键词搜索，返回匹配的文档片段列表（JSON 字符串）。
    query: 搜索关键词；limit: 最多返回条数，默认 5。
    """
    limit = max(1, min(limit, 20))
    query_lower = query.lower()
    hits = []
    for d in _DOCS:
        if query_lower in d["title"].lower() or query_lower in d["content"].lower():
            hits.append({"id": d["id"], "title": d["title"], "snippet": d["content"][:80] + "..."})
            if len(hits) >= limit:
                break
    return json.dumps(hits, ensure_ascii=False, indent=2)


@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    查询指定城市的模拟天气（演示用，返回固定格式的模拟数据）。
    city: 城市名；unit: 温度单位 "celsius" 或 "fahrenheit"，默认 celsius。
    """
    # 模拟：根据城市名生成伪随机但确定的结果
    temp_c = (hash(city) % 25) + 10
    if unit == "fahrenheit":
        temp = temp_c * 9 / 5 + 32
        u = "°F"
    else:
        temp = temp_c
        u = "°C"
    conditions = ["晴", "多云", "阴", "小雨"][hash(city) % 4]
    return f"{city}: {conditions}, {temp:.1f}{u}"


@mcp.tool()
def kv_store(key: str, value: Optional[str] = None) -> str:
    """
    键值存储：若只传 key 则读取；若传 key 和 value 则写入并返回 'OK'。
    key: 键名；value: 可选，要写入的值。
    """
    if value is not None:
        _KV_STORE[key] = value
        return "OK"
    return _KV_STORE.get(key, "<not_found>")


@mcp.tool()
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    对给定文本做简单摘要：按句号/问号/感叹号分句，取前 max_sentences 句并拼接。
    text: 原文；max_sentences: 最多保留句数，默认 3。
    """
    max_sentences = max(1, min(max_sentences, 10))
    sentences = re.split(r"[。？！.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()][:max_sentences]
    return "。".join(sentences) + "。"


@mcp.tool()
def batch_calculate(operations_json: str) -> str:
    """
    批量计算。operations_json 为 JSON 数组，每项为 {"op":"add"|"mul"|"sub","a":n,"b":n}。
    返回各结果组成的 JSON 数组字符串。
    """
    try:
        ops = json.loads(operations_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": str(e)})
    if not isinstance(ops, list):
        return json.dumps({"error": "operations 应为数组"})
    results = []
    for item in ops:
        if not isinstance(item, dict):
            results.append(None)
            continue
        op, a, b = item.get("op"), item.get("a"), item.get("b")
        if op == "add":
            results.append(a + b)
        elif op == "mul":
            results.append(a * b)
        elif op == "sub":
            results.append(a - b)
        else:
            results.append(None)
    return json.dumps(results)


# ========== Resources ==========

@mcp.resource("doc://{doc_id}")
def get_doc(doc_id: str) -> str:
    """根据文档 ID 返回文档内容（模拟文档库）。"""
    for d in _DOCS:
        if d["id"] == doc_id:
            return json.dumps(d, ensure_ascii=False, indent=2)
    return json.dumps({"error": "doc not found", "id": doc_id})


@mcp.resource("config://{key}")
def get_config(key: str) -> str:
    """根据配置 key 返回配置值。"""
    if key in _CONFIG:
        return json.dumps({"key": key, "value": _CONFIG[key]}, ensure_ascii=False)
    return json.dumps({"error": "key not found", "key": key})


# ========== Prompts ==========

@mcp.prompt()
def make_task_prompt(topic: str, style: str = "简洁") -> str:
    """
    生成一个面向 LLM 的任务说明提示词。
    topic: 任务主题；style: 风格，如 简洁/详细/专业，默认 简洁。
    """
    styles = {"简洁": "请用简洁的语言", "详细": "请详细展开说明", "专业": "请使用专业术语"}
    prefix = styles.get(style, styles["简洁"])
    return f"{prefix}回答以下主题：{topic}。\n请直接给出答案，不要复述题目。"


if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except TypeError:
        mcp.run()
