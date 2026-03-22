"""
Week18: Deep Search Agent（联网检索 + 多查询聚合）

运行: python tests/week18/5_deepsearch_agent.py
依赖:
  pip install duckduckgo-search langchain-openai langgraph
  可选: pip install tavily-python  并设置 TAVILY_API_KEY（结果通常更稳）

Deep Search 流程（LangGraph）:
  1) decompose  — LLM 把用户问题拆成 2~4 条搜索查询
  2) search     — 对每条查询调用联网搜索，收集标题/摘要/链接
  3) synthesize — LLM 基于检索结果写中文回答，并列出参考链接

另附: 使用 DeepAgents + 同一搜索工具的精简版（单 Agent 自主搜与写）。
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-kltrvjvmttyvoqjfbccwztkcmcmogrffassqssctsmyctsrm",
)
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def get_llm(temperature: float = 0.2, max_tokens: int = 2048) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# =============================================================================
# 联网搜索实现（DuckDuckGo 免费 / Tavily 可选）
# =============================================================================


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("请安装: pip install duckduckgo-search") from None

    results = []
    ddgs = DDGS()
    try:
        for item in ddgs.text(query, max_results=max_results):
            print(item)
            results.append(
                {
                    "title": item.get("title", ""),
                    "href": item.get("href", ""),
                    "body": item.get("body", "")[:500],
                }
            )
            if len(results) >= max_results:
                break
    finally:
        try:
            ddgs.close()
        except Exception:
            pass
    return results


def _search_tavily(query: str, max_results: int = 5) -> list[dict]:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return []
    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    client = TavilyClient(api_key=key)
    resp = client.search(query, max_results=max_results)
    out = []
    for r in resp.get("results", [])[:max_results]:
        out.append(
            {
                "title": r.get("title", ""),
                "href": r.get("url", ""),
                "body": (r.get("content") or "")[:500],
            }
        )
    return out


def run_web_search(query: str, max_results: int = 5) -> list[dict]:
    """优先 Tavily（若配置），否则 DuckDuckGo。"""
    t = _search_tavily(query, max_results)
    if t:
        return t
    return _search_duckduckgo(query, max_results)


@tool
def internet_search(query: str, max_results: int = 5) -> str:
    """联网搜索。query 为搜索关键词或一句话；返回若干条结果的标题、链接与摘要文本。"""
    try:
        rows = run_web_search(query, max_results=max_results)
    except Exception as e:
        return f"搜索失败: {e}"
    if not rows:
        return "未找到结果（可能被限流，请稍后重试或配置 TAVILY_API_KEY）。"
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(f"[{i}] {r['title']}\nURL: {r['href']}\n{r['body']}\n")
    return "\n".join(lines)


# =============================================================================
# LangGraph: Deep Search（多查询 + 聚合）
# =============================================================================


class DeepSearchState(TypedDict):
    question: str
    sub_queries: list[str]
    raw_context: str
    answer: str


def node_decompose(state: DeepSearchState) -> DeepSearchState:
    llm = get_llm(0.2)
    msg = llm.invoke(
        [
            SystemMessage(
                content="将用户问题拆成 2~4 条适合搜索引擎的短查询（中英文均可）。"
                "只输出 JSON 数组，例如 [\"查询1\",\"查询2\"]。不要其它文字。"
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    text = msg.content.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            sub = [str(x).strip() for x in arr if str(x).strip()][:4]
        else:
            sub = [state["question"]]
    except json.JSONDecodeError:
        sub = [state["question"]]
    if not sub:
        sub = [state["question"]]
    return {"sub_queries": sub}


def node_search(state: DeepSearchState) -> DeepSearchState:
    chunks = []
    for q in state["sub_queries"]:
        try:
            rows = run_web_search(q, max_results=4)
        except Exception as e:
            chunks.append(f"## Query: {q}\n(错误: {e})\n")
            continue
        chunks.append(f"## Query: {q}\n")
        for i, r in enumerate(rows, 1):
            chunks.append(f"### {i}. {r['title']}\n- URL: {r['href']}\n{r['body']}\n")
    ctx = "\n".join(chunks)
    if len(ctx) > 12000:
        ctx = ctx[:12000] + "\n...(truncated)"
    return {"raw_context": ctx}


def node_synthesize(state: DeepSearchState) -> DeepSearchState:
    llm = get_llm(0.3)
    msg = llm.invoke(
        [
            SystemMessage(
                content="你是研究助理。根据下方检索摘要回答问题：\n"
                "- 用中文作答，条理清晰\n"
                "- 若摘要不足以回答，请说明并给出建议\n"
                "- 文末用「参考链接」列出用到的 URL（去重，最多 8 条）"
            ),
            HumanMessage(
                content=f"问题:\n{state['question']}\n\n--- 检索摘要 ---\n{state['raw_context']}"
            ),
        ]
    )
    return {"answer": msg.content}


def build_deepsearch_graph():
    g = StateGraph(DeepSearchState)
    g.add_node("decompose", node_decompose)
    g.add_node("search", node_search)
    g.add_node("synthesize", node_synthesize)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "search")
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def demo_langgraph_deepsearch():
    print("\n" + "=" * 60)
    print("  [1] LangGraph DeepSearch: decompose -> search -> synthesize")
    print("=" * 60)
    app = build_deepsearch_graph()
    q = "LangGraph 1.0 和 LangChain 1.0 有什么关系？各适合什么场景？"
    out = app.invoke(
        {
            "question": q,
            "sub_queries": [],
            "raw_context": "",
            "answer": "",
        }
    )
    print("Sub-queries:", out["sub_queries"])
    print("--- Answer ---\n", out["answer"][:2500])


# =============================================================================
# DeepAgents: 单 Agent + internet_search 工具
# =============================================================================


def demo_deepagents_search():
    print("\n" + "=" * 60)
    print("  [2] DeepAgents + internet_search tool")
    print("=" * 60)
    try:
        from deepagents import create_deep_agent
    except ImportError:
        print("跳过: pip install deepagents")
        return

    llm = get_llm(0.2)
    instructions = """You are a web research assistant.
Use internet_search to find up-to-date facts. You may call it multiple times with different queries.
Write the final answer in Chinese, with a short bullet list of sources (titles + URLs) at the end."""

    agent = create_deep_agent(
        model=llm,
        tools=[internet_search],
        system_prompt=instructions,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "用中文简要说明：2024-2025 年多智能体框架（如 LangGraph、CrewAI）各有什么特点？请联网搜索后回答。",
                }
            ]
        }
    )
    msgs = result.get("messages", [])
    if msgs:
        print(getattr(msgs[-1], "content", str(msgs[-1]))[:3000])


if __name__ == "__main__":
    print("=" * 60)
    print("  Week18: Deep Search Agent (real web search)")
    print("=" * 60)
    if os.environ.get("TAVILY_API_KEY"):
        print("  Using: Tavily API")
    else:
        print("  Using: DuckDuckGo (install: pip install duckduckgo-search)")
        print("  Optional: TAVILY_API_KEY + pip install tavily-python for more stable results")

    try:
        demo_langgraph_deepsearch()
        demo_deepagents_search()
    except Exception as e:
        print("Error:", e)
        if "duckduckgo" in str(e).lower() or "ImportError" in type(e).__name__:
            print("Try: pip install duckduckgo-search")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
