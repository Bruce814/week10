"""
Week18: 多 Agent 协作 — 基于 DeepAgents (LangChain 官方 deep agent 框架)

运行: python tests/week18/1_deepagents_multi_agent.py
依赖: pip install deepagents langchain-openai

DeepAgents 特点:
  - 内置规划 (write_todos)、虚拟文件系统、子 Agent 委派 (task 工具)
  - 主 Agent 通过 task 工具启动「隔离上下文」的子专家，适合复杂长任务

本文件演示:
  1) 主协调者 + 多个 SubAgent (研究 / 撰写)
  2) 主协调者 + 代码审查子 Agent (与主 Agent 共享工具时可传 tools)

环境变量同 0_langchain_multi_agent.py (SILICONFLOW_*).
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-kltrvjvmttyvoqjfbccwztkcmcmogrffassqssctsmyctsrm",
)
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=0.2,
        max_tokens=2048,
    )


def demo_subagents_research_write():
    """主 Agent 通过内置 task 工具委派子 Agent: 先研究要点再撰写。"""
    from deepagents import create_deep_agent

    llm = get_llm()

    subagents = [
        {
            "name": "research-specialist",
            "description": (
                "Use for gathering structured facts: given a topic, output 4-8 bullet points in Chinese. "
                "No long prose; bullets only."
            ),
            "system_prompt": (
                "You are a research sub-agent. Given a task from the coordinator, output concise Chinese bullet points. "
                "Do not write a full article. End when bullets are done."
            ),
        },
        {
            "name": "writer-specialist",
            "description": (
                "Use for turning outlines or bullet notes into a short polished paragraph in Chinese."
            ),
            "system_prompt": (
                "You are a writing sub-agent. Turn the coordinator's notes into one coherent Chinese paragraph. "
                "Do not add unrelated facts."
            ),
        },
    ]

    system_prompt = """You are the coordinator of a multi-agent team.
For user requests that need both facts and prose:
1) Use the task tool with subagent_type research-specialist to collect bullets.
2) Then use task tool with subagent_type writer-specialist to turn bullets into a short article.
3) Finally give the user a brief summary in Chinese.

If the user only asks a trivial question, answer directly without subagents.
Always respond to the user in Chinese at the end."""

    agent = create_deep_agent(
        model=llm,
        tools=[],
        system_prompt=system_prompt,
        subagents=subagents,
    )

    user_msg = (
        "请协作完成：主题是「边缘计算在物联网中的应用」。"
        "先让研究专员列要点，再让撰写专员写成一段短文。"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    print(result)
    msgs = result.get("messages", [])
    if msgs:
        print(getattr(msgs[-1], "content", str(msgs[-1])))
    return result


def demo_subagents_with_tool():
    """子 Agent 带专用工具: 主 Agent + 计算器专家 (子 Agent 独享算术工具)."""
    from deepagents import create_deep_agent
    from langchain_core.tools import tool

    llm = get_llm()

    @tool
    def calc(expression: str) -> str:
        """Evaluate arithmetic expression with digits and +-*/()."""
        import re

        expr = re.sub(r"[^0-9+\-*/().]", "", expression)
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"error:{e}"

    subagents = [
        {
            "name": "math-specialist",
            "description": "Use when the user needs exact numeric calculation from an expression.",
            "system_prompt": "You have calc(). Parse the user's numbers into one expression, call calc once, return the number clearly.",
            "tools": [calc],
        },
    ]

    system_prompt = """You coordinate tasks. When the user needs precise arithmetic, delegate to math-specialist via task tool.
Otherwise answer briefly in Chinese."""

    agent = create_deep_agent(
        model=llm,
        tools=[calc],
        system_prompt=system_prompt,
        subagents=subagents,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我算 (888 + 222) * 3 等于多少，用中文回复结果。"}]}
    )
    msgs = result.get("messages", [])
    if msgs:
        print(getattr(msgs[-1], "content", str(msgs[-1])))
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Week18: DeepAgents multi-agent (subagents + task tool)")
    print("=" * 60)

    try:
        print("\n--- Demo 1: research + writer subagents ---\n")
        demo_subagents_research_write()
        print("\n--- Demo 2: math subagent with calc tool ---\n")
        demo_subagents_with_tool()
    except Exception as e:
        err = str(e)
        print("Error:", err)
        if "401" in err or "api" in err.lower():
            print("Check SILICONFLOW_API_KEY / model supports tool calling.")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
