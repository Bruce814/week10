"""
Week18: ReAct Agent (Reasoning + Acting)

运行: python tests/week18/2_react_agent.py
依赖: pip install langgraph langchain-openai langchain-core

ReAct 核心循环:
  Thought  -> 模型推理下一步
  Action   -> 调用工具
  Observation -> 工具结果写回对话
  ... 直到模型不再调用工具，输出最终 Thought/答案

本示例两种方式:
  A) LangGraph 预置 create_react_agent（与 LangChain tool-calling 等价图）
  B) 手写迷你循环（便于理解「无框架时」如何实现 ReAct）
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ========== 配置（与 tests/week18 其他脚本一致）==========
API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-kltrvjvmttyvoqjfbccwztkcmcmogrffassqssctsmyctsrm",
)
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature,
        max_tokens=1024,
    )


@tool
def add(a: int, b: int) -> int:
    """两整数相加。"""
    #return a + b
    if isinstance(a, int) and isinstance(b, int):
        return a+b
    else:
        return "error params , check your input"


@tool
def multiply(a: int, b: int) -> int:
    """两整数相乘。"""
    global  retry
    if retry <2:
        retry = retry + 1
        return '暂时无法计算，请重试'
    return a * b


@tool
def get_const(name: str) -> str:
    """查询内置常数名对应的数值。支持: pi, e"""
    m = {"pi": "3.14159265", "e": "2.71828182"}
    return m.get(name.lower(), "unknown")


TOOLS = [add, multiply, get_const]


def demo_react_langgraph():
    """方式 A: create_react_agent 编译好的 ReAct 图。"""
    llm = get_llm()
    agent = create_react_agent(llm, tools=TOOLS)
    query = "先算 12 + 8，再把结果乘以 3，最后用一句话用中文回答。"
    out = agent.invoke({"messages": [HumanMessage(content=query)]})
    msgs = out.get("messages", [])
    print("\n--- A) create_react_agent ---")
    print("Query:", query)
    for i, m in enumerate(msgs):
        role = type(m).__name__
        content = getattr(m, "content", "") or ""
        tcalls = getattr(m, "tool_calls", None)
        print(f"  [{i}] {role}: {content[:200]}{'...' if len(content) > 200 else ''}")
        if tcalls:
            for tc in tcalls:
                print(f"       tool_call: {tc.get('name', tc)} {tc.get('args', '')}")
    if msgs:
        last = msgs[-1]
        print("\nFinal:", getattr(last, "content", str(last)))


def demo_react_manual_loop():
    """
    方式 B: 手写 ReAct 风格循环（tool_calls 协议）。
    不依赖 create_react_agent，便于对照论文里的 Thought/Action/Observation。
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    system = """你是 ReAct 助手：通过思考并调用工具完成用户任务。
每次若需计算必须调用工具，不要心算大数。最后用中文简短总结。"""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content="计算 (7+5)*4，并说明用了哪些工具。"),
    ]
    print(messages)
    max_rounds = 8
    end_flag = False
    global retry
    retry = 0
    print("\n--- B) manual ReAct loop ---")


    while(not end_flag):
        ai: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai)

        for round_i in range(max_rounds):
            if not ai.tool_calls:
                print(f"Round {round_i}: final -> {ai.content}")
                break
            GOBS = ''
            for tc in ai.tool_calls:
                name = tc["name"]
                args = tc["args"]
                tool_fn = {t.name: t for t in TOOLS}[name]
                obs = tool_fn.invoke(args)
                messages.append(
                    ToolMessage(content=str(obs), tool_call_id=tc["id"])
                )

                GOBS += str(obs)

                print(f"Round {round_i}: Action {name}({args}) -> Observation {obs}")

            if GOBS.find('重试') != -1:
                end_flag = True
            else:
                ai: AIMessage = llm_with_tools.invoke(messages)



    else:
        print("Stopped: max_rounds")


if __name__ == "__main__":
    print("=" * 60)
    print("  Week18: ReAct Agent")
    print("=" * 60)
    try:
        # demo_react_langgraph()
        demo_react_manual_loop()
    except Exception as e:
        print("Error:", e)
        if "401" in str(e):
            print("Set SILICONFLOW_API_KEY.")
    print("=" * 60)
