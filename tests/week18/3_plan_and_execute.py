"""
Week18: Plan-and-Execute Agent

运行: python tests/week18/3_plan_and_execute.py
依赖: pip install langgraph langchain-openai langchain-core

与 ReAct 的区别:
  - ReAct: 每步边想边做，动作与规划交织。
  - Plan-and-Execute: 先产出步骤计划 (Plan)，再按步骤执行 (Execute)，最后汇总。

本图结构:
  START -> plan (LLM 输出 JSON 步骤列表)
       -> execute (逐步调用工具完成每步)
       -> summarize (LLM 根据步骤与结果写最终回答)
       -> END
"""

from __future__ import annotations

import json
import os
import re
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


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature,
        max_tokens=1024,
    )


@tool
def calc(expr: str) -> str:
    """计算仅含数字与 +-*/() 的算术表达式。"""
    e = re.sub(r"[^0-9+\-*/().]", "", expr)
    try:
        return str(eval(e, {"__builtins__": {}}, {}))
    except Exception as ex:
        return f"error:{ex}"


@tool
def word_count(text: str) -> int:
    """统计中英混合字符串中「词」的数量（按空白分词）。"""
    return len(text.split())


TOOLS = {"calc": calc, "word_count": word_count}


class PEState(TypedDict):
    goal: str
    plan: list[str]
    step_results: list[str]
    final_answer: str


def node_plan(state: PEState) -> PEState:
    llm = get_llm(0)
    prompt = f"""用户目标: {state['goal']}

可用工具:
- calc(expr): 算术
- word_count(text): 统计词数

请输出完成目标所需的步骤列表，严格为 JSON 数组，每步一句自然语言指令（要能映射到工具）。
示例: ["用 calc 计算 10+20", "用 word_count 统计短语 hello world 的词数"]
只输出 JSON 数组，不要其它文字。"""
    msg = llm.invoke([HumanMessage(content=prompt)])
    text = msg.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        plan = json.loads(text)
        if not isinstance(plan, list):
            plan = [str(plan)]
        plan = [str(p) for p in plan]
    except json.JSONDecodeError:
        plan = [state["goal"]]
    return {"plan": plan, "step_results": []}


def node_execute(state: PEState) -> PEState:
    """对每一步：让 LLM 决定调用哪个工具（单轮 tool 模拟）。"""
    llm = get_llm(0)
    results: list[str] = []
    tool_desc = "calc(expr), word_count(text)"
    for i, step in enumerate(state["plan"]):
        sub = llm.invoke(
            [
                SystemMessage(
                    content=f"根据步骤调用工具。可用: {tool_desc}。"
                    "只输出一行 JSON: {{\"tool\":\"calc|word_count\",\"arg\":\"参数\"}}。"
                    "若无需工具则 {{\"tool\":\"none\",\"arg\":\"\"}}"
                ),
                HumanMessage(content=f"步骤{i+1}: {step}"),
            ]
        )
        text = sub.content.strip()
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            spec = json.loads(text)
        except Exception:
            results.append(f"步骤{i+1}: 解析失败 -> {text[:120]}")
            continue
        tname = spec.get("tool", "none")
        arg = spec.get("arg", "")
        if tname == "none" or tname not in TOOLS:
            results.append(f"步骤{i+1}: {step} -> (无工具) {text[:200]}")
            continue
        try:
            out = TOOLS[tname].invoke(arg)
            results.append(f"步骤{i+1}: {tname}({arg!r}) -> {out}")
        except Exception as e:
            results.append(f"步骤{i+1}: 工具错误 {e}")
    return {"step_results": results}


def node_summarize(state: PEState) -> PEState:
    llm = get_llm(0.3)
    body = "\n".join(state["step_results"])
    msg = llm.invoke(
        [
            SystemMessage(content="根据下列计划与执行结果，用中文给用户一段简洁总结。"),
            HumanMessage(
                content=f"目标: {state['goal']}\n计划: {state['plan']}\n执行记录:\n{body}"
            ),
        ]
    )
    return {"final_answer": msg.content}


def build_plan_execute_graph():
    g = StateGraph(PEState)
    g.add_node("plan", node_plan)
    g.add_node("execute", node_execute)
    g.add_node("summarize", node_summarize)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def demo():
    app = build_plan_execute_graph()
    ## 作业1
    ## plan: 子问题相互独立，无依赖
    ### plan: 子问题有前后依赖
    goal = "先统计'plan execute agent pattern' 有多少个词，再计算词数量的立方+2 "
            #"先计算 (100+50)*2，再统计句子 'plan execute agent pattern' 有多少个词。")
    out = app.invoke(
        {
            "goal": goal,
            "plan": [],
            "step_results": [],
            "final_answer": "",
        }
    )
    print("Goal:", goal)
    print("Plan:", out["plan"])
    print("Step results:")
    for line in out["step_results"]:
        print(" ", line)
    print("Final:", out["final_answer"])


if __name__ == "__main__":
    print("=" * 60)
    print("  Week18: Plan-and-Execute")
    print("=" * 60)
    try:
        demo()
    except Exception as e:
        print("Error:", e)
    print("=" * 60)
