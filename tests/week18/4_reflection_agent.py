"""
Week18: Reflection Agent (生成 -> 反思 -> 修订)

运行: python tests/week18/4_reflection_agent.py
依赖: pip install langgraph langchain-openai langchain-core

Reflection 模式（类似 Reflexion / self-critique）:
  1. Generate: 产出初稿
  2. Reflect:  批评者角色指出问题（事实、逻辑、完整性）
  3. Revise:   根据批评改进答案
  可循环多轮，直到反思认为「通过」或达到 max_iterations。

本实现用 LangGraph: generate -> reflect -> (条件) revise -> reflect ... -> end
"""

from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-kltrvjvmttyvoqjfbccwztkcmcmogrffassqssctsmyctsrm",
)
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_ITERS = 3


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature,
        max_tokens=1024,
    )


class ReflectState(TypedDict):
    task: str
    draft: str
    critique: str
    iteration: int
    satisfied: bool
    final_answer: str


def node_generate(state: ReflectState) -> ReflectState:
    """首轮或修订后重新生成完整答案（修订节点会更新 draft）。"""
    llm = get_llm(0.4)
    if state.get("iteration", 0) == 0:
        msg = llm.invoke(
            [
                SystemMessage(content="你是助手，直接回答问题，条理清晰，用中文。"),
                HumanMessage(content=state["task"]),
            ]
        )
        draft = msg.content
    else:
        msg = llm.invoke(
            [
                SystemMessage(content="根据批评修订答案，输出完整新版，用中文。"),
                HumanMessage(
                    content=f"任务: {state['task']}\n当前版本:\n{state['draft']}\n批评:\n{state['critique']}"
                ),
            ]
        )
        draft = msg.content
    return {
        "draft": draft,
        "iteration": state.get("iteration", 0),
    }


def node_reflect(state: ReflectState) -> ReflectState:
    llm = get_llm(0.2)
    msg = llm.invoke(
        [
            SystemMessage(
                content="你是严格审稿人。检查回答是否切题、有无明显逻辑漏洞、是否过于空洞。"
                "只输出一行 JSON: {\"satisfied\": true/false, \"critique\": \"...\"}"
                "最少要让稿件修改一次"
            ),
            HumanMessage(content=f"任务:\n{state['task']}\n\n候选回答:\n{state['draft']}"),
        ]
    )
    text = msg.content.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    satisfied = False
    critique = text
    try:
        data = json.loads(text)
        satisfied = bool(data.get("satisfied", False))
        critique = str(data.get("critique", critique))
    except json.JSONDecodeError:
        satisfied = "满意" in text or "satisfied" in text.lower() and "false" not in text.lower()
    it = state.get("iteration", 0) + 1
    return {
        "critique": critique,
        "satisfied": not satisfied,
        "iteration": it,
    }


def route_after_reflect(state: ReflectState) -> Literal["revise", "finish"]:
    if state.get("satisfied"):
        return "finish"
    if state.get("iteration", 0) >= MAX_ITERS:
        return "finish"
    return "revise"


def node_finalize(state: ReflectState) -> ReflectState:
    state["final_answer"] = state["draft"]
    print(state)
    return state


def build_reflection_graph():
    """
    generate -> reflect -> (revise -> generate)* -> finalize
    这里 revise 用边连回 generate；generate 在 iteration>0 时读 critique 做修订。
    """
    g = StateGraph(ReflectState)

    def generate_wrapper(s: ReflectState) -> ReflectState:
        return node_generate(s)

    g.add_node("generate", generate_wrapper)
    g.add_node("reflect", node_reflect)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "generate")
    g.add_edge("generate", "reflect")
    g.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"revise": "generate", "finish": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


def demo():
    app = build_reflection_graph()
    task = "用三句话解释什么是 Plan-and-Execute Agent，并举一个生活类比。"
    out = app.invoke(
        {
            "task": task,
            "draft": "",
            "critique": "",
            "iteration": 0,
            "satisfied": False,
            "final_answer": "",
        }
    )
    print("Task:", task)
    print("Iterations (reflect count):", out.get("iteration"))
    print("Satisfied:", out.get("satisfied"))
    print("Critique (last):", (out.get("critique") or "")[:500])
    print("--- Final ---")
    print(out.get("final_answer", out.get("draft", "")))


if __name__ == "__main__":
    print("=" * 60)
    print("  Week18: Reflection Agent")
    print("=" * 60)
    try:
        demo()
    except Exception as e:
        print("Error:", e)
    print("=" * 60)
