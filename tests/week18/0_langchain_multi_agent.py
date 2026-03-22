"""
Week18: 常用多 Agent 协作模式 — 基于 LangChain + LangGraph

运行: python tests/week18/0_langchain_multi_agent.py
依赖: pip install langchain langchain-openai langgraph

本文件演示三种常见协作形态（均可用同一套 OpenAI 兼容 API，如 SiliconFlow）:

  1) 流水线 (Pipeline)
     Analyst -> Writer: 先分析要点，再撰写成文，固定顺序。

  2) 监督者路由 (Supervisor / Router)
     Supervisor 根据用户问题分类，路由到「研究」或「计算」节点，再汇总。

  3) Agent-as-Tool (工具化子 Agent)
     主 Agent 通过工具调用封装好的「子 LLM」，由主 Agent 决定何时调用谁。

环境变量:
  SILICONFLOW_API_KEY  (可选，代码内有占位)
  SILICONFLOW_BASE_URL (默认 https://api.siliconflow.cn/v1)
"""

from __future__ import annotations

import os
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# =============================================================================
# 共享 LLM
# =============================================================================

API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-kltrvjvmttyvoqjfbccwztkcmcmogrffassqssctsmyctsrm",
)

## ollama
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature,
        max_tokens=2048,
    )



# =============================================================================
# 1) 流水线: Analyst -> Writer
# =============================================================================


class PipelineState(TypedDict):
    topic: str
    bullets: str
    article: str
    show_message: list[str]


def node_analyst(state: PipelineState) -> PipelineState:
    llm = get_llm(0.3)
    msg = llm.invoke(
        [
            SystemMessage(
                content="You are an analyst. Output 3-6 bullet points in Chinese, one per line, no preamble."
            ),
            HumanMessage(content=f"分析主题要点: {state['topic']}"),
        ]
    )
    return {"bullets": msg.content}


def node_writer(state: PipelineState) -> PipelineState:
    llm = get_llm(0.5)
    msg = llm.invoke(
        [
            SystemMessage(
                content="You are a writer. Turn the bullet list into a short coherent paragraph in Chinese."
            ),
            HumanMessage(
                content=f"主题: {state['topic']}\n要点:\n{state['bullets']}"
            ),
        ]
    )
    return {"article": msg.content}


### workflow ： start --> analyst --> writer --> END
## 发小红书的agent : 确定主题 --> 生图 --> 文案 --> 组合，润色 --> pubslish
### 文案 --> 选人设 --> 确定账号 --> 生成文案 --> 润色，发布 --> 跟踪表现

def build_pipeline_graph():
    g = StateGraph(PipelineState)
    g.add_node("analyst", node_analyst)
    g.add_node("writer", node_writer)
    g.add_edge(START, "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", END)
    return g.compile()


def demo_pipeline():
    print("\n" + "=" * 60)
    print("  [1] Pipeline: Analyst -> Writer")
    print("=" * 60)
    app = build_pipeline_graph()
    topic = "多智能体系统在客服场景中的价值"
    out = app.invoke({"topic": topic, "bullets": "", "article": ""})
    print("  Topic:", topic)
    print("  Bullets:\n", out["bullets"][:800], "..." if len(out["bullets"]) > 800 else "")
    print("  Article:\n", out["article"][:1200], "..." if len(out["article"]) > 1200 else "")


# =============================================================================
# 2) 监督者路由: research vs math
# =============================================================================


class RouterState(TypedDict):
    question: str
    route: str
    worker_output: str
    final_answer: str


def node_supervisor_route(state: RouterState) -> RouterState:
    llm = get_llm(0)
    msg = llm.invoke(
        [
            SystemMessage(
                content='Classify the user question. Reply with exactly one word: "research" if it needs '
                'general knowledge / explanation / opinion; "math" if it is mainly arithmetic or a numeric expression.'
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    t = msg.content.strip().lower()
    route = "math" if "math" in t else "research"
    return {"route": route}


def route_decision(state: RouterState) -> Literal["research", "math"]:
    return "math" if state.get("route") == "math" else "research"


def node_research_worker(state: RouterState) -> RouterState:
    llm = get_llm(0.2)
    msg = llm.invoke(
        [
            SystemMessage(content="Answer briefly in Chinese."),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"worker_output": msg.content}


def node_math_worker(state: RouterState) -> RouterState:
    """安全计算: 仅允许数字与 + - * / ( ) 空格."""
    q = state["question"]
    expr = re.sub(r"[^0-9+\-*/().\s]", "", q)
    if not expr.strip():
        return {"worker_output": "无法从问题中提取纯算术表达式。"}
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        return {"worker_output": f"计算结果: {expr.strip()} = {val}"}
    except Exception as e:
        return {"worker_output": f"计算失败: {e}"}


def node_merge(state: RouterState) -> RouterState:
    llm = get_llm(0)
    msg = llm.invoke(
        [
            SystemMessage(content="用一两句中文把下面 worker 的结果整理给用户，不要编造新事实。"),
            HumanMessage(
                content=f"原问题: {state['question']}\n路由: {state['route']}\n结果:\n{state['worker_output']}"
            ),
        ]
    )
    return {"final_answer": msg.content}


def build_supervisor_graph():
    g = StateGraph(RouterState)
    g.add_node("supervisor", node_supervisor_route)
    g.add_node("research", node_research_worker)
    g.add_node("math", node_math_worker)
    g.add_node("merge", node_merge)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_decision, {"research": "research", "math": "math"})
    g.add_edge("research", "merge")
    g.add_edge("math", "merge")
    g.add_edge("merge", END)
    return g.compile()


def demo_supervisor():
    print("\n" + "=" * 60)
    print("  [2] Supervisor: route -> research | math -> merge")
    print("=" * 60)
    app = build_supervisor_graph()
    for q in ("17 * 23 + 5 等于多少？", "用一句话解释什么是强化学习"):
        out = app.invoke(
            {
                "question": q,
                "route": "",
                "worker_output": "",
                "final_answer": "",
            }
        )
        print("  Q:", q)
        print("  route:", out["route"])
        print("  final:", out["final_answer"][:500])


# =============================================================================
# 3) Agent-as-Tool: 主 Agent 通过工具调用子专家
# =============================================================================


def build_agent_as_tool_demo():
    from langchain.agents import create_agent

    llm = get_llm(0)
    sub_llm = get_llm(0.3)
    sub_llm2 = get_llm(0.8)

    @tool
    def expert_math(expression: str) -> str:
        """Solve a simple arithmetic expression. Input like '12+34*2'."""
        expr = re.sub(r"[^0-9+\-*/().]", "", expression)
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"error: {e}"

    @tool
    def expert_summarize(text: str) -> str:
        """Summarize long text in Chinese in 2 sentences."""
        r = sub_llm2.invoke(
            [
                SystemMessage(content="Summarize in 2 short Chinese sentences."),
                HumanMessage(content=text[:4000]),
            ]
        )
        return r.content

    system = """You are a coordinator. You have two specialist tools:
- expert_math: for arithmetic
- expert_summarize: for condensing long text
Choose the right tool(s). Answer the user in Chinese after tools return."""

    agent = create_agent(
        model=llm,
        tools=[expert_math, expert_summarize],
        system_prompt=system,
    )
    return agent


def demo_agent_as_tool():
    print("\n" + "=" * 60)
    print("  [3] Agent-as-Tool: create_agent + specialist tools")
    print("=" * 60)
    agent = build_agent_as_tool_demo()
    q = "先算 (100+50)*2，再用一句话总结：强化学习通过试错优化策略。"
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(" result:", result)
    msgs = result.get("messages", [])
    if msgs:
        last = msgs[-1]
        print("  Q:", q)
        print("  A:", getattr(last, "content", str(last))[:1200])


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Week18: LangChain / LangGraph multi-agent patterns")
    print("=" * 60)

    try:
        demo_pipeline()
        #demo_supervisor()
        # demo_agent_as_tool()
    except Exception as e:
        err = str(e)
        print("\n  Error:", err)
        if "401" in err or "invalid" in err.lower():
            print("  Set SILICONFLOW_API_KEY and retry.")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
