"""
Week16: 基于 LangChain / LangGraph 的 ReAct Agent 示例.
使用 Silicon 作为 LLM（OpenAI 兼容 base_url），定义若干 tools，由 ReAct agent 推理并调用工具.
需要: langgraph, langchain-openai, langchain-core；可选 SILICONFLOW_API_KEY.
"""
import os
from typing import Annotated

os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
# 支持 tool calling 的模型
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ---------- 定义 ReAct 可用的 Tools ----------

@tool
def add(a: int, b: int) -> int:
    """将两个整数相加，返回和。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘，返回积。"""
    return a * b


@tool
def power(
    base: Annotated[float, "底数"],
    exponent: Annotated[float, "指数"],
) -> float:
    """计算 base 的 exponent 次方。"""
    return base ** exponent


TOOLS = [add, multiply, power]


def get_silicon_llm(model: str = DEFAULT_MODEL, api_key: str | None = None):
    """返回配置了 Silicon base_url 的 ChatOpenAI."""
    key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    return ChatOpenAI(
        model=model,
        api_key=key or "dummy",
        base_url=SILICON_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
    )


def build_react_agent(llm=None):
    """构建 ReAct agent（Reasoning + Acting）：先推理再选工具执行."""
    if llm is None:
        llm = get_silicon_llm()
    agent = create_react_agent(llm, tools=TOOLS)
    return agent


def run_react_example(query: str, api_key: str | None = None):
    """
    运行一次 ReAct 示例：用户输入 query，agent 可多次推理与调用工具，最后返回答案.
    若无 SILICONFLOW_API_KEY 则仅构建 agent 不发起请求.
    """
    llm = get_silicon_llm(api_key=api_key)
    agent = build_react_agent(llm=llm)
    # LangGraph 的 create_react_agent 返回的 graph 使用 messages 作为输入
    from langchain_core.messages import HumanMessage
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result


if __name__ == "__main__":
    import os
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not key:
        print("未设置 SILICONFLOW_API_KEY；仅演示 tools 与 agent 构建。")
    print("=== 1. Tools 直接调用 ===")
    print("add(2, 3) =", add.invoke({"a": 2, "b": 3}))
    print("multiply(4, 5) =", multiply.invoke({"a": 4, "b": 5}))
    print("power(2, 3) =", power.invoke({"base": 2, "exponent": 3}))
    print("=== 2. 构建 ReAct agent ===")
    llm = get_silicon_llm(api_key=key)
    agent = build_react_agent(llm=llm)
    print("agent 已构建, base_url =")
    if key:
        print("=== 3. ReAct 示例：请用工具计算 5 +6 然后再 乘以 7 ===")
        result = run_react_example("请用工具计算 5 +6 然后再 乘以 7")
        messages = result.get("messages") or []
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None) or str(last)
            print("最后一条消息 content:", content)
        else:
            print("result:", result)
