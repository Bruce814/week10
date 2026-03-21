"""
Week17: SQL Agent — LangGraph 实现.

运行: python tests/week17/5_sql_agent_langgraph.py
依赖: pip install langgraph langchain-openai langchain-community sqlalchemy

==========================================================================
与 LangChain Agent 的区别:
  LangChain Agent: 封装好的循环, 自动决定工具调用顺序
  LangGraph Agent: 显式定义状态图, 每个节点是一个处理步骤, 边控制流转

LangGraph 的优势:
  1. 流程可视化 — 可以清晰看到 Agent 的决策路径
  2. 细粒度控制 — 可以在任意节点插入自定义逻辑
  3. 条件分支   — 根据状态灵活选择下一步
  4. 可恢复执行 — 支持断点续跑、人工审核
  5. 流式输出   — 逐节点输出中间结果

本示例的状态图:

    [START]
       |
       v
  [route_question]  -- "直接回答" --> [direct_answer] --> [END]
       |
       | "需要查询"
       v
  [generate_sql]
       |
       v
  [execute_sql]
       |
       v
  [check_result] -- "结果为空/出错" --> [fix_sql] --> [execute_sql]
       |
       | "结果正常"
       v
  [synthesize_answer]
       |
       v
     [END]
==========================================================================
"""

import os
import operator
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ====================================================================
# 1. Configuration
# ====================================================================

SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-kltrvjvmttyvoqjfbccwztkcmcmogrffassqssctsmyctsrm",
)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "works", "ecommerce.db")


def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
        temperature=0,
        max_tokens=1024,
    )


def get_db():
    return SQLDatabase.from_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")


# ====================================================================
# 2. State Definition (LangGraph 的核心: 定义状态)
# ====================================================================

class SQLAgentState(TypedDict):
    """Agent 在图中流转的状态."""
    question: str
    table_info: str
    sql_query: str
    query_result: str
    answer: str
    error: str
    retry_count: int
    route: str  # "query" or "direct"


# ====================================================================
# 3. Graph Nodes (每个节点是一个处理函数)
# ====================================================================

db = get_db()
llm = get_llm()

TABLE_NAMES = db.get_usable_table_names()
FULL_SCHEMA = db.get_table_info()


def route_question(state: SQLAgentState) -> SQLAgentState:
    print(state, "route_question")
    """
    Node 1: 判断问题是否需要查询数据库.
    简单问候或无关问题 -> direct_answer
    数据相关问题 -> generate_sql
    """
    question = state["question"]
    print(f"  [route_question] Analyzing: {question!r}")

    response = llm.invoke([
        SystemMessage(content=f"""You are a router. Given a user question, determine if it requires a SQL database query.
Available tables: {', '.join(TABLE_NAMES)}
Respond with ONLY one word: "query" if database access is needed, "direct" if not."""),
        HumanMessage(content=question),
    ])

    route = response.content.strip().lower()
    if "query" in route:
        state["route"] = "query"
        print(f"  [route_question] -> Need SQL query")
    else:
        state["route"] = "direct"
        print(f"  [route_question] -> Direct answer")

    state["table_info"] = FULL_SCHEMA
    return state


def direct_answer(state: SQLAgentState) -> SQLAgentState:
    """Node: 不需要查询数据库时, 直接回答."""
    print(f"  [direct_answer] Answering without SQL...")

    # response = llm.invoke([
    #     SystemMessage(content="You are a helpful assistant. Answer in Chinese."),
    #     HumanMessage(content=state["question"]),
    # ])
    # state["answer"] = response.content
    state['answer'] = '对不起，当前问题不支持。本agent只处理sql有关的问题'
    return state


def generate_sql(state: SQLAgentState) -> SQLAgentState:
    """
    Node 2: 根据用户问题和表结构, 生成 SQL 查询.
    """
    print(f"  [generate_sql] Generating SQL...")

    error_context = ""
    if state.get("error"):
        error_context = f"\nPrevious query failed: {state['error']}\nPrevious SQL: {state.get('sql_query', '')}\nPlease fix the query."

    response = llm.invoke([
        SystemMessage(content=f"""You are a SQL expert. Generate a SQLite SELECT query to answer the user's question.

Database schema:
{state['table_info']}

Rules:
- Output ONLY the SQL query, no explanations, no markdown
- Only use SELECT statements
- Use proper table and column names from the schema
- Handle potential NULL values
{error_context}"""),
        HumanMessage(content=state["question"]),
    ])

    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    state["sql_query"] = sql
    state["error"] = ""
    print(f"  [generate_sql] SQL: {sql}")
    return state


def execute_sql(state: SQLAgentState) -> SQLAgentState:
    """
    Node 3: 执行 SQL 查询.
    """
    sql = state["sql_query"]
    print(f"  [execute_sql] Running: {sql[:80]}...")

    if not sql.strip().upper().startswith("SELECT"):
        state["error"] = "Only SELECT queries allowed"
        state["query_result"] = ""
        return state

    try:
        result = db.run(sql)
        state["query_result"] = result
        state["error"] = ""
        print(f"  [execute_sql] Got {len(result)} chars of results")
    except Exception as e:
        state["error"] = str(e)
        state["query_result"] = ""
        state["retry_count"] = state.get("retry_count", 0) + 1
        print(f"  [execute_sql] Error: {e}")

    return state


def check_result(state: SQLAgentState) -> SQLAgentState:
    """
    Node 4: 检查查询结果, 决定是重试还是继续.
    """
    if state.get("error"):
        print(f"  [check_result] Error detected, will retry")
        state["route"] = "retry"
    elif not state.get("query_result") or state["query_result"].strip() == "":
        print(f"  [check_result] Empty result")
        state["route"] = "empty"
    else:
        print(f"  [check_result] Result OK")
        state["route"] = "ok"
    return state


def synthesize_answer(state: SQLAgentState) -> SQLAgentState:
    print(state, "synthesize_answer")
    """
    Node 5: 将 SQL 查询结果转化为自然语言回答.
    """
    print(f"  [synthesize_answer] Generating final answer...")

    if state.get("route") == "empty":
        query_info = f"SQL: {state['sql_query']}\nResult: (empty - no matching data)"
    else:
        query_info = f"SQL: {state['sql_query']}\nResult: {state['query_result']}"

    response = llm.invoke([
        SystemMessage(content="""Based on the SQL query and results, provide a clear and helpful answer to the user's question.
- Answer in Chinese
- Format numbers nicely
- If the result is a table, present it clearly
- Mention what SQL was used"""),
        HumanMessage(content=f"Question: {state['question']}\n\n{query_info}"),
    ])

    state["answer"] = response.content
    return state


# ====================================================================
# 4. Build the Graph (构建状态图)
# ====================================================================

def build_sql_agent_graph():
    """
    构建 LangGraph 状态图:
      START -> route -> generate_sql -> execute -> check -> synthesize -> END
                 |                                   |
                 +-> direct_answer -> END            +-> fix (retry) -> execute
    """
    graph = StateGraph(SQLAgentState)

    # Add nodes
    graph.add_node("route_question", route_question)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("check_result", check_result)
    graph.add_node("synthesize_answer", synthesize_answer)

    # Add edges
    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        lambda s: s["route"],
        {"query": "generate_sql", "direct": "direct_answer"},
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "check_result")

    graph.add_conditional_edges(
        "check_result",
        lambda s: "generate_sql" if s["route"] == "retry" and s.get("retry_count", 0) < 3 else "synthesize_answer",
    )

    graph.add_edge("synthesize_answer", END)

    return graph.compile()


# ====================================================================
# 5. Run
# ====================================================================

SAMPLE_QUESTIONS = [
    "数据库里有哪些表? 每个表有多少条数据?",
    "你是谁，你会干什么"
    # "列出所有产品的名称和重量",
    # "哪个仓库的库存成本最高? 总成本是多少?",
    # "从各仓库发货到各城市, 最贵和最便宜的路线分别是哪条?",
]


def run_agent(app, question: str) -> str:
    """运行 SQL Agent 并返回结果."""
    initial_state: SQLAgentState = {
        "question": question,
        "table_info": "",
        "sql_query": "",
        "query_result": "",
        "answer": "",
        "error": "",
        "retry_count": 0,
        "route": "",
    }

    final_state = app.invoke(initial_state)
    return final_state["answer"]

## https://docs.langchain.com/oss/python/langgraph/quickstart
if __name__ == "__main__":
    print("=" * 62)
    print("  Week17: SQL Agent -- LangGraph Implementation")
    print("=" * 62)

  #   print("""
  # Graph structure:
  #   START -> route_question
  #              |-- "direct" --> direct_answer --> END
  #              |-- "query"  --> generate_sql
  #                                   |
  #                               execute_sql
  #                                   |
  #                               check_result
  #                                   |-- "retry" --> generate_sql (max 3)
  #                                   |-- "ok"/"empty" --> synthesize_answer --> END
  #   """)

    app = build_sql_agent_graph()
    print("  Graph compiled successfully!")
    print(f"  Database tables: {TABLE_NAMES}")
    print(f"  LLM: {MODEL_NAME}")

    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"  Question {i}: {q}")
        print("=" * 60)

        try:
            answer = run_agent(app, q)
            print(f"\n  >> Final Answer:\n{answer}")
        except Exception as e:
            err = str(e)
            if "401" in err or "AuthenticationError" in err:
                print(f"\n  >> API Key invalid. Please update SILICONFLOW_API_KEY.")
                break
            print(f"\n  >> Error: {err}")

    print("\n" + "=" * 62)
    print("  Done!")
    print("=" * 62)

  #   print("""
  # --- LangChain vs LangGraph ---
  # LangChain Agent:
  #   + Simple: few lines of code
  #   + Built-in SQL toolkit
  #   - Less control over decision flow
  #
  # LangGraph Agent:
  #   + Explicit state graph: visualizable, debuggable
  #   + Fine-grained control at every step
  #   + Conditional branching, retry logic
  #   + Supports streaming, checkpointing, human-in-the-loop
  #   - More code to write
  #   """)
