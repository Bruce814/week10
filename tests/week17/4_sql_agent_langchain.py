"""
Week17: SQL Agent -- LangChain 实现.

运行: python tests/week17/4_sql_agent_langchain.py
依赖: pip install langchain langchain-openai langchain-community sqlalchemy

==========================================================================
架构:
  用户问题 (自然语言)
      |
      v
  LangChain SQL Agent (create_agent)
      |--- SQL 相关工具
      |       |-- list_tables:    列出所有表
      |       |-- get_schema:     查看表结构
      |       |-- run_query:      执行 SQL 查询
      |       v
      |--- LLM (SiliconFlow / Qwen) 决定调用哪个工具
      v
  最终回答 (自然语言)

LangChain 方式的特点:
  - 高度封装, 几行代码即可创建 SQL Agent
  - Agent 自动选择工具、生成 SQL、执行、解释结果
  - 提供两种构建方式: SQLDatabaseToolkit / 手动工具定义
==========================================================================
"""

import os

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.tools import tool

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

# ====================================================================
# 2. Shared components
# ====================================================================


def create_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
        temperature=0,
        max_tokens=1024,
    )


def create_db():
    db_uri = f"sqlite:///{os.path.abspath(DB_PATH)}"
    return SQLDatabase.from_uri(db_uri)


SYSTEM_PROMPT = """You are a helpful SQL database assistant for an e-commerce system.
You have access to a SQLite database. Follow these steps:
1. First list tables to see what's available
2. Check the schema of relevant tables
3. Write a SQL query to answer the question
4. Execute the query
5. Give a clear answer in Chinese based on the results

Rules:
- Only use SELECT statements, never modify data
- If a query returns no results, explain why
- Always show the SQL you used"""

# ====================================================================
# 3. Mode 1: SQLDatabaseToolkit (built-in tools)
# ====================================================================


def build_agent_with_toolkit():
    """
    使用 LangChain 内置的 SQLDatabaseToolkit 自动生成工具.
    Toolkit 包含: sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker
    """
    llm = create_llm()
    db = create_db()

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    print("  [Toolkit] Available tools:")
    for t in tools:
        desc = t.description[:70].replace("\n", " ")
        print(f"    - {t.name}: {desc}...")

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# ====================================================================
# 4. Mode 2: Custom tools (manual definition)
# ====================================================================


def build_agent_custom_tools():
    """
    手动定义 SQL 工具, 展示更细粒度的控制.
    可以自定义工具名称、参数验证、安全检查等.
    """
    db = create_db()
    llm = create_llm()

    @tool
    def list_tables() -> str:
        """List all table names in the database. Use this first to know what tables are available."""
        tables = db.get_usable_table_names()
        return f"Available tables: {', '.join(tables)}"

    @tool
    def get_table_schema(table_name: str) -> str:
        """Get the CREATE TABLE statement and sample rows for a specific table. Use this to understand table structure before writing queries."""
        try:
            return db.get_table_info(table_names=[table_name])
        except Exception as e:
            return f"Error: {e}. Available tables: {', '.join(db.get_usable_table_names())}"

    @tool
    def execute_sql(query: str) -> str:
        """Execute a SQL SELECT query and return results. Only SELECT is allowed. Always validate your query before executing."""
        q = query.strip()
        if not q.upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed for safety."
        try:
            result = db.run(q)
            if not result or result.strip() == "":
                return "Query returned no results."
            return result
        except Exception as e:
            return f"SQL Error: {e}. Please check your query syntax and table/column names."

    tools = [list_tables, get_table_schema, execute_sql]

    print("  [Custom] Available tools:")
    for t in tools:
        print(f"    - {t.name}: {t.description[:70]}...")

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# ====================================================================
# 5. Run
# ====================================================================

SAMPLE_QUESTIONS = [
    "数据库里有哪些表? 每个表有多少条数据?",
    "列出所有产品的名称和重量",
    "哪个仓库的库存成本最高? 总成本是多少?",
    "从各仓库发货到各城市的运费是怎样的? 最贵的路线是哪条?",
]


def run_demo(agent, questions, label=""):
    """运行 agent 回答一系列问题."""
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"  {label} Question {i}: {q}")
        print("=" * 60)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": q}]})
            msgs = result.get("messages", [])
            if msgs:
                final = msgs[-1]
                content = final.content if hasattr(final, "content") else str(final)
                print(f"\n  >> Answer: {content}")
            else:
                print(f"\n  >> Result: {result}")
        except Exception as e:
            err = str(e)
            if "401" in err or "AuthenticationError" in err or "Api key" in err:
                print("\n  >> API Key invalid. Please update SILICONFLOW_API_KEY.")
                return False
            print(f"\n  >> Error: {e}")
    return True


# ====================================================================
# 6. Main
# ====================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Week17: SQL Agent -- LangChain Implementation")
    print("=" * 62)

    db = create_db()
    print(f"\n  Database: {DB_PATH}")
    print(f"  Tables: {db.get_usable_table_names()}")
    print(f"  LLM: {MODEL_NAME}")

    # --- Mode 1: Toolkit ---
    # print("\n" + "-" * 62)
    # print("  [Mode 1] SQLDatabaseToolkit (built-in tools)\n")
    # agent1 = build_agent_with_toolkit()
    #
    # ok = run_demo(agent1, SAMPLE_QUESTIONS[:], "[Toolkit]")

    ok = True
    if ok:
        # --- Mode 2: Custom tools ---
        print("\n\n" + "-" * 62)
        print("  [Mode 2] Custom Tools (manual definition)\n")
        agent2 = build_agent_custom_tools()

        run_demo(agent2, SAMPLE_QUESTIONS[:], "[Custom]")

    print("\n" + "=" * 62)
    print("  Done!")
    print("=" * 62)
