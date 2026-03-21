# Week16 直接运行脚本（可调试）

所有逻辑都在各脚本的 `if __name__ == "__main__":` 下执行，可直接运行或断点调试。

## 文件说明

| 文件 | 说明 |
|------|------|
| `mcp_server_demo.py` | 基础 MCP Server（stdio），提供 `add`、`multiply` 和 `greeting` 资源 |
| `mcp_server_advanced.py` | **进阶 FastMCP Server**：复杂 tools（eval_math、search_docs、get_weather、kv_store、summarize_text、batch_calculate）、resources（doc、config）、prompts（make_task_prompt） |
| `run_mcp_server_client.py` | 基础 MCP Client 演示 |
| `run_mcp_advanced.py` | **进阶 MCP Client**：连接 `mcp_server_advanced`，演示上述复杂 tools/resources/prompts 的调用 |
| `plan_execute_agent.py` | **Plan-and-Execute Agent**：连接进阶 MCP Server，先规划步骤再依次执行；含多组预定义场景 + 可选 LLM 规划 |
| `run_silicon_tool_calling.py` | Silicon LLM 的 tool calling：add / compare / count_letter |
| `run_silicon_mcp_integration.py` | 模型与 MCP 联动 |
| `langchain_react_example.py` / `run_langchain_react.py` | ReAct 示例 |

## 依赖

```bash
pip install -r tests/week16/requirements-week16.txt
```

需要调用 Silicon API 的脚本会读环境变量 `SILICONFLOW_API_KEY`。

## 运行方式（在项目根目录 `week10` 下）

```bash
# 1. 基础 MCP Server 与 Client
python tests/week16/run_mcp_server_client.py

# 1b. 进阶 MCP（复杂 tools/resources/prompts）
python tests/week16/run_mcp_advanced.py

# 1c. Plan-and-Execute Agent（连接 MCP，多场景规划执行；可选 SILICONFLOW_API_KEY 做 LLM 规划）
python tests/week16/plan_execute_agent.py

# 2. Silicon tool calling（需设置 SILICONFLOW_API_KEY）
python tests/week16/run_silicon_tool_calling.py

# 3. 模型与 MCP 联动（需 SILICONFLOW_API_KEY）
python tests/week16/run_silicon_mcp_integration.py

# 4. LangChain ReAct（可选 SILICONFLOW_API_KEY）
python tests/week16/run_langchain_react.py
# 或
python tests/week16/langchain_react_example.py
```

单独启动 MCP Server（供其他 client 连接）：

```bash
# 基础
python -m tests.week16.mcp_server_demo
# 进阶（复杂工具）
python -m tests.week16.mcp_server_advanced
```
