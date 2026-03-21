# Week17: 强化学习 & SQL Agent

## 概述

本周包含两大主题: 强化学习 (RL) 和 SQL Agent (LLM 应用).

### Part 1: 强化学习

| 文件 | 算法 | 环境 | 依赖 |
|------|------|------|------|
| `0_q_learning_grid_world.py` | Q-Learning (表格方法) | 自定义 5x5 GridWorld | 无额外依赖 |
| `1_policy_gradient_cartpole.py` | REINFORCE (策略梯度) | CartPole-v1 | gymnasium, torch |
| `2_dqn_cartpole.py` | DQN (深度 Q 网络) | CartPole-v1 | gymnasium, torch |
| `3_rlhf_llm_ppo.py` | RLHF (PPO for LLM) | 自定义 Tiny GPT | torch |

### Part 2: SQL Agent

| 文件 | 框架 | 特点 | 依赖 |
|------|------|------|------|
| `4_sql_agent_langchain.py` | LangChain | 高度封装, SQLDatabaseToolkit | langchain, langchain-openai |
| `5_sql_agent_langgraph.py` | LangGraph | 显式状态图, 细粒度控制 | langgraph, langchain-openai |

## 快速开始

```bash
# === 强化学习 ===

# 示例0: Q-Learning (无额外依赖)
python tests/week17/0_q_learning_grid_world.py

# 示例1-2: 需要 gymnasium
pip install gymnasium torch matplotlib
python tests/week17/1_policy_gradient_cartpole.py
python tests/week17/2_dqn_cartpole.py

# 示例3: RLHF (纯 PyTorch, 无需下载模型)
python tests/week17/3_rlhf_llm_ppo.py

# === SQL Agent ===

# 需要设置 API Key
export SILICONFLOW_API_KEY="your-api-key"

# LangChain 实现
pip install langchain langchain-openai langchain-community sqlalchemy
python tests/week17/4_sql_agent_langchain.py

# LangGraph 实现
pip install langgraph
python tests/week17/5_sql_agent_langgraph.py
```

## RL 算法对比

```
                Q-Learning        REINFORCE         DQN             RLHF(PPO)
==============================================================================
学什么          Q(s,a) 值表       策略 pi(a|s)      Q(s,a) 网络     LLM 策略
策略类型        off-policy        on-policy         off-policy      on-policy
状态空间        离散(表格)        连续              连续            token 序列
关键技巧        e-greedy          MC 回报           经验回放+目标网络 KL 惩罚
应用场景        小型离散环境      连续控制          游戏 AI         LLM 对齐
```

## SQL Agent: LangChain vs LangGraph

```
                LangChain                   LangGraph
================================================================
构建方式        create_agent + toolkit      StateGraph + nodes/edges
代码量          ~20 行核心代码              ~100 行核心代码
控制粒度        粗 (黑盒循环)               细 (每个节点可控)
流程可视化      隐式                        显式状态图
错误重试        内置                        自定义 (条件边)
适合场景        快速原型                    生产级应用
```

### LangGraph 状态图

```
START -> route_question
           |-- "direct" --> direct_answer --> END
           |-- "query"  --> generate_sql
                                |
                            execute_sql
                                |
                            check_result
                                |-- "retry" --> generate_sql (max 3)
                                |-- "ok"    --> synthesize_answer --> END
```
