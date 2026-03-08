"""
Week16: Silicon（硅基流动）LLM API — 模型的 tool calling 能力，直接运行脚本.
运行: 在项目根目录执行  python tests/week16/0_run_silicon_tool_calling.py
需要: openai，环境变量 SILICONFLOW_API_KEY
"""
import json
import os
import re

os.environ["SILICONFLOW_API_KEY"] = "sk-yyyvuckwvwpzuanghmegtoszbpezmhfycaihzzsjicidshwc"
# 运行前请设置环境变量 SILICONFLOW_API_KEY

SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
TOOL_CALLING_MODEL = "Qwen/Qwen2.5-7B-Instruct"

key = os.environ["SILICONFLOW_API_KEY"]
from openai import OpenAI
client = OpenAI(api_key=key, base_url=SILICON_BASE_URL)


messages = [
    {"role": "system", "content": "你是个智能助手，你的名字叫小助"},
    {"role": "user", "content": "你是谁"},
    {"role": "assistant", "content": "我是小助，一个能够回答问题、提供信息、聊天交流的人工智能助手。有什么我可以帮助你的吗？"},
    {"role": "user", "content": "你不要叫小助了，你叫小智吧。以后我就叫你小智了。"},
    {"role": "assistant", "content": "好的，我记住了"},
    {"role": "user", "content": "你是谁"},
]
response = client.chat.completions.create(
    model=TOOL_CALLING_MODEL,
    messages=messages,
    tools=[],
    tool_choice="auto",
    max_tokens=1024,
)

print(response)
