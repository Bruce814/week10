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

SILICON_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Compute the sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Compute the product of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Compute a minus b (a - b). Use when you need the difference between two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Minuend"},
                    "b": {"type": "number", "description": "Subtrahend"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": "Compare two numbers and say which is larger or equal",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eval_math",
            "description": "Safely evaluate a math expression containing numbers and + - * / ( ). Example: '(2+3)*4' returns 20.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '10+2*3' or '(10-2)/2'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_letter_in_string",
            "description": "Count how many times a letter appears in a string",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "Source string"},
                    "b": {"type": "string", "description": "Letter to count"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def local_add(a: float, b: float) -> float:
    return a + b


def local_multiply(a: float, b: float) -> float:
    return a * b


def local_subtract(a: float, b: float) -> float:
    return a - b


def local_compare(a: float, b: float) -> str:
    if a > b:
        return f"{a} is greater than {b}"
    if a < b:
        return f"{b} is greater than {a}"
    return f"{a} is equal to {b}"


def local_eval_math(expression: str) -> str:
    expr = expression.strip()
    if not re.match(r"^[\d\s+\-*/().]+$", expr):
        return "Error: only numbers and + - * / ( ) allowed"
    try:
        result = eval(expr)
        return str(result) if isinstance(result, (int, float)) else str(result)
    except Exception as e:
        return f"Error: {e}"


def local_count_letter_in_string(a: str, b: str) -> str:
    return f"The letter '{b}' appears {a.lower().count(b.lower())} times in the string."


LOCAL_TOOLS = {
    "add": local_add,
    "multiply": local_multiply,
    "subtract": local_subtract,
    "compare": local_compare,
    "eval_math": local_eval_math,
    "count_letter_in_string": local_count_letter_in_string,
}


def run_tool_call_add(client):
    """模型决定调用 add，执行并打印结果."""
    messages = [{"role": "user", "content": "请计算 100 加 200 等于多少？用你的工具算。"}]
    response = client.chat.completions.create(
        model=TOOL_CALLING_MODEL,
        messages=messages,
        tools=SILICON_TOOLS,
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message
    print(msg)
    if not getattr(msg, "tool_calls", None) or not msg.tool_calls:
        print("模型未返回 tool_calls")
        return
    tc = msg.tool_calls[0]
    name, args_str = tc.function.name, tc.function.arguments
    args = json.loads(args_str)
    print(f'details : {name}: {args_str}')
    print('_'*30)
    result = LOCAL_TOOLS[name](**args)
    print(f"  tool_calls: {name}({args}) -> {result}")


def run_tool_call_compare(client):
    """模型调用 compare 比较两个数."""
    messages = [{"role": "user", "content": "9.11 和 9.9 哪个更大？用工具比较后回答。"}]
    response = client.chat.completions.create(
        model=TOOL_CALLING_MODEL,
        messages=messages,
        tools=SILICON_TOOLS,
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message
    if not getattr(msg, "tool_calls", None) or not msg.tool_calls:
        print("模型未返回 tool_calls")
        return
    tc = msg.tool_calls[0]
    name, args_str = tc.function.name, tc.function.arguments
    args = json.loads(args_str)
    out = LOCAL_TOOLS[name](**args)
    print(f"  tool_calls: {name}({args}) -> {out}")


def run_tool_call_count_letter(client):
    """模型调用 count_letter_in_string."""
    messages = [{"role": "user", "content": "英文单词 strawberry 里有几个字母 r？用工具统计后回答。"}]
    response = client.chat.completions.create(
        model=TOOL_CALLING_MODEL,
        messages=messages,
        tools=SILICON_TOOLS,
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message
    if not getattr(msg, "tool_calls", None) or not msg.tool_calls:
        print("模型未返回 tool_calls")
        return
    tc = msg.tool_calls[0]
    name, args_str = tc.function.name, tc.function.arguments
    args = json.loads(args_str)
    out = LOCAL_TOOLS[name](**args)
    print(f"  tool_calls: {name}({args}) -> {out}")

    print('不用tools')
    messages = [{"role": "user", "content": "英文单词 strawberry 里有几个字母 r？"}]
    response = client.chat.completions.create(
        model=TOOL_CALLING_MODEL,
        messages=messages,
        tools=[],
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message
    print(msg)


def _execute_tool_call(name: str, args: dict):
    """执行单次 tool call，返回结果字符串."""
    if name not in LOCAL_TOOLS:
        return f"Unknown tool: {name}"
    try:
        out = LOCAL_TOOLS[name](**args)
        return str(out) if not isinstance(out, str) else out
    except Exception as e:
        return f"Error: {e}"


def run_multi_turn_tool_calls(client, user_message: str, max_rounds: int = 6, verbose: bool = True):
    """
    多轮 tool call：若模型返回 tool_calls，则执行并把结果追加到 messages 再请求，
    直到模型返回最终文本或达到 max_rounds。返回最终回答文本。
    """
    messages = [{"role": "user", "content": user_message}]
    final_answer = ""
    for round_idx in range(max_rounds):
        response = client.chat.completions.create(
            model=TOOL_CALLING_MODEL,
            messages=messages,
            tools=SILICON_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )
        msg = response.choices[0].message
        if not getattr(msg, "tool_calls", None) or not msg.tool_calls:
            final_answer = (msg.content or "").strip()
            if verbose:
                print(f"  [round {round_idx + 1}] 最终回答: {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
            return final_answer
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool_call(name, args)
            if verbose:
                print(f"  [round {round_idx + 1}] {name}({args}) -> {result[:80]}{'...' if len(str(result)) > 80 else ''}")
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tc.id,
            })

            print(messages)
    return final_answer or "(达到最大轮数，未得到最终回答)"


# ---------- 更复杂的 case（多轮 / 多工具 / 推理）----------

def run_case_eval_math(client):
    """复杂 case: 用 eval_math 计算表达式."""
    print("  用户: 请用工具计算 (5+3)*2-4 等于多少？")
    run_multi_turn_tool_calls(client, "请用你的工具计算 (5+3)*2-4 等于多少？只输出最终数字。", verbose=True)


def run_case_chain_calc(client):
    """复杂 case: 多步链式 — 先 100+200 再乘以 2."""
    print("  用户: 先计算 100 加 200，再把结果乘以 2，告诉我最终数字。")
    run_multi_turn_tool_calls(client, "先计算 100 加 200，再把结果乘以 2，告诉我最终数字。请用工具逐步计算。", verbose=True)


def run_case_compare_then_diff(client):
    """复杂 case: 比较两数后算差值."""
    print("  用户: 123 和 456 哪个更大？大多少？请用工具。")
    run_multi_turn_tool_calls(client, "123 和 456 哪个更大？大多少？请用工具比较并计算差值后回答。", verbose=True)


def run_case_multi_letter_count(client):
    """复杂 case: 统计一个词里多个字母出现次数."""
    print("  用户: banana 里字母 a 和 n 各出现几次？用工具后总结。")
    run_multi_turn_tool_calls(client, "英文单词 banana 里字母 a 出现几次、字母 n 出现几次？请用工具分别统计后总结一句。", verbose=True)


def run_case_expression_then_compare(client):
    """复杂 case: 先算表达式再和常数比较."""
    print("  用户: 用工具计算 (10-2)/2+1，再和 5 比较大小。")
    run_multi_turn_tool_calls(client, "请用工具先计算 (10-2)/2+1 的结果，再用比较工具和 5 比较大小，最后用一句话总结。", verbose=True)


def run_case_three_way_compare(client):
    """复杂 case: 三个数比较（需多次 compare 或 eval）."""
    print("  用户: 7、12、9 这三个数里最大的是谁？请用工具。")
    run_multi_turn_tool_calls(client, "7、12、9 这三个数里最大的是谁？请用比较工具两两比较后得出结论。", verbose=True)


if __name__ == "__main__":
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not key:
        print("请设置环境变量 SILICONFLOW_API_KEY 后重试")
        exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=SILICON_BASE_URL)

    # print("=== 1. Tool call: add(100, 200) ===")
    # run_tool_call_add(client)
    # print()
    #
    # print("=== 2. Tool call: compare(9.11, 9.9) ===")
    # run_tool_call_compare(client)
    # print()

    # print("=== 3. Tool call: count_letter_in_string('strawberry', 'r') ===")
    # run_tool_call_count_letter(client)
    # print()

    # print("=== 4. 复杂 case: eval_math 表达式 (5+3)*2-4 ===")
    # run_case_eval_math(client)
    # print()

    # print("=== 5. 复杂 case: 多步链式 100+200 再乘 2 ===")
    # run_case_chain_calc(client)
    # print()
    #
    print("=== 6. 复杂 case: 比较后算差 123 vs 456 ===")
    run_case_compare_then_diff(client)
    print()

    print("=== 7. 复杂 case: banana 中 a/n 各出现几次 ===")
    run_case_multi_letter_count(client)
    print()

    print("=== 8. 复杂 case: (10-2)/2+1 再和 5 比较 ===")
    run_case_expression_then_compare(client)
    print()

    print("=== 9. 复杂 case: 三数取最大 7 vs 12 vs 9 ===")
    run_case_three_way_compare(client)
