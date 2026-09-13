import openai
import re
import datetime
import time

# ---------- 配置本地 Ollama ----------
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"          # 本地模型不需要真实 key
)
MODEL = "qwen2:1.5b"

# ---------- 工具函数 ----------
def get_current_time() -> str:
    """返回当前日期和时间（格式：YYYY-MM-DD HH:MM:SS）"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """计算数学表达式（支持 + - * / 和括号），返回结果字符串"""
    try:
        # 安全起见，只允许数字和运算符，防止恶意代码注入
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# ---------- 工具映射 ----------
tools = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

# ---------- ReAct 循环 ----------
def run_agent(user_question: str, max_rounds: int = 3):
    """
    手动实现 ReAct 推理循环：
    1. 构造 Prompt，要求模型输出 Thought 和 Action
    2. 解析 Action，调用对应工具
    3. 将 Observation 加入对话，直到得出最终答案
    """
    # 初始化消息列表（包含系统提示和用户问题）
    system_prompt = (
        "你是一个能使用工具的智能助手。"
        "你可以使用以下工具：\n"
        "- get_current_time: 获取当前时间，不需要参数\n"
        "- calculate: 计算数学表达式，参数为表达式字符串，例如 '123+456'\n\n"
        "在每轮回答中，你**必须**严格按照以下格式输出：\n"
        "Thought: 你当前的思考\n"
        "Action: 工具名称(参数)\n\n"
        "如果已经知道最终答案，则输出：\n"
        "Thought: 我已经知道答案\n"
        "Final Answer: 你的最终回答\n\n"
        "开始！"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    for round_num in range(1, max_rounds + 1):
        print(f"\n=== 第 {round_num} 轮 ===")
        # 调用模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0  # 降低随机性，让输出更稳定
        )
        reply = response.choices[0].message.content.strip()
        print(f"模型回复：\n{reply}\n")

        # 检查是否包含 "Final Answer"
        if "Final Answer:" in reply:
            final_answer = reply.split("Final Answer:")[-1].strip()
            print(f"✅ 最终答案：{final_answer}")
            return final_answer

        # 解析 Action
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", reply)
        if action_match:
            tool_name = action_match.group(1)
            arg = action_match.group(2).strip()
            if tool_name in tools:
                try:
                    if arg:
                        observation = tools[tool_name](arg)
                    else:
                        observation = tools[tool_name]()
                    print(f"🔧 执行工具 '{tool_name}' 得到观察：{observation}")
                except Exception as e:
                    observation = f"工具执行出错：{e}"
                    print(f"❌ {observation}")
            else:
                observation = f"未知工具：{tool_name}"
                print(f"❌ {observation}")
            # 将模型回复和观察结果加入对话
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # 如果模型没有输出 Action，则可能是直接回答了，但缺少 Final Answer，我们需要追问
            print("⚠️ 模型未按格式输出 Action，尝试将其作为最终回答...")
            messages.append({"role": "assistant", "content": reply})
            # 再次追问
            messages.append({"role": "user", "content": "请以 Final Answer: 格式给出最终答案，或提供 Action。"})
            continue

    print("⚠️ 达到最大轮数，未得到最终答案")
    return None

# ---------- 主程序 ----------
if __name__ == "__main__":
    question = "现在几点？请计算 123 + 456 等于多少？"
    print(f"用户问题：{question}")
    run_agent(question, max_rounds=3)