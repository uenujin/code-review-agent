"""
代码审查 Agent 入口。

使用方式：
  python main.py                     # 交互式 CLI
  python main.py --seed              # 仅初始化向量库
  python main.py --demo              # 运行内置演示案例

LangSmith 追踪：设置 .env 中的 LANGCHAIN_TRACING_V2=true 即可，
所有 LangGraph 运行自动上报到 LangSmith 项目。
"""
import argparse
import sys
import uuid

from langchain_core.messages import HumanMessage

from agent.graph import agent
from vector_store.seed_data import seed_vector_store

# ── 演示用代码片段（包含多种典型问题）─────────────────────────────────────
DEMO_CODE = '''
import requests

DB_PASSWORD = "admin123"   # 硬编码密码

def get_user(username):
    """从数据库查询用户"""
    import sqlite3
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # 直接拼接 SQL
    cursor.execute(f"SELECT * FROM users WHERE name='{username}'")
    return cursor.fetchone()

def find_duplicates(items, existing=[]):
    """找出 items 中与 existing 重复的元素"""
    result = []
    for item in items:
        for e in existing:
            if item == e:
                result.append(item)
    return result

def fetch_data(url):
    """拉取远程数据"""
    response = requests.get(url)
    return response.json()["data"]
'''.strip()


def run_review(code: str, thread_id: str | None = None) -> str:
    """
    对输入代码执行一次完整的 ReAct review 流程。

    Args:
        code: 待审查的代码字符串
        thread_id: 会话 ID，用于 checkpointer 区分不同对话；
                   None 时自动生成，确保每次独立运行不共享历史。

    Returns:
        Agent 生成的完整审查报告（最后一条 AI 消息内容）
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # LangGraph 通过 config["configurable"]["thread_id"] 路由到正确的 checkpoint
    config = {"configurable": {"thread_id": thread_id}}

    user_message = f"请对以下代码进行全面的代码审查：\n\n```\n{code}\n```"

    print("\n⏳ Agent 正在分析代码...\n")

    # 流式输出：逐步展示 Agent 的推理过程和工具调用
    final_message = ""
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if messages:
            last = messages[-1]
            # 打印非用户消息的类型标注（便于观察 ReAct 步骤）
            kind = type(last).__name__
            if kind == "AIMessage":
                # 仅显示有文本内容的 AI 消息（过滤纯工具调用消息）
                text = last.content if isinstance(last.content, str) else ""
                if text:
                    print(f"[Agent]\n{text}\n{'─' * 60}")
                    final_message = text
            elif kind == "ToolMessage":
                print(f"[Tool: {last.name}] 检索完成，获得 {len(last.content)} 字符的结果\n")

    return final_message


def interactive_mode():
    """交互式 CLI：支持多轮代码输入，输入 'quit' 退出。"""
    print("=" * 60)
    print("  代码审查 Agent（LangGraph + Claude + Chroma）")
    print("  输入代码后按两次 Enter 开始审查，输入 quit 退出")
    print("=" * 60)

    while True:
        print("\n📝 请粘贴待审查代码（输入空行结束）：")
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                if line.lower() == "quit":
                    print("👋 退出。")
                    sys.exit(0)
                lines.append(line)
        except EOFError:
            break

        code = "\n".join(lines).strip()
        if not code:
            print("⚠️  代码为空，跳过。")
            continue

        report = run_review(code)
        if not report:
            print("⚠️  Agent 未返回报告，请检查 API Key 配置。")


def main():
    parser = argparse.ArgumentParser(description="代码审查 Agent")
    parser.add_argument("--seed", action="store_true", help="初始化向量库（导入种子案例）")
    parser.add_argument("--demo", action="store_true", help="运行内置演示案例")
    args = parser.parse_args()

    # 始终确保向量库已初始化（幂等，已有数据时跳过）
    seed_vector_store()

    if args.seed:
        print("✅ 向量库初始化完成。")
        return

    if args.demo:
        print("🔍 运行演示案例...\n")
        print("=" * 60)
        print("待审查代码：")
        print(DEMO_CODE)
        print("=" * 60)
        run_review(DEMO_CODE)
        return

    # 默认：交互模式
    interactive_mode()


if __name__ == "__main__":
    main()
