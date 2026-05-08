"""
LangGraph ReAct 图定义 — 代码审查 Agent 的核心编排逻辑。

设计理由：
  - 使用 `create_react_agent` 预置图，避免手动构建 reason/act 节点；
    它已实现标准的 ReAct 循环：LLM → ToolCall? → ToolExecute → LLM → ...
  - MemorySaver checkpointer 支持多轮对话（可扩展），本示例用于演示架构；
  - LangSmith 追踪由环境变量自动启用，无需额外代码侵入。
"""
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent.prompts import SYSTEM_PROMPT
from config import CLAUDE_MODEL
from tools.retrieval_tool import retrieve_similar_reviews


def build_agent():
    """
    构建并返回 ReAct Agent 实例（编译后的 LangGraph CompiledGraph）。

    架构说明：
      LLM: Claude Opus 4.7（Anthropic 最强推理模型）
      Tools: [retrieve_similar_reviews]（可扩展）
      Checkpointer: MemorySaver（内存级持久化，生产用 SqliteSaver/PostgresSaver）

    ReAct 循环流程：
      User Input
        ↓
      [agent 节点] Claude 推理 → 决定是否调用工具
        ↓ (若调用)
      [tools 节点] 执行 retrieve_similar_reviews → 返回观察结果
        ↓
      [agent 节点] Claude 基于观察继续推理 → 直到生成最终报告
        ↓
      Final Output
    """
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        # 设置合理上限：代码 review 报告通常不超过 4096 tokens
        max_tokens=4096,
    )

    tools = [retrieve_similar_reviews]

    # MemorySaver 为每个 thread_id 维护对话历史
    checkpointer = MemorySaver()

    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,      # 注入系统提示词
        checkpointer=checkpointer,
    )

    return graph


# 全局单例 Agent，main.py 直接导入使用
agent = build_agent()
