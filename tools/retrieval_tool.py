"""
代码 Review 检索工具 — Agent 可调用的 LangChain Tool。
设计理由：将 Chroma 检索封装为 @tool，使 LangGraph ReAct 图能自动调度它。
          返回格式化字符串而非原始 Document 对象，便于 Claude 直接解读。
"""
import json
from langchain_core.tools import tool

from config import TOP_K_RESULTS
from vector_store.chroma_client import get_vector_store


@tool
def retrieve_similar_reviews(query: str) -> str:
    """
    根据代码问题描述，从历史 review 案例库中检索最相关的参考案例。

    当需要参考过去的 review 经验时调用此工具。
    输入应为对当前代码问题的简洁描述（中英文均可）。

    Args:
        query: 描述当前代码潜在问题的关键词或短句，
               例如 "SQL 字符串拼接"、"nested loop performance"、"未捕获异常"

    Returns:
        格式化的历史案例列表（JSON 字符串），包含标题、严重级别、修复建议等。
    """
    store = get_vector_store()

    # similarity_search 使用嵌入向量做语义相似度匹配
    docs = store.similarity_search(query, k=TOP_K_RESULTS)

    if not docs:
        return "未找到相关历史案例。"

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        results.append({
            "rank": i,
            "title": meta.get("title", "未知"),
            "severity": meta.get("severity", "unknown"),
            "category": meta.get("category", "unknown"),
            "description": doc.page_content,
            "fix": meta.get("fix", ""),
            "example_bad": meta.get("example_bad", ""),
            "example_good": meta.get("example_good", ""),
        })

    return json.dumps(results, ensure_ascii=False, indent=2)
