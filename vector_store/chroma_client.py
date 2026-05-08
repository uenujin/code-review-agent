"""
Chroma 向量数据库客户端。
设计理由：使用 LangChain 的 Chroma 封装，统一向量化与检索接口；
          持久化存储保证历史案例跨进程可用；
          单例模式避免重复建立数据库连接。
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


def _build_embeddings():
    """
    使用 FakeEmbeddings 做演示，生产中替换为真实嵌入模型
    （如 OpenAIEmbeddings、HuggingFaceEmbeddings 等）。
    FakeEmbeddings 生成固定维度随机向量，足以验证检索流程。
    """
    return FakeEmbeddings(size=1536)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    返回全局唯一的 Chroma 实例（懒加载 + 缓存）。
    lru_cache 保证整个进程生命周期内只创建一次连接。
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_build_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
