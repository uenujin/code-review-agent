"""
Chroma 向量数据库客户端。
设计理由：使用 LangChain 的 Chroma 封装，统一向量化与检索接口；
          持久化存储保证历史案例跨进程可用；
          单例模式避免重复建立数据库连接。
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


def _build_embeddings():
    """
    多语言语义嵌入模型，支持中英文混合检索。
    paraphrase-multilingual-MiniLM-L12-v2 覆盖 50+ 语言，首次运行自动下载（约 470 MB）。
    """
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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
