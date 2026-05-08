"""
配置模块 — 集中管理所有运行时参数。
设计理由：将配置与业务逻辑解耦，方便切换模型/向量库路径而无需改动主代码。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
# 使用 Opus 4.7 以获得最强推理能力，适合复杂代码分析任务
CLAUDE_MODEL = "claude-opus-4-7"

# --- Chroma ---
# 持久化路径：跨进程保留历史 review 案例，避免每次重新索引
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "code_reviews"

# --- Retrieval ---
# 每次最多召回 3 条相似案例：足以提供参考，不会淹没上下文
TOP_K_RESULTS = 3

# --- LangSmith (从环境变量自动读取，无需显式代码) ---
# LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
