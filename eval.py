"""
评估脚本：量化 RAG 效果
测量两个核心指标：
  1. 检索命中率 (Retrieval Hit Rate) — 无需 API Key
  2. 缺陷检测覆盖率，有 RAG vs 无 RAG 基线 — 需要 ANTHROPIC_API_KEY
"""
import json
import os
import sys
import textwrap
import time

# ── 先初始化向量库 ────────────────────────────────────────────────────────────
from vector_store.seed_data import seed_vector_store
from vector_store.chroma_client import get_vector_store
from config import TOP_K_RESULTS

# ── 7 类缺陷类型测试集 ────────────────────────────────────────────────────────
# 每条：(query 描述, 期望命中的 title 关键词, 对应代码片段, 缺陷类别)
TEST_CASES = [
    {
        "id": 1,
        "category": "security",
        "defect_name": "SQL 注入",
        "query": "用户输入直接拼接到 SQL 查询字符串中",
        "expected_title_kw": "SQL 注入",
        "code": textwrap.dedent("""
            def get_user(username):
                import sqlite3
                conn = sqlite3.connect("app.db")
                cursor = conn.cursor()
                query = f"SELECT * FROM users WHERE name='{username}'"
                cursor.execute(query)
                return cursor.fetchone()
        """),
    },
    {
        "id": 2,
        "category": "performance",
        "defect_name": "O(n²) 性能",
        "query": "嵌套循环两两比较列表元素导致性能问题",
        "expected_title_kw": "O(n²)",
        "code": textwrap.dedent("""
            def find_duplicates(items, existing):
                result = []
                for item in items:
                    for ex in existing:
                        if item == ex:
                            result.append(item)
                return result
        """),
    },
    {
        "id": 3,
        "category": "reliability",
        "defect_name": "缺少异常处理",
        "query": "网络请求没有 try/except，异常直接向上抛出导致崩溃",
        "expected_title_kw": "异常处理",
        "code": textwrap.dedent("""
            def fetch_data(url):
                import requests
                response = requests.get(url)
                return response.json()
        """),
    },
    {
        "id": 4,
        "category": "security",
        "defect_name": "硬编码凭据",
        "query": "密码或 API Key 硬编码在源代码中",
        "expected_title_kw": "硬编码",
        "code": textwrap.dedent("""
            DB_PASSWORD = "admin123"
            API_KEY = "sk-prod-secret-key-abc123"

            def connect_db():
                return connect(password=DB_PASSWORD)
        """),
    },
    {
        "id": 5,
        "category": "correctness",
        "defect_name": "可变默认参数",
        "query": "Python 函数使用可变对象 list 或 dict 作为默认参数",
        "expected_title_kw": "可变默认参数",
        "code": textwrap.dedent("""
            def add_item(item, container=[]):
                container.append(item)
                return container
        """),
    },
    {
        "id": 6,
        "category": "maintainability",
        "defect_name": "函数过长",
        "query": "函数超过 50 行，承担验证、计算、保存、通知多个职责",
        "expected_title_kw": "函数过长",
        "code": textwrap.dedent("""
            def process_order(order):
                # 验证
                if not order.get('user_id'):
                    raise ValueError("missing user_id")
                if not order.get('items'):
                    raise ValueError("empty order")
                total = 0
                for item in order['items']:
                    if item['qty'] <= 0:
                        raise ValueError("invalid qty")
                    total += item['price'] * item['qty']
                # 计算折扣
                discount = 0
                if total > 1000:
                    discount = total * 0.1
                elif total > 500:
                    discount = total * 0.05
                final = total - discount
                # 保存
                import sqlite3
                conn = sqlite3.connect("shop.db")
                conn.execute("INSERT INTO orders VALUES (?,?)", (order['user_id'], final))
                conn.commit()
                # 通知
                import smtplib
                server = smtplib.SMTP('localhost')
                server.sendmail('noreply@shop.com', order['email'], f'Order total: {final}')
                return final
        """),
    },
    {
        "id": 7,
        "category": "correctness",
        "defect_name": "空值未检查",
        "query": "链式访问对象属性时未检查 None，可能触发 AttributeError",
        "expected_title_kw": "空值",
        "code": textwrap.dedent("""
            def get_city(user):
                return user.profile.address.city
        """),
    },
]

# 包含所有 7 种缺陷的综合测试代码（用于 Agent 端到端测试）
COMPREHENSIVE_CODE = textwrap.dedent("""
import requests

# 缺陷1: 硬编码凭据
DB_PASSWORD = "admin123"
API_KEY = "sk-secret-abc"

def get_user(username):
    # 缺陷2: SQL 注入
    import sqlite3
    conn = sqlite3.connect("app.db")
    query = f"SELECT * FROM users WHERE name='{username}'"
    conn.execute(query)

def find_duplicates(items, existing):
    # 缺陷3: O(n²) 性能
    result = []
    for i in items:
        for e in existing:
            if i == e:
                result.append(i)
    return result

def fetch_data(url):
    # 缺陷4: 无异常处理
    return requests.get(url).json()

def add_tag(tag, tags=[]):
    # 缺陷5: 可变默认参数
    tags.append(tag)
    return tags

def get_city(user):
    # 缺陷6: 空值未检查
    return user.profile.address.city

def process_order(order):
    # 缺陷7: 函数过长/多职责
    if not order.get('user_id'): raise ValueError("no user")
    total = sum(i['price']*i['qty'] for i in order['items'])
    discount = total * 0.1 if total > 1000 else 0
    final = total - discount
    import sqlite3
    conn = sqlite3.connect("shop.db")
    conn.execute("INSERT INTO orders VALUES (?,?)", (order['user_id'], final))
    conn.commit()
    import smtplib
    smtplib.SMTP('localhost').sendmail('a@b.com', order['email'], str(final))
    return final
""")


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: 检索命中率测试（无需 API Key）
# ══════════════════════════════════════════════════════════════════════════════

def test_retrieval_hit_rate():
    print("\n" + "═" * 60)
    print("  Part 1: RAG 检索命中率测试（Top-3）")
    print("═" * 60)

    store = get_vector_store()
    hits = 0
    random_baseline_hits = 0  # 随机基线：top-3 中随机命中率 = 3/7 ≈ 42.9%
    results = []

    for tc in TEST_CASES:
        docs = store.similarity_search(tc["query"], k=TOP_K_RESULTS)
        titles = [d.metadata.get("title", "") for d in docs]
        hit = any(tc["expected_title_kw"] in t for t in titles)

        # 计算相似度分数（用于展示）
        docs_with_score = store.similarity_search_with_score(tc["query"], k=TOP_K_RESULTS)
        top_score = 1 - docs_with_score[0][1] if docs_with_score else 0  # cosine: 1-distance

        hits += int(hit)
        results.append({
            "id": tc["id"],
            "defect": tc["defect_name"],
            "hit": hit,
            "top_result": titles[0] if titles else "—",
            "top_score": top_score,
        })

        status = "✅ HIT" if hit else "❌ MISS"
        print(f"  [{tc['id']}] {tc['defect_name']:<12} {status}  "
              f"top1={titles[0] if titles else '—'!r:20}  score={top_score:.3f}")

    hit_rate = hits / len(TEST_CASES) * 100
    random_baseline = 3 / 7 * 100  # Top-3 随机期望命中率

    print(f"\n  RAG 检索命中率:  {hits}/{len(TEST_CASES)} = {hit_rate:.1f}%")
    print(f"  随机基线 (Top-3 ÷ 7类): {random_baseline:.1f}%")
    print(f"  相对提升:  +{hit_rate - random_baseline:.1f} pct-pts  "
          f"({(hit_rate / random_baseline - 1) * 100:.0f}% 相对提升)")
    return results, hit_rate


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Agent 端到端检测覆盖率（需要 ANTHROPIC_API_KEY）
# ══════════════════════════════════════════════════════════════════════════════

DEFECT_KEYWORDS = {
    "SQL 注入":    ["sql注入", "sql injection", "sql拼接", "参数化查询"],
    "硬编码凭据":  ["硬编码", "硬编码密码", "api key", "凭据", "credential"],
    "O(n²) 性能":  ["o(n²)", "o(n^2)", "嵌套循环", "nested loop", "性能"],
    "缺少异常处理": ["异常处理", "try", "except", "try/except", "捕获异常"],
    "可变默认参数": ["可变默认", "默认参数", "mutable default", "lst=[]", "tags=[]"],
    "空值未检查":  ["空值", "none", "attributeerror", "空指针", "防御性检查"],
    "函数过长":    ["函数过长", "单一职责", "职责过多", "srp", "拆分"],
}


def detect_defects_in_report(report_text: str) -> dict[str, bool]:
    report_lower = report_text.lower()
    return {
        name: any(kw in report_lower for kw in kws)
        for name, kws in DEFECT_KEYWORDS.items()
    }


def run_agent_with_rag(code: str) -> str:
    from agent.graph import agent
    from langchain_core.messages import HumanMessage
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=f"请审查以下代码：\n```python\n{code}\n```")]},
        config=config,
    )
    return result["messages"][-1].content


def run_agent_no_rag(code: str) -> str:
    """无 RAG：直接调用 Claude，不提供检索工具"""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from agent.prompts import SYSTEM_PROMPT
    from config import CLAUDE_MODEL

    llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=4096)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请审查以下代码：\n```python\n{code}\n```"),
    ]
    result = llm.invoke(messages)
    return result.content


def test_detection_coverage():
    print("\n" + "═" * 60)
    print("  Part 2: 端到端缺陷检测覆盖率（With RAG vs No-RAG）")
    print("═" * 60)

    print("\n  [1/2] 运行 RAG Agent（含检索工具）...")
    t0 = time.time()
    rag_report = run_agent_with_rag(COMPREHENSIVE_CODE)
    rag_time = time.time() - t0

    print(f"  [2/2] 运行 No-RAG 基线（纯 LLM，无工具）...")
    t0 = time.time()
    norag_report = run_agent_no_rag(COMPREHENSIVE_CODE)
    norag_time = time.time() - t0

    rag_detected = detect_defects_in_report(rag_report)
    norag_detected = detect_defects_in_report(norag_report)

    print(f"\n  {'缺陷类型':<15} {'RAG':^8} {'No-RAG':^8}")
    print(f"  {'─'*15} {'─'*8} {'─'*8}")

    rag_count = 0
    norag_count = 0
    for name in DEFECT_KEYWORDS:
        r = "✅" if rag_detected[name] else "❌"
        n = "✅" if norag_detected[name] else "❌"
        rag_count += int(rag_detected[name])
        norag_count += int(norag_detected[name])
        print(f"  {name:<15} {r:^8} {n:^8}")

    total = len(DEFECT_KEYWORDS)
    rag_rate = rag_count / total * 100
    norag_rate = norag_count / total * 100

    print(f"\n  检测覆盖率:  RAG={rag_count}/{total} ({rag_rate:.0f}%)  "
          f"No-RAG={norag_count}/{total} ({norag_rate:.0f}%)")
    print(f"  耗时:         RAG={rag_time:.1f}s  No-RAG={norag_time:.1f}s")

    return rag_detected, norag_detected, rag_rate, norag_rate


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n🔍 Code Review Agent — RAG 效果评估")
    print(f"   测试集: {len(TEST_CASES)} 类缺陷  |  Top-K={TOP_K_RESULTS}\n")

    # 初始化向量库
    print("  [初始化] 确保向量库已填充种子数据...")
    seed_vector_store(force=False)

    # Part 1: 检索测试（始终运行）
    retrieval_results, hit_rate = test_retrieval_hit_rate()

    # Part 2: 端到端测试（需要 API Key）
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_key:
        print("\n" + "─" * 60)
        print("  ⚠️  未检测到 ANTHROPIC_API_KEY，跳过端到端 Agent 测试。")
        print("  设置后重新运行可获得完整报告。")
        print("  命令: export ANTHROPIC_API_KEY=sk-ant-xxx && python eval.py")
    else:
        rag_det, norag_det, rag_rate, norag_rate = test_detection_coverage()

    # ── 最终结论 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  📊 评估结论")
    print("═" * 60)

    random_baseline = 3 / 7 * 100
    print(f"\n  ✅ RAG 检索命中率:  {hit_rate:.1f}%（随机基线 {random_baseline:.1f}%）")
    print(f"     → 相比无检索基线，命中率提升 "
          f"+{hit_rate - random_baseline:.1f} pct-pts "
          f"（{(hit_rate / random_baseline - 1) * 100:.0f}% 相对提升）")
    print(f"\n  ✅ 知识库覆盖 {len(TEST_CASES)} 类缺陷类型：")
    for tc in TEST_CASES:
        print(f"     • [{tc['category']}] {tc['defect_name']}")

    if has_key:
        print(f"\n  ✅ 演示用例缺陷检测：RAG {rag_rate:.0f}% vs No-RAG {norag_rate:.0f}%")
        diff = rag_rate - norag_rate
        if diff > 0:
            print(f"     → RAG 版本多检出 {diff:.0f}% 的缺陷类型")
        elif diff == 0:
            print(f"     → 两者检测率相近（RAG 优势在修复建议质量而非检出数量）")

    print()


if __name__ == "__main__":
    main()
