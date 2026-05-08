"""
历史 review 案例种子数据。
设计理由：真实生产中这些案例来自过往 PR review 记录；
          这里预置典型问题类型，涵盖安全、性能、可维护性三大维度，
          让 Agent 在首次运行时就有参考基础。
"""
from langchain_core.documents import Document

from vector_store.chroma_client import get_vector_store

# 每条文档包含：问题描述（用于语义检索）+ 结构化元数据（用于报告渲染）
SEED_CASES: list[Document] = [
    Document(
        page_content=(
            "SQL 拼接字符串导致注入漏洞。代码直接将用户输入拼入查询语句，"
            "攻击者可构造恶意字符串绕过认证或读取敏感数据。"
        ),
        metadata={
            "title": "SQL 注入漏洞",
            "severity": "critical",
            "category": "security",
            "fix": "使用参数化查询（PreparedStatement / ORM）替代字符串拼接。",
            "example_bad": "query = f\"SELECT * FROM users WHERE name='{user_input}'\"",
            "example_good": "cursor.execute('SELECT * FROM users WHERE name=?', (user_input,))",
        },
    ),
    Document(
        page_content=(
            "嵌套循环导致 O(n²) 时间复杂度。对列表两两比较使用双层 for 循环，"
            "数据量增大时性能急剧下降。"
        ),
        metadata={
            "title": "O(n²) 性能问题",
            "severity": "high",
            "category": "performance",
            "fix": "使用哈希集合（set/dict）将内层循环降至 O(1)，整体降至 O(n)。",
            "example_bad": "for i in items:\n    for j in existing:\n        if i == j: ...",
            "example_good": "existing_set = set(existing)\nfor i in items:\n    if i in existing_set: ...",
        },
    ),
    Document(
        page_content=(
            "未处理异常，函数直接抛出未捕获错误导致程序崩溃。"
            "网络请求、文件操作等外部 IO 均应有错误处理路径。"
        ),
        metadata={
            "title": "缺少异常处理",
            "severity": "medium",
            "category": "reliability",
            "fix": "使用 try/except 捕获预期异常，记录日志并返回合理默认值或错误码。",
            "example_bad": "data = requests.get(url).json()",
            "example_good": "try:\n    data = requests.get(url, timeout=5).json()\nexcept (RequestException, ValueError) as e:\n    logger.error(e); return None",
        },
    ),
    Document(
        page_content=(
            "硬编码密码或 API Key 写在源码中，提交到版本控制后会泄露凭据。"
        ),
        metadata={
            "title": "硬编码凭据",
            "severity": "critical",
            "category": "security",
            "fix": "从环境变量或 Secret Manager 读取敏感配置，绝不写入代码库。",
            "example_bad": "DB_PASSWORD = 'super_secret_123'",
            "example_good": "DB_PASSWORD = os.environ['DB_PASSWORD']",
        },
    ),
    Document(
        page_content=(
            "可变对象作为函数默认参数，所有调用共享同一实例，导致状态污染。"
            "在 Python 中使用 list/dict 作为默认参数是常见陷阱。"
        ),
        metadata={
            "title": "可变默认参数陷阱",
            "severity": "medium",
            "category": "correctness",
            "fix": "将默认值设为 None，在函数体内初始化新对象。",
            "example_bad": "def append_to(item, lst=[]):\n    lst.append(item); return lst",
            "example_good": "def append_to(item, lst=None):\n    if lst is None: lst = []\n    lst.append(item); return lst",
        },
    ),
    Document(
        page_content=(
            "函数过长（超过 50 行），承担多个职责，违反单一职责原则，"
            "难以测试和复用。"
        ),
        metadata={
            "title": "函数过长 / 职责过多",
            "severity": "low",
            "category": "maintainability",
            "fix": "按职责拆分为若干小函数，每个函数只做一件事，便于单元测试。",
            "example_bad": "def process_order(order):\n    # 100 lines: validate + compute + save + notify",
            "example_good": "validate_order(order)\nprice = compute_price(order)\nsave_order(order)\nnotify_customer(order)",
        },
    ),
    Document(
        page_content=(
            "对空值/None 未作防御性检查，在链式调用时容易触发 AttributeError 或 NullPointerException。"
        ),
        metadata={
            "title": "空值未检查",
            "severity": "medium",
            "category": "correctness",
            "fix": "在访问对象属性前检查是否为 None，或使用 Optional chaining / getattr 安全访问。",
            "example_bad": "user.profile.address.city",
            "example_good": "city = user.profile.address.city if user and user.profile and user.profile.address else ''",
        },
    ),
]


def seed_vector_store(force: bool = False) -> None:
    """
    将种子案例写入 Chroma。
    force=False 时检查集合是否已有数据，避免重复索引（幂等操作）。
    """
    store = get_vector_store()

    # 检查是否已初始化（Chroma 集合存在且非空则跳过）
    existing = store.get()
    if existing["ids"] and not force:
        print(f"[seed] 向量库已有 {len(existing['ids'])} 条案例，跳过重复导入。")
        return

    store.add_documents(SEED_CASES)
    print(f"[seed] 成功写入 {len(SEED_CASES)} 条历史 review 案例。")


if __name__ == "__main__":
    seed_vector_store(force=True)
