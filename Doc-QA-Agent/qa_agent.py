# qa_agent.py：问答层

# 核心思路：Agentic RAG（create_agent + @tool）
# 把"知识库混合检索"封装成一个 @tool 工具，让大模型自主决定是否检索。
# 防幻觉靠 system_prompt 硬约束，不让模型用常识编造。
#
# 本版本支持"上传文档即建库"，空库起步：
#   启动时   init()         → 空库初始化：没有默认知识库，闲聊可直接答，知识类问题引导上传
#   上传后   rebuild(路径s) → 重新建库 + 重建 Agent（问答立刻换到新知识库）
#   问答     ask(问题,历史) → 读全局 agent 变量（被 rebuild 替换后自动用新库）
#
# 代码分六段：
#   第1段  构建混合检索（BM25 + 向量库 + RRF 融合）→ 返回 ensemble
#   第2段  用 @tool 把 ensemble 包成 retrieve 工具，create_agent 创建 Agent
#   第3段  init()：空库初始化（没有知识库时 retrieve 返回提示，引导用户上传）
#   第4段  rebuild()：上传新文档后 重建向量库 + 重建 Agent
#   第5段  ask()：可复用接口函数（历史拼进 messages = 多轮记忆，含滑动窗口）
#   第6段  命令行对话循环


import sys
sys.stdout.reconfigure(encoding="utf-8")  # 修复 Windows 控制台中文乱码

import config
import retrieval_core

import jieba  # 中文分词库
from bm25_retriever import BM25Retriever  # 自实现的 BM25 检索器（不依赖 langchain-community）
from langchain_classic.retrievers.ensemble import EnsembleRetriever  # 检索器融合器
from langchain_core.tools import tool  # 工具装饰器
from langchain.agents import create_agent  # Agent 工厂
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(  # 大模型客户端（全局，rebuild 时复用）
    model=config.LLM_MODEL,
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
    max_retries=3,  # 改进5：失败重试（指数退避，OpenAI 客户端内置）
    timeout=60.0,   # 超时保护
)

# 全局变量：按"会话(session)"隔离的 混合检索器 和 Agent
# 之前是单例（ensemble/agent 各一个），多用户并发上传会互相覆盖；
# 现在改成 dict：key 是 session_id，每个用户/会话有自己的库和 Agent。
# rebuild(session_id) 只替换该会话的实例，不影响其他会话。
ENSEMBLES: dict = {}   # session_id -> EnsembleRetriever（混合检索）
AGENTS: dict = {}      # session_id -> Agent
DEFAULT_SESSION = "default"  # 不传 session_id 时用的会话（兼容旧调用）


''' 第1段：从向量库构建混合检索（BM25 + 向量 + RRF）→ 返回 ensemble '''
def build_ensemble(vs):
    # 从向量库中取出所有文档,只取"文本内容"那部分
    chunks = vs.get(include=["documents"])["documents"]

    # BM25 检索器（关键字检索器）
    def jieba_tokenize(text):  # 定义一个函数：将输入的文字分词
        return list(jieba.cut(text))  # 让 jieba 把 text 切成词（返回的是一个生成器），并将结果变成列表

    # 使用 BM25 算法创建检索器
    bm25 = BM25Retriever.from_texts(
        chunks,  # 切好的段落列表
        preprocess_func=jieba_tokenize  # 分词函数
    )
    bm25.k = 5  # BM25 每路返回 5 段，与向量路对齐

    # 向量库检索
    vector_retriever = vs.as_retriever(search_kwargs={"k": 5})  # 把向量库变成"检索器"，每路各取 5 段

    # RRF 融合检索（BM25 + 向量库）
    # (将两条路上的排名融合成一个新的排名，RRF 的强项是"融合")
    ensemble = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[1, 1])  # 两路检索器融合，各占一半权重
    return ensemble


''' 第2段：知识库检索工具 + 用 create_agent 创建 Agent → 返回 agent '''
# retrieve 是模块级函数，通过"当前会话"访问对应的知识库
# （改成模块级是为了能单独测试它，也避免每次 build_agent 重新定义函数）
# 注意：@tool 不能直接接收 session_id，所以 ask() 在调用前会把 CURRENT_SESSION
#       设成当前会话，retrieve 用 CURRENT_SESSION 去 ENSEMBLES 里取对应的检索器。
CURRENT_SESSION = DEFAULT_SESSION


@tool  # 装饰器，将下面的普通函数升级成 LangChain Tool (自动提取函数签名、docstring 发给模型)
def retrieve(query: str) -> str:
    """根据用户问题，从当前知识库检索相关资料。"""  # docstring 必须是普通字符串，不能是 f-string
    try:
        # 空库状态：该会话还没上传任何文档，直接告诉模型"没有知识库"
        ensemble = ENSEMBLES.get(CURRENT_SESSION)
        if ensemble is None:
            print("\n⚠️ 当前会话没有知识库（用户还未上传文档）\n")
            return "当前还没有任何知识库。"
        docs = ensemble.invoke(query)
        print("\n🔍 已检索知识库\n")
        # 把检索到的几段拼成长字符串返回，并带上来源（引用溯源，见 retrieve 的 metadata）
        parts = []
        for d in docs:
            src = d.metadata.get("source", "未知来源")
            parts.append(f"【来源：{src}】\n{d.page_content}")
        return "\n\n".join(parts)  # 工具返回只能是字符串，模型才好读
    except Exception as e:
        return f"检索失败：{e}"


def build_agent():
    # 官方 Agent 工厂
    agent = create_agent(
        model=llm,
        tools=[retrieve],
        system_prompt=(
            "你是文档问答助手。遵守以下规则：\n"
            "1. 闲聊（打招呼、问你是谁、寒暄等）或不需要知识库的问题，直接正常回答，不要调用工具。\n"
            "2. 知识类问题（需要查文档才能回答的事实性问题）必须调用 retrieve 工具，不能凭自己的常识回答。\n"
            "3. 如果 retrieve 返回'当前还没有任何知识库'，说明用户还没上传文档，"
            "请回答：'我还不知道这个问题哦，请上传你的文档让我学习学习吧'。\n"
            "4. 如果检索到了资料，只根据这些资料回答；资料里没有就直说'库里没有'，禁止编造。\n"
            "5. 回答末尾请标注引用来源：列出你依据的文档文件名（检索结果中的【来源：xxx】）。"
        )
    )
    return agent


''' 第3段：init() 初始化一个会话（空库起步，等用户上传文档）'''
def init(session_id=DEFAULT_SESSION):
    """确保某个会话可用：会话不存在则创建空库 Agent。
    幂等设计（重要）：不删已有知识库！
    之前这里有 ENSEMBLES.pop()，导致 Streamlit 每次 rerun 调用 init()
    时把用户刚上传的知识库清掉，表现为"上传了却一直提示空库"。已修复。
    """
    if session_id in AGENTS:
        return  # 会话已存在：保持现状（含已上传的知识库），不重建
    AGENTS[session_id] = build_agent()  # 创建 Agent（闲聊可直接答，知识类问题引导上传）
    print(f"✅ 会话 [{session_id}] Agent 就绪（暂无知识库，上传文档后可提问）")


''' 第4段：rebuild() 上传新文档后重建某个会话的问答链路 '''
def rebuild(doc_paths, session_id=DEFAULT_SESSION):
    print(f"🔄 会话 [{session_id}] 正在解析文档并重建知识库...")
    # 每个会话用独立的 collection（session_id 拼进 collection 名），
    # 这样 A 上传不会影响 B 的库（改进1 真正生效：不止内存隔离，存储也隔离）
    vs = retrieval_core.build_vectorstore(
        doc_paths, reset=True,
        collection_name=f"{config.COLLECTION_NAME}_{session_id}",
    )
    ENSEMBLES[session_id] = build_ensemble(vs)   # 重建该会话的混合检索
    AGENTS[session_id] = build_agent()           # 重建该会话的 Agent
    print(f"✅ 会话 [{session_id}] 重建完成，共 {len(doc_paths)} 个文档已入库，可以提问了")

''' 第5段：模块导入时自动初始化默认会话（命令行/API/网页 import 后直接用）'''
init()

''' 第6段：可复用的 ask() 函数（API / 网页都可以 import 它）'''

MAX_HISTORY = 10     # 滑动窗口：最多携带 10 条历史消息（≈5 轮对话），防上下文无限膨胀
SUMMARY_STEP = 20    # 触发摘要压缩的阈值：历史超过 20 条时，把最早的压成摘要
SUMMARY_BUDGET = 6   # 摘要保留的条数预算（摘要算 1 条消息）

def _summarize_history(messages):
    """改进2：长对话摘要压缩。
    记忆两级策略：
      ≤ MAX_HISTORY(10) 条        → 全量保留，不处理；
      MAX_HISTORY ~ SUMMARY_STEP(20) 条 → 纯滑动窗口截断（省 LLM 调用）；
      > SUMMARY_STEP 条           → 把最旧的压缩成一句摘要，保留"长期记忆 + 短期窗口"。
    """
    if len(messages) <= MAX_HISTORY:
        return messages
    # 超过 SUMMARY_STEP 才做摘要压缩；中间地带只用滑动窗口截断
    if len(messages) <= SUMMARY_STEP:
        return messages[-MAX_HISTORY:]
    # 最旧的 N 条拿去压缩成摘要
    old = messages[:-MAX_HISTORY]
    recent = messages[-MAX_HISTORY:]
    dialog = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    try:
        resp = llm.invoke(
            f"把下面的对话压缩成一句中文摘要（保留关键人物/事实/用户偏好，20 字以内）：\n{dialog}"
        )
        summary = resp.content.strip()
        return [{"role": "system", "content": f"【长期记忆摘要】{summary}"}] + recent
    except Exception:
        # 压缩失败就退回纯滑动窗口（丢弃最旧的），不让一次失败拖垮对话
        return recent


def ask(question, history, session_id=DEFAULT_SESSION):
    """单轮问答：历史 + 当前问题 → 回答。

    ask() 是给外部用的"接口函数"：命令行、FastAPI、网页全都调它，
    保持一套问答逻辑到处复用。
    session_id：哪个会话的库和 Agent 来答（多用户隔离的关键参数）。
    记忆两级：
      短期窗口：只取最近 MAX_HISTORY 条历史；
      长期记忆：超过 SUMMARY_STEP 条时，把最旧的压成一句摘要（见 _summarize_history），
                既保留长期上下文，又防止消息无限堆积撑爆上下文窗口。
    """
    global CURRENT_SESSION
    # 让 retrieve 工具知道该去哪个会话的库检索
    CURRENT_SESSION = session_id
    agent = AGENTS.get(session_id)
    if agent is None:
        init(session_id)  # 会话不存在则先初始化（懒加载）
        agent = AGENTS[session_id]

    messages = _summarize_history(history) + [{"role": "user", "content": question}]
    result = agent.invoke({"messages": messages})                 # 调 Agent（让它自主决定要不要检索）
    return result["messages"][-1].content                         # 取最后一条消息 = Agent 最终回答


''' 第7段：主程序入口 + 无限循环问答（本文件运行）'''
if __name__ == "__main__":  # 只有直接运行本文件才进来；被 import 时不执行 → API 不会被卡住
    history = []  # 累积对话历史

    # while True 无限循环语句, break 跳出循环
    while True:
        user_input = input("你问（输 exit/q 退出）：")
        if user_input.strip().lower() in ("exit", "q"):
            print("再见！")
            break
        answer = ask(user_input, history)  # 复用上面的 ask()，一套逻辑
        print("回答：", answer)
        # 记下这一轮，下次循环才有上下文
        history += [{"role": "user", "content": user_input},
                    {"role": "assistant", "content": answer}]
