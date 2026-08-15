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
    base_url=config.BASE_URL
)

# 全局变量：当前正在使用的 混合检索器 和 Agent
# rebuild() 会替换这两个值，ask() 读最新值 → 上传后立刻换库问答
ensemble = None
agent = None


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
# retrieve 是模块级函数，通过全局 ensemble 变量访问当前知识库
# （改成模块级是为了能单独测试它，也避免每次 build_agent 重新定义函数）
@tool  # 装饰器，将下面的普通函数升级成 LangChain Tool (自动提取函数签名、docstring 发给模型)
def retrieve(query: str) -> str:
    """根据用户问题，从当前知识库检索相关资料。"""  # docstring 必须是普通字符串，不能是 f-string
    try:
        # 空库状态：还没上传任何文档，直接告诉模型"没有知识库"
        if ensemble is None:
            print("\n⚠️ 当前没有知识库（用户还未上传文档）\n")
            return "当前还没有任何知识库。"
        docs = ensemble.invoke(query)
        print("\n🔍 已检索知识库\n")
        return "\n\n".join(d.page_content for d in docs)  # 把检索到的几段拼成一个长字符串返回（工具返回只能是字符串，模型才好读）
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
            "4. 如果检索到了资料，只根据这些资料回答；资料里没有就直说'库里没有'，禁止编造。"
        )
    )
    return agent


''' 第3段：init() 启动初始化（空库起步，等用户上传文档）'''
def init():
    global ensemble, agent
    # 不再自动加载默认知识库 —— 空库起步
    # ensemble=None 表示"还没有任何知识库"，retrieve 工具会返回对应提示
    ensemble = None
    agent = build_agent()  # 构建 Agent（闲聊可直接答，知识类问题引导上传）
    print("✅ Agent 就绪（暂无知识库，上传文档后可提问）")


''' 第4段：rebuild() 上传新文档后重建整个问答链路 '''
def rebuild(doc_paths):
    global ensemble, agent
    print("🔄 正在解析文档并重建知识库...")
    # 清空旧库 + 重新建库（reset=True 会先删掉旧 collection，避免新旧混一起）
    vs = retrieval_core.build_vectorstore(doc_paths, reset=True)
    ensemble = build_ensemble(vs)                       # 重建混合检索
    agent = build_agent()                               # 重建 Agent
    print(f"✅ 重建完成，共 {len(doc_paths)} 个文档已入库，可以提问了")

''' 第5段：模块导入时自动初始化（命令行/API/网页 import 后直接用）'''
init()

''' 第6段：可复用的 ask() 函数（API / 网页都可以 import 它）'''

MAX_HISTORY = 10  # 滑动窗口：最多携带 10 条历史消息（≈5 轮对话），防上下文无限膨胀


def ask(question, history):
    """单轮问答：历史 + 当前问题 → 回答。

    ask() 是给外部用的"接口函数"：命令行、FastAPI、网页全都调它，
    保持一套问答逻辑到处复用。
    滑动窗口：只取最近 MAX_HISTORY 条历史，太长的对话自动丢弃最旧的，
    避免消息无限堆积把模型上下文窗口撑爆。
    """
    recent = history[-MAX_HISTORY:]  # 只保留最近 N 条历史（丢弃最早的消息）
    messages = recent + [{"role": "user", "content": question}]  # 最近历史 + 用户这一句（这就是"记忆"）
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
