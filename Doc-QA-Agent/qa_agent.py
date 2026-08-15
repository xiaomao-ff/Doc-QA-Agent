# qa_agent.py：问答层

# 方案一：LangChain 框架（Agentic RAG —— create_agent + @tool）
# 把"知识库混合检索"封装成一个 @tool 工具，让大模型自主决定
# 防幻觉靠 system_prompt 硬约束，不让模型用常识编造。
#
# 代码分四段：
#   第1段  加载已有向量库（Chroma）
#   第2段  混合检索 = BM25(关键字) + 向量库(语义)，RRF 融合成 ensemble，
#          再用 @tool 包成 retrieve 工具（try/except 兜底：工具永不崩）
#   第3段  create_agent 创建 Agent：模型 + 工具 + 防幻觉 system_prompt
#   第4段  命令行对话循环：把历史拼进 messages 一起发 = 多轮记忆


import sys
sys.stdout.reconfigure(encoding="utf-8")  # 修复 Windows 控制台中文乱码

import os
import config
import retrieval_core

''' 部署守卫：云端没有向量库时，启动自动重建（本地已有 chroma_db 则跳过）'''
if not os.path.isdir(config.PERSIST_DIR):
    print("🚀 首次启动：检测到向量库不存在，正在从源文档重建（约10-60秒）...")
    # 用 config.BASE_DIR 绝对定位源文档，避免云端运行目录不一致导致读不到文件
    doc_path = str(config.BASE_DIR / "data" / "汽配知识介绍.txt")
    retrieval_core.build_vectorstore([doc_path])

''' 第1段：加载已有向量库 '''
vs = retrieval_core.load_vectorstore()

# 从向量库中取出所有文档,只取"文本内容"那部分
chunks = vs.get(include=["documents"])["documents"] 

''' 第2段：混合检索工具(BM25 + 向量库 + RRF 融合 + @tool)'''
import jieba  # 中文分词库
from langchain_community.retrievers import BM25Retriever  # 关键字检索器


# BM25 检索器（关键字检索器）
def jieba_tokenize(text):  # 定义一个函数：将输入的文字分词
    return list(jieba.cut(text)) # 让 jieba 把 text 切成词（返回的是一个生成器），并将结果变成列表

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
from langchain_classic.retrievers.ensemble import EnsembleRetriever # 检索器融合器

ensemble = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[1, 1])  # 两路检索器融合，各占一半权重



# 知识库检索工具（ @tool ）
from langchain_core.tools import tool # 工具装饰器

@tool # 装饰器，将下面的普通函数升级成 LangChain Tool (自动提取函数签名、docstring 发给模型)
def retrieve(query: str) -> str:
    """根据用户问题，从汽车配件知识库检索相关资料。"""  # docstring 工具说明书
    try:
        docs = ensemble.invoke(query)
        print("\n🔍 已检索知识库\n")
        return "\n\n".join(d.page_content for d in docs)  # 把检索到的几段拼成一个长字符串返回（工具返回只能是字符串，模型才好读）
    except Exception as e:
        return f"检索失败：{e}"

''' 第3段：Agent '''
from langchain.agents import create_agent 
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    api_key=config.API_KEY,
    base_url=config.BASE_URL
)

# 官方 Agent 工厂
agent = create_agent(
    model=llm,
    tools=[retrieve],
    system_prompt="你是汽车配件专家。用户问汽车配件问题时，先检索知识库再回答；闲聊或非配件问题直接回答，不要检索；如果用户想了解某个配件细节，用工具检索，只检索不猜（禁止用你自己的常识补充或编造）。"
)

''' 第4段：可复用的 ask() 函数（API / 网页都可以 import 它）'''

def ask(question, history):
    """单轮问答：历史 + 当前问题 → 回答。

    ask() 是给外部用的"接口函数"：命令行、FastAPI、网页全都调它，
    保持一套问答逻辑到处复用。
    """
    messages = history + [{"role": "user", "content": question}]  # 历史 + 用户这一句（这就是"记忆"）
    result = agent.invoke({"messages": messages})                 # 调 Agent（让它自主决定要不要检索）
    return result["messages"][-1].content                         # 取最后一条消息 = Agent 最终回答


''' 第5段：主程序入口 + 无限循环问答（本文件运行）'''
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


# ================================================================
# 方案二：LangGraph 条件分支（Day28 思路）
# ----------------------------------------------------------------
# 与方案一的区别：
#   方案一（create_agent）：模型"自主决定"要不要调工具 —— 轻量、灵活
#   方案二（LangGraph）：把流程人为固定成"判断→检索→回答/直接答"的节点图，
#          每一步走哪个节点由代码说了算 —— 流程可控、可插桩看日志
# 选型话术：日常轻量问答用方案一；多步工单/审批这样的强流程场景用方案二。
# 用法说明：把下面三引号里的代码复制出去、删掉首尾的两个 ''' 就能运行
#   （ensemble 复用上文已建好的混合检索器）
# ================================================================

'''
# ---------------- 方案二（LangGraph 条件分支）----------------

# TypedDict：声明字典长什么样的工具
# Annotated：给函数参数加类型提示的工具(这里用来贴 reducer 的类型)
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages  # add_messages：消息追加规约器
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver  # 内存检查器：记忆存内存

# 1. 状态字典模板：图里每个节点都读写这份"状态病历本"
class QueryState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话记录，add_messages 自动追加
    context: list[str]   # 存放本轮检索到的上下文，默认新的覆盖旧的
    need_search: str     # route 节点写入的判决：'yes' 或 'no'

# 2. 路由节点：判断用户问题是否需要检索知识库
def route_node(state: QueryState) -> dict:
    # 从后往前找最近一条用户消息，没有则给默认问题（防空输入）
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    ) or "刹车片多久换一次"
    judge = llm.invoke(
        f"判断这个问题是否需要查询汽车配件知识库才能回答。"
        f"涉及配件具体信息(功能/类型/更换周期/保养)输出 yes，否则输出 no。"
        f"问题：{question}"
    )
    return {"need_search": judge.content.strip().lower()}

# 3. 检索节点：把混合检索结果塞进 context（复用上文的 ensemble）
def retrieve_node(state: QueryState) -> dict:
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    ) or "刹车片多久换一次"
    # 上下文寻回，只取文本内容
    docs = ensemble.invoke(question)
    return {"context": [d.page_content for d in docs]}

# 4. 回答节点（走检索路）：只根据检索到的资料回答，没有就说没有
def answer_node(state: QueryState) -> dict:
    docs_block = "\n\n".join(state["context"])
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    ) or "刹车片多久换一次"
    reply = llm.invoke([
        SystemMessage("只根据知识库检索到的资料回答，资料里没有就说'库里没有'，禁止编造。"),
        HumanMessage(f"资料：\n{docs_block}\n\n问题：{question}")
    ])
    return {"messages": [reply]}

# 5. 直接回答节点（不走检索路）：闲聊等直接答
def answer_direct_node(state: QueryState) -> dict:
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    ) or "你好"
    reply = llm.invoke([HumanMessage(question)])
    return {"messages": [reply]}

# 6. 岔路口函数：读 need_search 决定下一步去哪
def decide_route(state: QueryState) -> str:
    return "retrieve" if state["need_search"] == "yes" else "answer_direct"

# 7. 组装成图：节点 + 边 + 条件分支
builder = StateGraph(QueryState)
builder.add_node("route", route_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_node("answer_direct", answer_direct_node)

builder.add_edge(START, "route")
# 条件分支：decide_route 返回 'retrieve' → "retrieve"节点，'answer_direct' → 直接答节点
builder.add_conditional_edges("route", decide_route, {
    "retrieve": "retrieve",
    "answer_direct": "answer_direct"
})
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
builder.add_edge("answer_direct", END)

graph = builder.compile(checkpointer=InMemorySaver())  # 挂上检查器，支持多轮记忆

# 8. 对话循环：invoke 必带 config，thread_id 是会话钥匙
while True:
    user_input = input("你问（输 exit/q 退出）：")
    if user_input.strip().lower() in ("exit", "q"):
        print("再见！")
        break
    result = graph.invoke(
        {
            "messages": [HumanMessage(user_input)],
            "context": [],
            "need_search": ""
        },
        config={
            "recursion_limit": 10,                     # 防死循环
            "configurable": {"thread_id": "qa1"}       # 同一 id 才共享记忆
        }
    )
    print("回答：", result["messages"][-1].content)
'''
