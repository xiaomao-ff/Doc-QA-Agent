# langgraph_v2.py：方案二（LangGraph 条件分支）—— 强流程控的备选实现
# 与 qa_agent.py 方案一（create_agent 自主决策）的区别：
#   方案一：模型自主决定要不要调工具 —— 轻量、灵活
#   方案二：流程人为固定成"判断→检索→回答"的节点图 —— 可控、可插桩看日志
# 用法：cd Doc-QA-Agent && python langgraph_v2.py
# 注意：本文件独立运行，请先运行 qa_agent.py 的 rebuild() 建好知识库，
#       或本文件顶部自行加载已有向量库。
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
