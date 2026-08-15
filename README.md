# 📚 Doc QA Agent · 文档问答智能体

> 企业级文档问答系统 —— 基于 **Agentic RAG**（检索增强生成）的端到端落地项目。
> 技术栈：LangChain 1.x · LangGraph 1.x · create_agent / @tool · FastAPI · Streamlit · ChromaDB

## 🔗 在线演示

[Streamlit Community Cloud](https://ai-application-learning-fksmjo4uuwt7ex9mzb3lyv.streamlit.app/)

## ✨ 功能亮点

| 能力 | 说明 |
|------|------|
| 🧩 **混合检索 (Hybrid RAG)** | `bge-m3` 向量检索（懂语义）+ `jieba`/`BM25` 关键词检索（精匹配），**RRF 融合**双路排名 |
| 🤖 **Agentic 决策** | 检索封装成 `@tool`，模型自主决定"要不要检索"——知识问题才检索，闲聊直接答 |
| 🛡️ **防幻觉硬约束** | system_prompt 定死红线：必须检索、库里没有就直说、禁止编造 |
| 🗺️ **双方案实现** | `create_agent`（轻量）与 `LangGraph` 条件分支（流程强控）两套架构 |
| 🔌 **三端复用** | 同一套 `ask()` 逻辑，命令行 / FastAPI / Streamlit 三端共用 |
| ☁️ **云端可部署** | requirements 置仓库根 + Secrets 管密钥 + 启动自动重建向量库 |

## 🏗️ 项目结构

```
Doc-QA-Agent/
├── config.py            # 配置层：API key / 模型 / 路径（BASE_DIR 绝对定位）
├── retrieval_core.py    # 核心层：Embedding 类 + 建库 + 加载向量库
├── qa_agent.py          # 问答层：混合检索 @tool + Agent + ask() + LangGraph 方案
├── api.py               # 接口层：FastAPI POST /ask（供外部系统调用）
├── app.py               # 展示层：Streamlit 网页
├── data/
│   └── 汽配知识介绍.txt  # 示例知识库文档（可替换为任意业务文档）
└── chroma_db/           # ChromaDB 持久化（git 忽略，云端自动重建）
```

## 🧠 架构流程

```
用户提问
   │
   ▼
[route / 自主决策]   ← 方案二 LangGraph：need_search 判断；方案一 create_agent 自主
   │ yes                      │ no
   ▼                          ▼
[混合检索]               [直接回答]
 bge-m3 + BM25                │
   └─ RRF 融合                │
   │                          │
   ▼                          ▼
[基于资料回答]  ← 防幻觉：只根据检索结果答，没有就说"库里没有"
```

## 🚀 本地运行

```bash
# 1. 安装依赖（仓库根目录）
pip install -r requirements.txt

# 2. 配置密钥（项目根 .env）
#    siliconflow_api=sk-xxxx

# 3. 运行（任选其一）
cd Doc-QA-Agent
python retrieval_core.py   # 重建向量库
python qa_agent.py         # 命令行问答
uvicorn api:app --reload   # FastAPI 接口 → http://127.0.0.1:8000/docs
streamlit run app.py       # Streamlit 网页 → http://localhost:8501
```

## ☁️ 云端部署（Streamlit Community Cloud）

1. `requirements.txt` 放在**仓库根目录**（不是子目录！否则云端装不上依赖）
2. `.env` 不进仓库，改用平台 **Secrets** 填 `siliconflow_api = "sk-xxx"`
3. 向量库 `chroma_db/` 不提交 git，`qa_agent.py` 里加了**启动自动重建守卫**：
   ```python
   if not os.path.isdir(config.PERSIST_DIR):
       doc_path = str(config.BASE_DIR / "data" / "汽配知识介绍.txt")
       retrieval_core.build_vectorstore([doc_path])
   ```
4. Streamlit Cloud 中 Deploy：Main file path 填 `Doc-QA-Agent/app.py`

> 部署坑总结：依赖文件位置 / Secrets 密钥 / 向量库空库——三个坑各有解法，见 `qa_agent.py` 顶部守卫。

## 🧪 实测案例

| 提问 | 行为 | 结果 |
|------|------|------|
| 刹车片多久换一次 | 检索知识库 | ✅ 给出更换周期/判断标准 |
| 打火的那个东西多久换一次 | 换说法 → 语义检索命中 | ✅ 正确识别"火花塞" |
| 火花塞价格一般多少 | 检索（库中无价格） | ✅ 直说"库里没有"，未编造 |
| 你好，你叫什么 | 不检索，直接回答 | ✅ 闲聊正常 |

## 🛠️ 技术栈

Python · LangChain 1.x · LangGraph 1.x · create_agent / @tool · FastAPI · Streamlit · ChromaDB · BAAI/bge-m3 · jieba / BM25（rank-bm25）· RRF · SiliconFlow · Git / GitHub · python-dotenv
