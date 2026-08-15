# 📚 Doc QA Agent · 文档问答智能体

> 企业级文档问答系统 —— 基于 **Agentic RAG**（检索增强生成）的端到端落地项目。
> 支持**用户自助上传文档**（PDF / Word / TXT），任意领域即传即问；
> 语义检索 + 关键词检索 + 模型自主决策，内置多轮对话记忆。
> 2026 年技术栈：LangChain 1.x + LangGraph 1.x + create_agent / @tool。

## ✨ 核心亮点

1. **用户上传即建库（多格式支持）**
   - 支持上传单个 / 多个文档：PDF（pypdf）、Word（docx2txt）、TXT / Markdown
   - 上传后现场解析 → 切块 → 向量化入库 → 自动重建问答 Agent
   - 前端带「正在解析中...」实时状态，解析完成才允许提问
   - **领域不限**：汽配、法律、财务、个人资料…传什么答什么
2. **Hybrid RAG（混合检索）**
   - 语义路：`BAAI/bge-m3` 向量检索，理解"换一种说法"的提问
   - 关键词路：`jieba` 中文分词 + `BM25` 精确检索
   - **RRF 融合**双路排名，兼顾语义理解与字面匹配
3. **Agentic 决策（create_agent + @tool）**
   - 检索封装成工具，由模型自主决定"要不要检索"
   - 知识类问题 → 检索知识库；闲聊 → 直接回答，不浪费工具调用
4. **防幻觉硬约束**
   - `system_prompt` 定死红线：必须检索、库里没有就直说、禁止编造
   - 实测 bad case：「火花塞价格」→ 如实回答"库里没有"，绝不瞎编
5. **多轮对话记忆**
   - 前端：历史对话显示在上方，下方是新的输入框
   - 后端：历史消息拼进 prompt 一起发送，Agent 记住上下文
   - 复用 `ask(question, history)`：命令行 / FastAPI / Streamlit 三端共用
6. **完整工程链路**
   - 云端部署：requirements 置仓库根 + Secrets 管密钥 + 启动自动重建向量库

---

## 🏗️ 项目结构

```
Doc-QA-Agent/            # 仓库根目录
├── config.py            # 配置层：API key / 模型 / 路径（BASE_DIR 绝对定位）
├── retrieval_core.py    # 核心层：多格式文档解析 + Embedding + 建库/加载
├── qa_agent.py          # 问答层：混合检索 @tool + Agent + ask() + rebuild()
├── api.py               # 接口层：FastAPI POST /ask + /upload（供外部系统调用）
├── app.py               # 展示层：Streamlit 网页（上传区 + 解析状态 + 对话记忆）
├── data/
│   └── 汽配知识介绍.txt  # 示例默认知识库（首次启动自动建库）
├── uploads/             # 用户上传的文档（运行时生成）
└── chroma_db/           # ChromaDB 持久化（git 忽略，云端自动重建）
```

## 🧠 架构流程

```
[用户上传 PDF/Word/TXT]          [用户提问]
        │                            │
        ▼                            ▼
 正在解析中...                 [Agent 自主决策]  ← 要不要检索？
  读取 → 切块                     │ yes        │ no
  → 向量化 → 入库                 ▼            ▼
        │                     [混合检索]     [直接回答]
        ▼                   bge-m3 + BM25      │
 重建问答 Agent                 └─ RRF 融合     │
        │                            │          │
        └──────────►  [基于资料回答]  ◄─────────┘
                      防幻觉：没有就说"库里没有"
```

---

## 🚀 本地运行

```bash
# 1. 安装依赖（仓库根目录）
pip install -r requirements.txt

# 2. 配置密钥（项目根 .env）
#    siliconflow_api=sk-xxxx

# 3. 运行（任选其一）
python retrieval_core.py   # 重建默认向量库
python qa_agent.py         # 命令行问答
uvicorn api:app --reload   # FastAPI 接口 → http://127.0.0.1:8000/docs
streamlit run app.py       # Streamlit 网页 → http://localhost:8501
```

### 网页版使用
1. 左侧**侧边栏**选择文档（可多选 PDF / Word / TXT）
2. 点**「上传并解析」**按钮，等待「解析完成」提示
3. 在下方输入框提问，Agent 基于你上传的文档回答
4. 历史对话显示在上方，可连续追问，Agent 记住上下文

## ☁️ 云端部署（Streamlit Community Cloud）

1. `requirements.txt` 放在**仓库根目录**（不是子目录！否则云端装不上依赖）
2. `.env` 不进仓库，改用平台 **Secrets** 填 `siliconflow_api = "sk-xxx"`
3. 向量库 `chroma_db/` 不提交 git，`qa_agent.py` 里加了**启动自动重建守卫**：
   ```python
   if not os.path.isdir(config.PERSIST_DIR):
       doc_path = str(config.BASE_DIR / "data" / "汽配知识介绍.txt")
       retrieval_core.build_vectorstore([doc_path])
   ```
4. Streamlit Cloud 中 Deploy：Main file path 填 `app.py`（项目文件就在仓库根目录）

> 部署坑总结：依赖文件位置 / Secrets 密钥 / 向量库空库——三个坑各有解法，见 qa_agent.py 顶部守卫。

---

## 🧪 实测案例（默认汽配库）

| 提问 | 行为 | 结果 |
|------|------|------|
| 刹车片多久换一次 | 检索知识库 | ✅ 给出更换周期/判断标准 |
| 打火的那个东西多久换一次 | 换说法 → 语义检索命中 | ✅ 正确识别"火花塞" |
| 火花塞价格一般多少 | 检索（库中无价格） | ✅ 直说"库里没有"，未编造 |
| 你好，你叫什么 | 不检索，直接回答 | ✅ 闲聊正常 |
| 上传法律文档后提问条款 | 检索新上传的库 | ✅ 围绕新领域回答 |

## 🛠️ 技术栈

Python · LangChain 1.x · LangGraph 1.x · create_agent / @tool · FastAPI · Streamlit · ChromaDB · BAAI/bge-m3 · jieba / BM25（rank-bm25）· RRF · pypdf · docx2txt · SiliconFlow · Git / GitHub · python-dotenv

## 📌 面试亮点速记

- **为什么 Hybrid**：向量检索强于"换说法"、BM25 强于"精确词"，RRF 融合兼顾
- **为什么 Agentic**：让模型自主判断是否检索，但防幻觉必须硬约束兜底
- **上传即建库**：多格式解析器 + 按格式分发加载器 + 重建 agent（`rebuild()`）
- **多轮记忆**：前端 session_state 存历史 + 后端拼进 messages 发送
- **部署三坑**：requirements 位置 / Secrets / 向量库自动重建
