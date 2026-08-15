# 📚 Doc QA Agent · 文档问答智能体

> 企业级文档问答系统 —— 基于 **Agentic RAG**（检索增强生成）的端到端落地项目。
> 支持**用户自助上传文档**（PDF / Word / TXT），任意领域即传即问；
> 语义检索 + 关键词检索 + 模型自主决策，内置多轮对话记忆。
> 2026 年技术栈：LangChain 1.x + create_agent / @tool + LangGraph（备选方案）。

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
4. **防幻觉硬约束 + 空库引导**
   - `system_prompt` 定死红线：必须检索、库里没有就直说、禁止编造
   - **空库起步**：未上传文档时，闲聊可答、知识类问题引导"请上传文档让我学习"
   - 实测 bad case：「火花塞价格」→ 如实回答"库里没有"，绝不瞎编
5. **多轮对话记忆**
   - 前端：历史对话显示在上方，下方是新的输入框
   - 后端：历史消息拼进 prompt 一起发送，Agent 记住上下文
   - **滑动窗口**：只带最近 10 条历史（≈5 轮），防长会话 token 无限膨胀
   - 复用 `ask(question, history)`：命令行 / FastAPI / Streamlit 三端共用
 6. **自动化测试（pytest）**
   - 单元测试：滑动窗口裁剪 / 空库工具返回值 / 检索失败兜底 / 多格式加载（不调 API，快）
   - 集成测试：空库引导 / 上传后问答 / 多轮记忆（调真实 API，`-m integration` 运行）
 7. **检索质量评估（Recall@k / MRR）**
   - `eval_retrieval.py`：20 条"问题→标准答案片段"测试集，量化对比纯向量 / 纯BM25 / RRF 融合
   - 结果：RRF 融合 MRR=0.950 > 纯BM25 0.908 > 纯向量 0.900，证明混合检索"取长补短"
   - 参数调优：`chunk_size=300`（MRR 0.950）优于 200（0.925）与 500（0.942），确认线上默认值
 8. **完整工程链路**
   - 云端部署：requirements 置仓库根 + Secrets 管密钥 + 空库起步、上传即建库

---

## 🏗️ 项目结构

```
Doc-QA-Agent/
├── config.py            # 配置层：API key / 模型 / 路径（BASE_DIR 绝对定位）
├── retrieval_core.py    # 核心层：多格式文档解析 + Embedding + 建库/加载
├── qa_agent.py          # 问答层：混合检索 @tool + Agent + ask() + rebuild()
├── api.py               # 接口层：FastAPI POST /ask + /upload（供外部系统调用）
├── app.py               # 展示层：Streamlit 网页（上传区 + 解析状态 + 对话记忆）
├── data/
│   └── 汽配知识介绍.txt  # 示例文档（可选，不自动建库，仅供测试上传）
├── eval_retrieval.py    # 检索质量评估：Recall@k / MRR，三路策略对比 + chunk_size 调参
├── uploads/             # 用户上传的文档（运行时生成）
└── chroma_db/           # ChromaDB 持久化（git 忽略，上传文档后生成）
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

# 3. 进入项目目录并运行（任选其一）
cd Doc-QA-Agent
python qa_agent.py         # 命令行问答
uvicorn api:app --reload   # FastAPI 接口 → http://127.0.0.1:8000/docs
streamlit run app.py       # Streamlit 网页 → http://localhost:8501
```

### 运行测试

```bash
# 单元测试（快，不调 API，验证滑动窗口/空库/多格式解析）
python -m pytest tests/ -v

# 集成测试（调真实 API，验证空库引导/上传问答/多轮记忆）
python -m pytest tests/test_integration.py -v -m integration
```

### 网页版使用
1. 左侧**侧边栏**选择文档（可多选 PDF / Word / TXT）
2. 点**「上传并解析」**按钮，等待「解析完成」提示
3. 在下方输入框提问，Agent 基于你上传的文档回答
4. 历史对话显示在上方，可连续追问，Agent 记住上下文

## ☁️ 云端部署（Streamlit Community Cloud）

1. `requirements.txt` 放在**仓库根目录**（不是子目录！否则云端装不上依赖）
2. `.env` 不进仓库，改用平台 **Secrets** 填 `siliconflow_api = "sk-xxx"`
3. 向量库 `chroma_db/` 不提交 git。系统**空库起步**：不自动加载默认知识库，用户上传文档时才动态建库
4. Streamlit Cloud 中 Deploy：Main file path 填 `Doc-QA-Agent/app.py`

> 部署坑总结：依赖文件位置 / Secrets 密钥 / 首次无库——三个坑各有解法。

---

## ⚠️ 已知限制（面试可聊改进方向）

1. **单用户设计**：`ensemble` / `agent` 是模块级全局变量，多用户并发上传会互相覆盖。
   改进方向：按会话（session）隔离，每个用户独立的 vectorstore + agent。
2. **历史窗口固定**：滑动窗口固定 10 条，长对话会丢弃最早内容。
   改进方向：用 LLM 把旧历史压缩成摘要，保留"长期记忆 + 短期窗口"。
3. **无引用溯源**：回答不标注来自哪个文档哪一段。
   改进方向：检索时保留 metadata 来源，回答末尾附引用。
4. **上传文件无后端校验**：只靠前端限制格式。
   改进方向：后端校验文件大小、格式白名单、防恶意文件。
5. **无失败重试**：LLM / Embedding 调用一次失败即报错。
   改进方向：指数退避重试 + 熔断。

---

## 🧪 实测案例

| 提问 | 场景 | 结果 |
|------|------|------|
| 你好 | 未上传文档 | ✅ 正常闲聊，友好回复 |
| 刹车片多久换一次 | 未上传文档 | ✅ 引导"我还不知道这个问题哦，请上传你的文档让我学习学习吧" |
| 刹车片多久换一次 | 已上传汽配文档 | ✅ 给出更换周期/厚度标准 |
| 打火的那个东西多久换一次 | 已上传文档 | ✅ 换说法 → 语义检索命中"火花塞" |
| 火花塞价格一般多少 | 已上传文档（库中无价格） | ✅ 直说"库里没有"，未编造 |

## 🛠️ 技术栈

Python · LangChain 1.x · LangGraph 1.x · create_agent / @tool · FastAPI · Streamlit · ChromaDB · BAAI/bge-m3 · jieba / BM25（rank-bm25）· RRF · pypdf · docx2txt · SiliconFlow · Git / GitHub · python-dotenv

> 💡 零弃用依赖：`langchain-community` 已于 2026-05 停服（sunset）。本项目自实现 `BM25Retriever`（`bm25_retriever.py`）与多格式加载器（pypdf / docx2txt / 原生读取），不依赖任何已停服包。

