# Streamlit 网页 —— 文档问答助手（支持上传文档 + 多轮记忆）

# 导入 Streamlit 库，用于构建 Web 交互界面
import streamlit as st

# 页面全局配置（可设置浏览器标签页标题和图标）
st.set_page_config(page_title="文档问答助手", page_icon="📚")

# 会话隔离：每个浏览器会话（st.session_state 本身）对应一个 session_id
# 不同用户打开页面互不影响——每个人的知识库和 Agent 独立（改进1：多用户隔离）
import uuid
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
# 复用同一套问答逻辑，包括检索和 LLM 调用
from qa_agent import ask, rebuild, init
# 只在"首次进入页面"时初始化该会话的 Agent（init 已幂等，重复调用也不删库）
# 注意：不能每次 rerun 都重设 session_id！否则上传的知识库会对不上号
if "agent_ready" not in st.session_state:
    init(st.session_state.session_id)
    st.session_state.agent_ready = True

# 在页面顶部显示一个大标题
st.title("📚 文档问答助手")

# ------------------ 侧边栏：文档上传区 ------------------
# with st.sidebar：以下控件都渲染到页面左侧的侧边栏里
with st.sidebar:
    st.header("📄 上传文档")
    st.caption("支持 PDF / Word / TXT，可多选。解析完成后即可提问。")
    st.info("当前无默认知识库，上传文档后才能回答知识类问题；闲聊可直接回复。")

    # st.file_uploader：文件上传控件
    # accept_multiple_files=True → 允许一次选多个文件
    # type=["pdf","docx","txt","md"] → 只允许这些格式
    uploaded_files = st.file_uploader(
        "选择文档",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    # 上传 + 解析按钮
    if st.button("🚀 上传并解析", type="primary"):
        if not uploaded_files:
            st.warning("请先选择要上传的文档")
        else:
            # st.status：显示"正在解析中..."的进度容器
            # expanded=True → 默认展开，能看到里面每一小步
            with st.status("正在解析文档，请稍候...", expanded=True) as status:
                try:
                    # 把上传的文件保存到本地磁盘（uploads/ 目录）
                    import os
                    import config
                    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

                    saved_paths = []  # 收集保存后的文件路径
                    # 改进4：后端校验——格式白名单 + 大小限制，不只靠前端 type 限制
                    ALLOWED = {".pdf", ".docx", ".txt", ".md"}
                    MAX_SIZE = 20 * 1024 * 1024
                    for f in uploaded_files:
                        ext = os.path.splitext(f.name or "")[1].lower()
                        if ext not in ALLOWED:
                            raise ValueError(f"不支持的文件格式：{f.name}（仅支持 PDF/Word/TXT/Markdown）")
                        content = f.getvalue()
                        if len(content) > MAX_SIZE:
                            raise ValueError(f"文件过大：{f.name}（上限 20MB）")
                        # f.name：文件名；content：文件的二进制内容
                        save_path = os.path.join(config.UPLOAD_DIR, f.name)
                        with open(save_path, "wb") as fp:  # wb = 写二进制
                            fp.write(content)
                        saved_paths.append(save_path)
                        status.write(f"✅ 已接收：{f.name}")

                    # 调用 rebuild：解析 → 切块 → 向量化入库 → 重建该会话的 Agent
                    status.write("🔄 正在解析、切块并写入向量库（首次较慢）...")
                    rebuild(saved_paths, session_id=st.session_state.session_id)

                    # 解析完成：状态标签变绿勾 ✓，并折叠起来
                    status.update(
                        label=f"✅ 解析完成！共 {len(saved_paths)} 个文档，可以提问了",
                        state="complete",
                        expanded=False,
                    )
                except Exception as e:
                    status.update(label="❌ 解析失败", state="error")
                    st.error(f"出错了：{e}")

# ------------------ 对话区 ------------------
# 检查 Streamlit 的会话状态中是否已有 "history" 键
# 会话状态（st.session_state）在每次用户交互时保持，用于跨重运行保存数据
if "history" not in st.session_state:
    st.session_state.history = []  # 初始化为空列表，存储对话历史（实现多轮记忆）

# 把历史对话全部渲染出来（显示在上方）
for msg in st.session_state.history:
    # st.chat_message：创建一个"消息气泡"容器
    # msg["role"] 决定气泡样式（user 左侧 / assistant 右侧）
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# st.chat_input：专为聊天对话场景设计的输入框
# 生成一个固定在底部的文本输入框（即"下方的新对话框"），返回用户输入内容
user_input = st.chat_input("上传文档后，在这里提问：")

# 如果用户输入了非空内容
if user_input:
    # 立即把用户这条消息显示出来
    with st.chat_message("user"):
        st.write(user_input)

    # 调用问答函数 ask，传入用户问题 + 当前会话历史（按会话隔离）
    # ask 返回 LLM 生成的回答（字符串）
    answer = ask(user_input, st.session_state.history, session_id=st.session_state.session_id)

    # 显示助手回答
    with st.chat_message("assistant"):
        st.write(answer)

    # 更新会话历史：把这一轮追加进历史列表
    # 下次提问时 ask 能收到完整上下文 → 实现多轮记忆
    st.session_state.history += [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ]
