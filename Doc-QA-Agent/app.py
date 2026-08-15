# Streamlit 网页 —— 汽配知识库助手

# 导入 Streamlit 库，用于构建 Web 交互界面
import streamlit as st

# 从自定义模块 qa_agent 中导入 ask 函数（复用同一套问答逻辑，包括检索和 LLM 调用）
from qa_agent import ask

# 页面全局配置（可设置浏览器标签页标题和图标）
st.set_page_config(page_title="汽配知识库助手", page_icon="🔧")

# 在页面顶部显示一个大标题，使用扳手表情符号作为装饰
st.title("🔧 汽配知识库助手")

# 检查 Streamlit 的会话状态中是否已有 "history" 键
# 会话状态（st.session_state）在每次用户交互时保持，用于跨重运行保存数据
if "history" not in st.session_state:
    # 如果没有，则初始化为一个空列表，用于存储对话历史（实现多轮记忆）
    st.session_state.history = []

# st.chat_input：专为聊天对话场景设计的输入框
# 它会生成一个固定在底部的文本输入框，并返回用户输入的内容（按下回车后触发）
# 参数为占位提示文字
user_input = st.chat_input("问问汽配知识：")

# 如果用户输入了非空内容
if user_input:
    # st.chat_message：创建一个“消息气泡”容器，用于显示对话消息
    # 参数 "user" 表示这是用户发送的消息（左侧显示，样式不同）
    with st.chat_message("user"):
        # 在消息气泡内显示用户输入的文本
        st.write(user_input)

    # 调用问答函数 ask，传入用户问题以及当前会话中保存的历史记录
    # ask 函数返回 LLM 生成的回答（字符串）
    answer = ask(user_input, st.session_state.history)

    # 创建一个助手消息气泡（"assistant" 表示 AI 回复，样式一般为右侧，带不同颜色）
    with st.chat_message("assistant"):
        # 在气泡中显示助手的回答
        st.write(answer)

    # 更新会话历史：将本轮的用户问题和助手回答追加到历史列表中
    # 这样下一次提问时，ask 函数就能接收到完整的对话上下文，实现记忆
    st.session_state.history += [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer}
    ]