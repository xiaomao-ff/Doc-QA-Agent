# test_qa_agent.py —— 问答层核心逻辑单元测试
# 只测"纯逻辑"：滑动窗口、摘要压缩、会话隔离、空库状态、引用溯源，不调真实 LLM（快、稳、省）
# 运行：cd Doc-QA-Agent && python -m pytest tests/ -v

import sys
import os
from pathlib import Path

# 保证能 import 到项目模块（tests/ 的上一级是项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qa_agent
import retrieval_core
import config


class FakeMessage:
    """模拟 agent 返回的消息对象（有 .content 属性）"""
    def __init__(self, content):
        self.content = content


''' ===== 1. 滑动窗口测试 ===== '''
class TestHistoryWindow:
    def test_ask_keeps_recent_history(self, monkeypatch):
        """ask() 应把历史传给 agent，且只保留最近 MAX_HISTORY 条"""
        captured = {}
        session = "test-sess-1"

        class FakeAgent:
            def invoke(self, payload):
                captured["messages"] = payload["messages"]
                return {"messages": [FakeMessage("ok")]}

        monkeypatch.setattr(qa_agent, "AGENTS", {session: FakeAgent()})

        # 造 20 条历史（超过 MAX_HISTORY=10）
        long_history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        qa_agent.ask("新问题", long_history, session_id=session)

        # 最终发给模型的 = 最近10条历史 + 当前问题 = 11条
        assert len(captured["messages"]) == 11
        # 最后一条是当前问题
        assert captured["messages"][-1] == {"role": "user", "content": "新问题"}
        # 历史部分是最新的10条（q10~q19）
        assert captured["messages"][0] == {"role": "user", "content": "q10"}

    def test_short_history_not_truncated(self, monkeypatch):
        """历史不足 MAX_HISTORY 时原样保留"""
        captured = {}
        session = "test-sess-2"

        class FakeAgent:
            def invoke(self, payload):
                captured["messages"] = payload["messages"]
                return {"messages": [FakeMessage("ok")]}

        monkeypatch.setattr(qa_agent, "AGENTS", {session: FakeAgent()})

        short_history = [{"role": "user", "content": "q1"}]
        qa_agent.ask("新问题", short_history, session_id=session)
        assert len(captured["messages"]) == 2


''' ===== 1.5 会话隔离测试（改进1） ===== '''
class TestSessionIsolation:
    def test_ask_uses_right_session_agent(self, monkeypatch):
        """不同 session_id 应命中不同的 Agent（互不影响）"""
        captured = {}

        class FakeAgent:
            def __init__(self, tag):
                self.tag = tag
            def invoke(self, payload):
                captured["agent"] = self.tag
                return {"messages": [FakeMessage("ok")]}

        monkeypatch.setattr(
            qa_agent, "AGENTS",
            {"sess-a": FakeAgent("A"), "sess-b": FakeAgent("B")},
        )
        qa_agent.ask("q", [], session_id="sess-a")
        assert captured["agent"] == "A"
        qa_agent.ask("q", [], session_id="sess-b")
        assert captured["agent"] == "B"

    def test_unknown_session_lazy_initialized(self, monkeypatch):
        """不存在的会话应被 init() 懒加载，不崩"""
        monkeypatch.setattr(qa_agent, "AGENTS", {})
        qa_agent.ask("q", [], session_id="brand-new")
        assert "brand-new" in qa_agent.AGENTS


''' ===== 1.6 长对话摘要压缩测试（改进2） ===== '''
class TestSummarize:
    def test_below_window_kept_all(self):
        """历史 ≤ MAX_HISTORY 时全量保留，不压缩"""
        messages = [{"role": "user", "content": f"q{i}"} for i in range(5)]
        assert qa_agent._summarize_history(messages) == messages

    def test_mid_range_truncated_to_window(self):
        """MAX_HISTORY~SUMMARY_STEP 之间纯滑动窗口截断（不调 LLM）"""
        messages = [{"role": "user", "content": f"q{i}"} for i in range(15)]
        result = qa_agent._summarize_history(messages)
        assert len(result) == qa_agent.MAX_HISTORY
        assert result[0] == {"role": "user", "content": "q5"}  # 只留最后 10 条

    def test_beyond_step_compresses_to_summary(self, monkeypatch):
        """超过 SUMMARY_STEP 条时压成摘要（模拟 LLM 返回摘要）"""
        messages = [{"role": "user", "content": f"q{i}"} for i in range(25)]

        class FakeLLM:
            def invoke(self, prompt):
                return FakeMessage("用户叫小明，聊了刹车片")

        monkeypatch.setattr(qa_agent, "llm", FakeLLM())
        result = qa_agent._summarize_history(messages)
        # 结果 = 1 条摘要 + 最近 10 条 = 11 条
        assert len(result) == qa_agent.MAX_HISTORY + 1
        assert result[0]["role"] == "system" and "小明" in result[0]["content"]


''' ===== 2. 空库状态测试 ===== '''
class TestEmptyKB:
    def test_retrieve_returns_hint_when_no_kb(self, monkeypatch):
        """空库时 retrieve 工具应返回引导提示，而不是崩溃"""
        session = "test-empty"
        monkeypatch.setattr(qa_agent, "CURRENT_SESSION", session)
        monkeypatch.setattr(qa_agent, "ENSEMBLES", {})  # 该会话没有知识库
        result = qa_agent.retrieve.invoke("刹车片多久换一次")
        assert "没有任何知识库" in result  # 返回的是空库提示

    def test_retrieve_fails_gracefully(self, monkeypatch):
        """检索抛异常时应返回'检索失败'，不崩溃"""
        class BrokenEns:
            def invoke(self, q):
                raise Exception("boom")
        monkeypatch.setattr(qa_agent, "CURRENT_SESSION", "test-broken")
        monkeypatch.setattr(qa_agent, "ENSEMBLES", {"test-broken": BrokenEns()})
        result = qa_agent.retrieve.invoke("问题")
        assert "检索失败" in result


''' ===== 2.5 引用溯源测试（改进3） ===== '''
class TestCitation:
    def test_retrieve_includes_source(self, monkeypatch):
        """检索结果应带上【来源：文件名】，供回答引用溯源"""
        class FakeDoc:
            def __init__(self, content, src):
                self.page_content = content
                self.metadata = {"source": src}
        class FakeEns:
            def invoke(self, q):
                return [FakeDoc("刹车片 30,000~50,000 公里", "汽配知识介绍.txt")]

        monkeypatch.setattr(qa_agent, "CURRENT_SESSION", "test-cite")
        monkeypatch.setattr(qa_agent, "ENSEMBLES", {"test-cite": FakeEns()})
        result = qa_agent.retrieve.invoke("刹车片多久换")
        assert "汽配知识介绍.txt" in result  # 带出来源文件名


''' ===== 3. 配置路径测试 ===== '''
class TestConfig:
    def test_upload_dir_defined(self):
        """config 应有上传目录配置，且是绝对路径"""
        assert config.UPLOAD_DIR
        assert os.path.isabs(config.UPLOAD_DIR)

    def test_persist_dir_absolute(self):
        """向量库目录必须是绝对路径（BASE_DIR 定位，与运行目录无关）"""
        assert os.path.isabs(config.PERSIST_DIR)
        assert config.PERSIST_DIR.endswith("chroma_db")
