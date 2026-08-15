# test_qa_agent.py —— 问答层核心逻辑单元测试
# 只测"纯逻辑"：滑动窗口、空库状态、工具返回值，不调真实 LLM（快、稳、省）
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

        class FakeAgent:
            def invoke(self, payload):
                captured["messages"] = payload["messages"]
                return {"messages": [FakeMessage("ok")]}

        monkeypatch.setattr(qa_agent, "agent", FakeAgent())

        # 造 20 条历史（超过 MAX_HISTORY=10）
        long_history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        qa_agent.ask("新问题", long_history)

        # 最终发给模型的 = 最近10条历史 + 当前问题 = 11条
        assert len(captured["messages"]) == 11
        # 最后一条是当前问题
        assert captured["messages"][-1] == {"role": "user", "content": "新问题"}
        # 历史部分是最新的10条（q10~q19）
        assert captured["messages"][0] == {"role": "user", "content": "q10"}

    def test_short_history_not_truncated(self, monkeypatch):
        """历史不足 MAX_HISTORY 时原样保留"""
        captured = {}

        class FakeAgent:
            def invoke(self, payload):
                captured["messages"] = payload["messages"]
                return {"messages": [FakeMessage("ok")]}

        monkeypatch.setattr(qa_agent, "agent", FakeAgent())

        short_history = [{"role": "user", "content": "q1"}]
        qa_agent.ask("新问题", short_history)
        assert len(captured["messages"]) == 2


''' ===== 2. 空库状态测试 ===== '''
class TestEmptyKB:
    def test_retrieve_returns_hint_when_no_kb(self, monkeypatch):
        """空库时 retrieve 工具应返回引导提示，而不是崩溃"""
        monkeypatch.setattr(qa_agent, "ensemble", None)  # 模拟空库
        result = qa_agent.retrieve.invoke("刹车片多久换一次")
        assert "没有任何知识库" in result  # 返回的是空库提示

    def test_retrieve_fails_gracefully(self, monkeypatch):
        """检索抛异常时应返回'检索失败'，不崩溃"""
        class BrokenEns:
            def invoke(self, q):
                raise Exception("boom")
        monkeypatch.setattr(qa_agent, "ensemble", BrokenEns())
        result = qa_agent.retrieve.invoke("问题")
        assert "检索失败" in result


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
