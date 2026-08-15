# test_integration.py —— 端到端集成测试（调真实 LLM / Embedding API）
# 默认跳过，手动运行：cd Doc-QA-Agent && python -m pytest tests/test_integration.py -v -m integration
# 需要 .env 里配置好 api_key

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import qa_agent
import config


def reset_empty_kb():
    """重置为空库（每个测试开头调用，保证隔离且复用已构建的 agent）"""
    qa_agent.ensemble = None  # 只置空知识库，不复建 agent（快）


''' ===== 空库行为（未上传文档） ===== '''
@pytest.mark.integration
class TestEmptyKBBehavior:
    def test_chitchat_answered(self):
        """闲聊不应调检索，直接回答"""
        reset_empty_kb()
        ans = qa_agent.ask("你好", [])
        assert isinstance(ans, str) and len(ans) > 0

    def test_knowledge_question_guides_upload(self):
        """知识类问题在空库时应引导上传文档"""
        reset_empty_kb()
        ans = qa_agent.ask("刹车片多久换一次？", [])
        assert ("上传" in ans) or ("学习" in ans)


''' ===== 上传文档后问答 ===== '''
@pytest.mark.integration
class TestAfterUpload:
    def test_ask_after_upload(self):
        """上传默认示例文档后，知识类问题应能检索到资料回答"""
        doc = str(config.BASE_DIR / "data" / "汽配知识介绍.txt")
        qa_agent.rebuild([doc])
        ans = qa_agent.ask("刹车片多久换一次？", [])
        assert isinstance(ans, str) and len(ans) > 10  # 有实质内容


''' ===== 多轮记忆 ===== '''
@pytest.mark.integration
class TestMemory:
    def test_multi_turn_remembers_context(self):
        """连续两轮：第二轮应记住第一轮的人名"""
        reset_empty_kb()
        h = []
        a1 = qa_agent.ask("我叫小明", h)
        h += [{"role": "user", "content": "我叫小明"}, {"role": "assistant", "content": a1}]
        a2 = qa_agent.ask("我叫什么名字？", h)
        assert "小明" in a2  # 从历史里记住了名字
