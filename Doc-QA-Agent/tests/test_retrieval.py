# test_retrieval.py —— 多格式文档解析单元测试
# 测 load_documents() 按文件后缀选择正确加载器
# 运行：cd Doc-QA-Agent && python -m pytest tests/test_retrieval.py -v

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retrieval_core


''' ===== 多格式加载测试 ===== '''
class TestLoadDocuments:
    def test_txt_loaded(self, tmp_path):
        """TXT 文件应被正确解析出内容"""
        f = tmp_path / "a.txt"
        f.write_text("这是一段测试文本。", encoding="utf-8")
        docs = retrieval_core.load_documents([str(f)])
        assert len(docs) == 1
        assert "这是一段测试文本" in docs[0].page_content

    def test_md_loaded_as_text(self, tmp_path):
        """Markdown 走 TextLoader，应能读出正文"""
        f = tmp_path / "b.md"
        f.write_text("# 标题\n\n这是markdown正文。", encoding="utf-8")
        docs = retrieval_core.load_documents([str(f)])
        assert len(docs) >= 1
        assert "这是markdown正文" in docs[0].page_content

    def test_multiple_files_accumulated(self, tmp_path):
        """多文件应累加（不覆盖），不会丢前面的文件"""
        f1 = tmp_path / "one.txt"
        f2 = tmp_path / "two.txt"
        f1.write_text("第一个文件内容。", encoding="utf-8")
        f2.write_text("第二个文件内容。", encoding="utf-8")
        docs = retrieval_core.load_documents([str(f1), str(f2)])
        # 两个文件都被读进来
        texts = [d.page_content for d in docs]
        assert any("第一个文件" in t for t in texts)
        assert any("第二个文件" in t for t in texts)

    def test_nonexistent_file_no_crash(self, tmp_path):
        """不存在的文件应被 try/except 兜住，不抛异常"""
        docs = retrieval_core.load_documents([str(tmp_path / "nope.txt")])
        assert docs == []  # 空列表，不崩溃
