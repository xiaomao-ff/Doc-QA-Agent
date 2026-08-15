# bm25_retriever.py：自实现的 BM25 关键词检索器（替代已停服的 langchain-community）
# 背景：langchain-community 已 sunset（2026-05），BM25Retriever 不再维护。
#       这里基于 rank_bm25 库自己实现一个检索器，接口与 langchain 一致：
#         from_texts(texts, preprocess_func) → .k → .invoke(query) → [Document...]
#       这样不依赖任何 langchain-community 包，也更可控。

from typing import Callable, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from rank_bm25 import BM25Okapi  # 经典 BM25 算法库


class BM25Retriever(BaseRetriever):
    """自实现的 BM25 检索器（不依赖 langchain-community）。

    用法（与 langchain 的 BM25Retriever 一致）：
        retriever = BM25Retriever.from_texts(chunks, preprocess_func=jieba_tokenize)
        retriever.k = 5                      # 返回 5 条
        docs = retriever.invoke("问题")       # 得到 Document 列表
    """

    vectorizer: BM25Okapi = Field(default=None, exclude=True)  # BM25 算法实例（核心）
    docs: List[Document] = Field(repr=False)                   # 原始文档列表（用来映射返回）
    k: int = 4                                                 # 返回条数
    preprocess_func: Callable[[str], List[str]] = lambda text: text.split()  # 分词函数

    # 允许 vectorizer 是非 pydantic 对象
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        preprocess_func: Callable[[str], List[str]] = lambda text: text.split(),
        **kwargs,
    ) -> "BM25Retriever":
        """从文本列表创建检索器。

        texts: 每段文档的文本（通常是切块后的 chunks）
        preprocess_func: 分词函数（中文用 jieba.cut）
        """
        texts_processed = [preprocess_func(t) for t in texts]
        metadatas = list(metadatas) if metadatas else [{} for _ in texts]
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(texts, metadatas)
        ]
        return cls(
            vectorizer=BM25Okapi(texts_processed),  # 用 rank_bm25 训练
            docs=docs,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """核心检索逻辑：查询词分词 → BM25 打分 → 取 top-k 文档"""
        processed_query = self.preprocess_func(query)          # 查询也要分词（中文关键！）
        top_indices = self.vectorizer.get_top_n(               # BM25 返回分数最高的文档下标
            processed_query, list(range(len(self.docs))), n=self.k
        )
        # 把下标映射回 Document 列表
        return [self.docs[i] for i in top_indices]
