# eval_retrieval.py —— 检索质量评估脚本（Recall@k / MRR）
#
# 目的：回答"混合检索（BM25+向量+RRF）到底比单路好多少？chunk_size 怎么定？"
#      用一套"问题 → 标准答案片段"的测试集，分别跑三种检索策略打分。
#
# 用法：
#   cd Doc-QA-Agent
#   python eval_retrieval.py                  # 用默认 chunk_size=300
#   python eval_retrieval.py --chunk-size 200 # 试不同切块大小
#
# 指标说明：
#   Recall@5：top5 里是否出现"答案所在块"（每问 1~2 个标准块）
#   MRR     ：标准块在 top-k 中的位置倒数（越靠前越大，满分 1.0）
#
# 三种策略（与线上 qa_agent 一致）：
#   vector   纯向量检索（bge-m3）
#   bm25     纯关键词检索（自实现 BM25 + jieba 分词）
#   rrf      两者 RRF 融合（EnsembleRetriever）

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jieba
import retrieval_core
import config
from bm25_retriever import BM25Retriever


# ===== 测试集：每问带一个"标准答案片段"，该片段只出现在正确答案所在的块里 =====
EVAL_SET = [
    ("刹车片多久换一次？", "30,000~50,000"),
    ("刹车片摩擦材料厚度小于多少必须更换？", "3mm"),
    ("新刹车片需要磨合多少公里？", "200~300"),
    ("空调滤清器能过滤PM2.5颗粒吗？", "PM2.5"),
    ("火花塞铱金材质的寿命大概多久？", "60,000~80,000"),
    ("火花塞选错热值会有什么后果？", "爆震"),
    ("轮胎一般几年或多少公里需要更换？", "40,000~60,000"),
    ("轮胎磨损标记的高度是多少？", "1.6mm"),
    ("雨刮一般多久更换一次？", "6~12 个月"),
    ("LED灯的能耗是卤素灯的多少？", "1/5"),
    ("LED灯建议选择多少色温？", "5000K~6000K"),
    ("空气滤清器一般多少公里更换？", "10,000~15,000 公里"),
    ("氧传感器多少公里后性能会衰减？", "80,000~100,000"),
    ("前氧传感器故障主要影响什么？", "前氧故障影响油耗"),
    ("喇叭一般几年可能出现触点氧化？", "3~5 年"),
    ("刹车线出现什么情况需要更换？", "锈蚀"),
    ("传感器更换后通常需要做什么？", "清除故障码"),
    ("胎压建议多久检查一次？", "每月一次"),
    ("半金属刹车片有什么特点？", "耐磨"),
    ("手刹拉线正常应在几个齿内锁止？", "4~7"),
]


def make_chunks(doc_path, chunk_size, chunk_overlap=40):
    """按指定 chunk_size 切块，返回块文本列表"""
    docs = retrieval_core.load_documents([doc_path])
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["───", "◆", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def golden_chunks(chunks, fragment):
    """返回包含"标准答案片段"的块下标集合"""
    return {i for i, d in enumerate(chunks) if fragment in d.page_content}


def recall_mrr(retrieved_docs, gold_idx):
    """retrieved_docs: 检索返回的 Document 列表（带 metadata）
    gold_idx: 标准块下标集合
    返回 (recall@k, mrr)：任一标准块命中→recall=1；mrr=第一个命中位置的倒数
    """
    hits = []
    for rank, d in enumerate(retrieved_docs, start=1):
        idx = d.metadata.get("_idx")
        if idx in gold_idx:
            hits.append(rank)
    recall = 1.0 if hits else 0.0
    mrr = 1.0 / hits[0] if hits else 0.0
    return recall, mrr


def run_eval(chunk_size):
    doc = str(config.BASE_DIR / "data" / "汽配知识介绍.txt")
    print(f"\n{'='*60}\nchunk_size={chunk_size} 调参评估\n{'='*60}")

    # 1) 切块 + 给每块塞一个 _idx 元数据（用于定位"标准块"）
    #    注意：必须用同一批块同时建"向量库"和"BM25"，两边才能对上号。
    chunks = make_chunks(doc, chunk_size)
    for i, d in enumerate(chunks):
        d.metadata["_idx"] = i
    print(f"共 {len(chunks)} 个块")

    # 2) 构建三路检索器（k=5，与线上对齐）
    #    vector 路：用同一批块直接建临时向量库（带上 _idx 元数据）
    #    注意：Chroma.from_documents 对同名已存在 collection 是"追加"，
    #    必须先删掉上次评估的旧数据，否则新旧切块混在一起会污染结果。
    import chromadb
    client = chromadb.PersistentClient(path=config.PERSIST_DIR)
    try:
        client.delete_collection("eval_kb")
    except Exception:
        pass
    from langchain_chroma import Chroma
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=retrieval_core.SiliconflowEmbedding(),
        persist_directory=config.PERSIST_DIR,
        collection_name="eval_kb",
    )
    vector_retriever = vs.as_retriever(search_kwargs={"k": 5})

    #    bm25 路：自实现检索器（同一批块，带 _idx）
    bm25 = BM25Retriever.from_texts(
        [d.page_content for d in chunks],
        metadatas=[d.metadata for d in chunks],
        preprocess_func=lambda t: list(jieba.cut(t)),
    )
    bm25.k = 5

    #    rrf 路：EnsembleRetriever 融合
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
    rrf = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[1, 1])

    # 3) 逐问打分
    stats = {"vector": [], "bm25": [], "rrf": []}
    retrievers = {"vector": vector_retriever, "bm25": bm25, "rrf": rrf}

    for q, fragment in EVAL_SET:
        gold = golden_chunks(chunks, fragment)
        if not gold:
            print(f"  ⚠️ 片段未匹配任何块（可能被切块切断，请调整测试集）：{fragment}")
        for name, retriever in retrievers.items():
            docs = retriever.invoke(q)
            r, m = recall_mrr(docs, gold)
            stats[name].append((r, m))

    # 4) 汇总
    def summary(name):
        rs = [x[0] for x in stats[name]]
        ms = [x[1] for x in stats[name]]
        return sum(rs) / len(rs), sum(ms) / len(ms)

    print(f"\n{'策略':<8}{'Recall@5':<10}{'MRR':<8}  (20 问)")
    print("-" * 34)
    result = {}
    for name in stats:
        rec, mrr = summary(name)
        result[name] = {"recall@5": round(rec, 3), "mrr": round(mrr, 3)}
        print(f"{name:<8}{rec:<10.3f}{mrr:<8.3f}")

    return {"chunk_size": chunk_size, "n_chunks": len(chunks), **result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=300)
    args = ap.parse_args()

    res = run_eval(args.chunk_size)

    # 保存到结果文件（追加），方便跨 chunk_size 对比
    out = Path(__file__).resolve().parent / "eval_results.json"
    data = []
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data = [r for r in data if r["chunk_size"] != args.chunk_size]
    data.append(res)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out}")
