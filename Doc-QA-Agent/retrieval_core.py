# retrieval_core.py 核心层：项目的心脏，把"Embedding + 向量库 + BM25 + 融合 + 重排"全部封装成函数

''' 第 1 段：Embedding 类 计算向量 '''
import config  # 从自己的配置模块读值，而不是硬编码
from openai import OpenAI  
from langchain_core.embeddings import Embeddings

client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL
)

class SiliconflowEmbedding(Embeddings):
    def embed_documents(self, texts):
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts
        )
        return [r.embedding for r in resp.data]
    def embed_query(self, text):
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text
        )
        return resp.data[0].embedding

''' 第 2 段：建库函数（读取 -> 切块 -> 存入向量库）'''
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分块器

def build_vectorstore(doc_paths, persist_dir=config.PERSIST_DIR, collection_name=config.COLLECTION_NAME):
    # 读取文件
    from langchain_core.documents import Document
    from langchain_community.document_loaders import TextLoader  # 文件加载器，自动识别格式（TXT / MD / PDF）(只具备读取功能)
    docs_all = []
    for p in doc_paths:
        try:
            loader = TextLoader(p, encoding="utf-8") # TXT 要设 encoding="utf-8"，PDF 不用

            # 启动加载器，读取文件，并将内容与来源（源自哪个文件）打包在一起
            # 返回一个列表 [Document]，里面每个 Document 对象有两个属性：
            # page_content —— 文本内容
            # metadata —— 文件来源信息（比如文件名）
            docs_all = loader.load()
        except EnvironmentError as e:
            print(f"读取文件失败：{e}")


    # 切块
    splitter = RecursiveCharacterTextSplitter( # 创建分块器实例
        chunk_size=300,  # 300 字符/块
        chunk_overlap=40,  # 重叠 40 防切断语义
        separators=["───", "◆", "\n\n", "\n", " ", ""]  # 自定义分隔符
    )

    # split_text() → 接收字符串，返回字符串列表
    # split_documents() → 接收 Document 列表，返回 Document 列表（自动保留每个块的 metadata，比如来源文件名）
    chunks = splitter.split_documents(docs_all)


    # 存入 Chroma (建向量库)
    vectorstore = Chroma.from_documents( # 自动遍历下面的 chunks，调用 embedding 模型算出向量，然后存入 Chroma 数据库
        documents=chunks,
        embedding=SiliconflowEmbedding(),
        persist_directory=persist_dir,  # 指定数据的位置
        collection_name=collection_name # 起个自己的名字，避免以后误撞默认库
    )

    return vectorstore  # 把建好的库还给调用者

''' 第 3 端：问答层读取向量库函数 '''
def load_vectorstore(persist_dir=config.PERSIST_DIR, embedding_fct=SiliconflowEmbedding(), collection_na=config.COLLECTION_NAME):
    # 加载已有的向量库
    load_vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fct,
        collection_name=collection_na
        )
    return load_vs

'''   测试代码
if __name__ == "__main__": 
    vs = build_vectorstore(["./data/汽配知识介绍.txt"])
    print("知识库文档块数量：", vs._collection.count())
'''