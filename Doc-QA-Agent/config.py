# config.py 配置层：改 API key/模型只动这一个文件

import os
from pathlib import Path  # 用于把相对路径改造成"相对文件定位"
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("api_key")

BASE_URL = "https://api.siliconflow.cn/v1"

EMBEDDING_MODEL = "BAAI/bge-m3"

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# 用"文件自身位置"绝对定位，避免从不同目录运行时库存错位置
BASE_DIR = Path(__file__).resolve().parent  # 本文件所在目录，即 Doc-QA-Agent
PERSIST_DIR = str(BASE_DIR / "chroma_db")   # 向量库固定存在项目目录下的 chroma_db
UPLOAD_DIR = str(BASE_DIR / "uploads")      # 用户上传的文档暂存目录

COLLECTION_NAME = "qipeizhishi_kb"  # 该项目专属 collection