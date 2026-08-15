# api.py：FastAPI 接口层，把问答层的 ask() / rebuild() 露成 HTTP 接口

import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文不乱码

# FastAPI 是一个用于构建 API（应用程序接口）的现代 Python Web 框架，基于 ASGI（异步服务器网关接口），支持自动生成交互式文档（Swagger UI）和类型校验
from fastapi import FastAPI, File, UploadFile
# Pydantic 是一个数据校验库，通过定义继承自 BaseModel 的类来声明请求和响应的数据结构，并自动进行类型验证和序列化
from pydantic import BaseModel
# HTTPException：主动抛出带状态码的错误响应
from fastapi import HTTPException

import os
import config
from qa_agent import ask, rebuild, init

# 创建一个 FastAPI 应用实例。title 参数为 API 指定一个名称，该名称会显示在自动生成的文档（如 /docs 路径）中。后续通过 app 对象来注册路由（端点)
app = FastAPI(title="文档问答 Agentic RAG")

# 定义一个 Pydantic 模型类，用于描述客户端发送 POST 请求时的请求体（JSON 格式）应该包含哪些字段及其类型。它作为"模板"，FastAPI 会自动根据这个模型校验传入的 JSON 数据
class AskRequest(BaseModel):  # 请求体格式模板
    question: str  # 用户问题
    history: list = []  # 对话历史（实现记忆）
    session_id: str = "default"  # 会话隔离：不同用户传不同 id

# 响应体模板：声明"服务端回啥"
class AskResponse(BaseModel):
    answer: str                                     # 回答
    history: list                                   # 带起新历史，方便前端存

# 路由装饰器，声明一个 POST 请求的路由。@app.post 表示该函数处理的是 HTTP POST 方法，路径为 /ask。也就是说，当客户端向 http://服务器地址/ask 发送 POST 请求时，会触发下面的函数
@app.post("/ask")
# 定义处理函数，接收一个参数 req，其类型是前面定义的 AskRequest。FastAPI 会自动将请求体 JSON 解析并转换为 AskRequest 实例，并完成类型校验
def do_ask(req: AskRequest):
    answer = ask(req.question, req.history, session_id=req.session_id)   # 复用问答层的 ask()（按会话隔离）
    # 记下这一轮，下次循环才有上下文
    new_history = req.history + [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": answer}
    ]
    return AskResponse(answer=answer, history=new_history)


# ------------------ 上传文档接口（含后端校验，改进4） ------------------
# 路由装饰器：路径为 /upload，接收 multipart/form-data 文件上传

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}  # 格式白名单
MAX_FILE_SIZE = 20 * 1024 * 1024                        # 单文件上限 20MB


@app.post("/upload")
# files：UploadFile 列表，用 File(...) 声明这是上传的文件（可多个）
def do_upload(
    files: list[UploadFile] = File(...),
    session_id: str = "default",
):
    if not files:
        raise HTTPException(status_code=400, detail="未收到文件")

    # 改进4：后端校验——格式白名单 + 文件大小限制（不再只靠前端限制）
    saved_paths = []
    for f in files:
        # 1) 格式白名单：只看真实后缀，防止改扩展名绕过
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式：{f.filename}（仅支持 PDF/Word/TXT/Markdown）",
            )
        # 2) 文件大小限制：防止超大文件拖垮服务
        content = f.file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件过大：{f.filename}（上限 20MB）")

        save_path = os.path.join(config.UPLOAD_DIR, f.filename)
        with open(save_path, "wb") as fp:  # wb = 写二进制
            fp.write(content)              # 写已校验的二进制内容到磁盘
        saved_paths.append(save_path)

    # 调用 rebuild：解析 → 入库 → 重建该会话的 Agent（按 session_id 隔离）
    rebuild(saved_paths, session_id=session_id)

    return {"status": "ok", "parsed": len(saved_paths), "session_id": session_id, "files": [f.filename for f in files]}


# 健康检查：面试演示"接口活着"
# 路由装饰器，添加一个根路径的 GET 接口。它告诉 FastAPI：当客户端向服务器根路径（即 http://域名或IP/，注意后面没有 /ask）发送 GET 请求时，执行下面的函数
@app.get("/")
def hello():
    return {"status": "ok", "service": "doc-qa-agentic-rag"}
