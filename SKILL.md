---
name: build-llm-rag-service
description: 从零搭一个带检索增强（RAG）的 LLM 问答服务：文档切分、向量化、检索、拼 prompt、FastAPI 接口、Docker 部署。当需要把一堆自己的资料做成能问答的服务时使用。
---

# 搭一个 LLM RAG 问答服务

目标：把一堆文档（md / pdf / txt）变成能问答的服务，接口对外暴露一个 `/ask`。

技术栈不固定，下面用最小可用组合：Python + FastAPI + Chroma + 大模型 HTTP 接口。

---

## 一、先定清楚三件事

动手前必须回答，不然后面全要返工：

1. **资料有多少** —— 几十份和几千份的切分策略完全不同。
2. **问题长什么样** —— 是"某个文件里写了啥"（关键词检索够用），还是"跨文档总结"（需要更好的切分和重排）。
3. **要不要引用来源** —— 要的话每个 chunk 必须带 `source` 和位置信息，切分时就带上。

---

## 二、目录结构

```
llm-rag/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── ingest.py        # 文档入库
│   ├── retrieve.py      # 检索
│   └── prompt.py        # prompt 拼装
├── data/                # 原始文档
├── chroma/              # 向量库落盘
├── requirements.txt
└── Dockerfile
```

---

## 三、切分（最容易做错的一步）

原则：**按语义边界切，不按固定字数切**。

- Markdown 按标题切，一个二级标题下的内容为一个 chunk，超过 1000 字再按段落拆。
- PDF 先转文本，保留页码，chunk 里带上 `page`。
- 每个 chunk 携带元数据：`source`（文件名）、`title`（所在标题）、`page`（可选）。

```python
def split_markdown(text: str, source: str) -> list[dict]:
    chunks, cur_title, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                chunks.append({"text": "\n".join(buf).strip(),
                               "source": source, "title": cur_title})
            cur_title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if buf:
        chunks.append({"text": "\n".join(buf).strip(),
                       "source": source, "title": cur_title})
    return [c for c in chunks if len(c["text"]) > 50]
```

坑：切太碎会导致检索命中但上下文不够；切太大导致检索不准。经验值 300–800 字/块。

---

## 四、向量化与入库

- 用任何一个 embedding 接口都行，关键是**入库和查询必须用同一个模型**。
- 一次批量提交（每批 16–64 条），比逐条快一个数量级。
- 向量库用 Chroma，落盘就是 `persist_directory`，不用起服务。

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma")
col = client.get_or_create_collection("docs")

def ingest(chunks: list[dict]):
    for i in range(0, len(chunks), 32):
        batch = chunks[i:i+32]
        col.add(
            ids=[f"{c['source']}-{i+j}" for j, c in enumerate(batch)],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "title": c["title"]} for c in batch],
            embeddings=embed([c["text"] for c in batch]),
        )
```

---

## 五、检索

- `top_k` 先给 5，别一上来给 20，上下文全是噪音。
- 检索结果按分数过滤，低于阈值直接丢弃，宁可答"资料里没有"。
- 有引用需求时，把 `source` + `title` 拼进 prompt，让模型在回答里标注。

```python
def retrieve(question: str, top_k: int = 5):
    res = col.query(query_embeddings=embed([question]),
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0],
                               res["metadatas"][0],
                               res["distances"][0]):
        if dist < 0.8:              # 阈值按你的 embedding 调
            hits.append({"text": doc, **meta})
    return hits
```

---

## 六、拼 prompt

结构固定成三段，别让模型自由发挥：

```
你是问答助手，只根据下面的资料回答。
资料里没有的，直接说"资料里没有提到"。

【资料】
[1] {title} — {source}
{text}

[2] ...

【问题】
{question}
```

坑：不写"只根据资料"会让模型开始编。

---

## 七、FastAPI 接口

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(req: Ask):
    hits = retrieve(req.question)
    if not hits:
        return {"answer": "资料里没有提到。", "sources": []}
    answer = call_llm(build_prompt(req.question, hits))
    return {"answer": answer,
            "sources": [{"source": h["source"], "title": h["title"]} for h in hits]}
```

必加的三样东西：

1. **超时** —— 调模型接口给 60s，别用默认值。
2. **重试** —— 网络抖动重试 2 次，指数退避。
3. **请求日志** —— 记下问题、命中的 chunk id、耗时。排查效果问题时全靠它。

---

## 八、部署

- 打成镜像，`data/` 和 `chroma/` 用 volume 挂进去，别烤进镜像。
- 前面挂 nginx 反代，只开 80/443。
- 服务本身只监听 `127.0.0.1`，别直接裸奔端口。
- 内存是最容易踩的：embedding 模型和向量库都吃内存，1G 的机器基本跑不动，2G 起步。

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
```

---

## 九、验收怎么判断做好了

- 问三个问题：一个资料里明确有的、一个需要跨两份文档的、一个资料里完全没有的。
- 第一个要答对并给出正确来源；第二个要答对；第三个必须说"没有提到"，不能编。
- 三个都对，才算能用。

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 检索命中但回答跑题 | chunk 太大，混了多个主题 | 按标题重新切 |
| 老是说"没有提到" | 阈值太严或 embedding 模型不匹配 | 调阈值 / 统一模型 |
| 回答里编造内容 | prompt 没约束 | 加"只根据资料"和兜底话术 |
| 首次响应很慢 | embedding 模型冷启动 | 启动时预热一次 |
| 内存爆掉 | 模型 + 向量库同时加载 | 换小模型 / 加内存 |
