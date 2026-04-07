---
date_start: 2026-04-07T14:48:00
tags:
  - AI
date_end: 2026-04-07T20:02:00
---
## rag流程



- 按照多行换行符 对文本进行切割存储至数组
```python
from typing import List  
  
def split_into_chunks(doc_file: str) -> List[str]:  
    with open(doc_file, 'r', encoding='utf-8') as file:  
        content = file.read()  
    return [chunk for chunk in content.split("\n\n")]  
  
chunks = split_into_chunks("doc.md")  
  
for i, chunk in enumerate(chunks):  
    print(f"{i} {chunk}\n")
```



- 下载中文文本的向量化模型 
- 文本转为`768`维的归一化向量 （消除模长影响
```python
from sentence_transformers import SentenceTransformer  
from typing import List  
import os  
  
# 指定缓存目录（直接让模型下载到项目目录）  
cache_dir = os.path.join(os.getcwd(), "models")  
os.makedirs(cache_dir, exist_ok=True)  
  
# 加载模型  
embedding_model = SentenceTransformer(  
    "shibing624/text2vec-base-chinese",  
    cache_folder=cache_dir  
)  
  
# 定义向量化函数  
def embed_chunk(chunk: str) -> List[float]:  
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)  
    return embedding.tolist()  
  
# 测试  
embedding = embed_chunk("测试内容")  
print(len(embedding))  
print(embedding)
```




- 用文本转向量函数处理之前切分好的中文文本
```python
embeddings = [embed_chunk(chunk) for chunk in chunks]  
  
print(len(embeddings))  
print(embeddings[0])
```




- 将`(序号,向量,文件)`存入本地数据库 
- 采用`tempfile`方便演示 运行后自动删除
```python
import chromadb  
from chromadb.config import Settings  
from typing import List  
import tempfile  
  
# 创建一个临时目录作为 
persist_directorytmp_dir = tempfile.mkdtemp()  
  
# 新版 ChromaDB 客户端  
chromadb_client = chromadb.PersistentClient(path=tmp_dir)  
  
chromadb_collection = chromadb_client.get_or_create_collection(name="default")  
  
def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:  
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):  
        chromadb_collection.add(  
            documents=[chunk],  
            embeddings=[embedding],  
            ids=[str(i)]  
        )  
  
# 示例调用  
save_embeddings(chunks, embeddings)
```



- 对问题进行向量化
- 使用向量化数据库ChromaDB自带的向量检索函数进行自动`点积运算`检索 
```python
def retrieve(query: str, top_k: int) -> List[str]:  
    # 对问题进行向量化  
    query_embedding = embed_chunk(query)  
    results = chromadb_collection.query(  
        query_embeddings=[query_embedding],  
        n_results=top_k  
    )  
    return results['documents'][0]  
  
# 问题  
query = "哆啦A梦使用的3个秘密道具分别是什么？"  
retrieved_chunks = retrieve(query, 5)  
  
for i, chunk in enumerate(retrieved_chunks):  
    print(f"[{i}] {chunk}\n")
```



- 重排 在之前的5个相关片段再次精选 
- 神经网络模型进行打分
```python
from sentence_transformers import CrossEncoder  
import os  
from typing import List  
  
# 读取模型缓存目录的模型文件  
cache_dir = os.path.join(os.getcwd(), "models")  
os.makedirs(cache_dir, exist_ok=True)  
  
# 加载模型  
cross_encoder = CrossEncoder(  
    'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',  
    cache_folder=cache_dir  
)  
  
def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:  
    # 问题与答案配对  
    pairs = [(query, chunk) for chunk in retrieved_chunks]  
  
    # 模型打分  
    scores = cross_encoder.predict(pairs)  
  
    # 配对 + 排序  
    scored_chunks = list(zip(retrieved_chunks, scores))  
    scored_chunks.sort(key=lambda x: x[1], reverse=True)  
  
    return [chunk for chunk, _ in scored_chunks][:top_k]  
  
reranked_chunks = rerank(query, retrieved_chunks, 3)  
  
for i, chunk in enumerate(reranked_chunks):  
    print(f"[{i}] {chunk}\n")
```



- 大模型结合材料回复用户问题
```python
import os  
from dotenv import load_dotenv  
from openai import OpenAI  
  
load_dotenv()  
  
# 初始化 DeepSeek client，直接用 OPENAI_API_KEYdeepseek_client = OpenAI(  
    api_key=os.environ.get("OPENAI_API_KEY"),  
    base_url="https://api.deepseek.com"  
)  
  
def generate_with_deepseek(query: str, chunks: list) -> str:  
    """  
    使用 DeepSeek 生成回答  
    """    # 将 chunks 拼接成一段文本  
    chunks_text = "\n\n".join(chunks)  
  
    # 系统角色定义模型身份和行为规则  
    system_prompt = (  
        "你是一位知识助手，请基于提供的相关片段回答问题，"  
        "不要编造信息，回答应准确、简明。"  
    )  
  
    # 用户角色只包含问题和相关上下文  
    user_prompt = f"用户问题: {query}\n\n相关片段:\n{chunks_text}"  
  
    # 调用 DeepSeek API    response = deepseek_client.chat.completions.create(  
        model="deepseek-chat",  
        messages=[  
            {"role": "system", "content": system_prompt},  
            {"role": "user", "content": user_prompt},  
        ],  
        stream=False  
    )  
  
    return response.choices[0].message.content  
  
# 使用已有的 query 和 reranked_chunksanswer = generate_with_deepseek(query, reranked_chunks)  
print(answer)
```