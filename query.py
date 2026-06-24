#!/usr/bin/env python3
"""麦肯锡向量库检索测试"""

import os, json
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

import chromadb
import requests
from pathlib import Path

CHROMA_DIR = Path(__file__).parent / "vector_db"
COLLECTION = "mckinsey_case_library"
OLLAMA_API = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "all-minilm:latest"

def get_embedding(text: str) -> list[float]:
    resp = requests.post(OLLAMA_API, json={
        "model": EMBED_MODEL, "prompt": text
    }, timeout=60, proxies={"http": None, "https": None})
    if resp.status_code != 200:
        print(f"  ⚠️ Embedding error: {resp.status_code}")
        return None
    return resp.json()["embedding"]

def search(query: str, n=5):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    emb = get_embedding(query)
    if not emb:
        return
    results = collection.query(query_embeddings=[emb], n_results=n)
    
    print(f"\n🔍 查询: {query}")
    print("-" * 60)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        preview = doc[:120].replace('\n', ' ')
        print(f"#{i+1} [{meta['source']}::{meta['title']}] dist={dist:.4f}")
        print(f"    {preview}...")
        print()

# 测试多组查询
queries = [
    "麦肯锡如何解决商业问题",
    "金字塔原理的核心要点",
    "战略分析工具波特五力模型",
    "MECE原则是什么",
    "500套案例资料怎么获取",
]

for q in queries:
    search(q)
