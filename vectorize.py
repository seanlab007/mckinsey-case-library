#!/usr/bin/env python3
"""
麦肯锡案例库向量化处理
- 读取 3 个 markdown 文件
- 智能分段（按 ## 标题 chunking）
- 用 Ollama all-minilm 生成嵌入
- 存入 ChromaDB
"""

import os
import re
import hashlib
import json
import chromadb
import requests
import subprocess
from pathlib import Path

# 禁用代理，避免沙箱代理超时
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

# ====== 配置 ======
BASE_DIR = Path(__file__).parent
FILES = ["README.md", "catalog-500.md", "methodology.md"]
CHROMA_DIR = BASE_DIR / "vector_db"
COLLECTION_NAME = "mckinsey_case_library"
OLLAMA_API = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "all-minilm:latest"

# ====== 1. 文档分割 ======
def chunk_text(text: str, source: str, max_chars=400, overlap=80) -> list[dict]:
    """按 ## 标题分段，过大块进一步切分"""
    # 先按 ## / ### 标题分割
    sections = re.split(r'\n(?=## )', text)
    chunks = []
    
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        
        # 提取标题
        title_match = re.match(r'^## (.+)', sec)
        title = title_match.group(1).strip() if title_match else "前言"
        
        # 如果块太大，按段落继续分
        if len(sec) > max_chars:
            paragraphs = sec.split('\n\n')
            para_chunks = [""]
            for p in paragraphs:
                if len(para_chunks[-1]) + len(p) < max_chars:
                    para_chunks[-1] += '\n\n' + p
                else:
                    para_chunks.append(p)
            para_chunks[0] = para_chunks[0].strip()
            
            for i, pc in enumerate(para_chunks):
                pc = pc.strip()
                if not pc:
                    continue
                chunk_id = hashlib.md5(f"{source}:{title}:{i}".encode()).hexdigest()[:12]
                chunks.append({
                    "id": chunk_id,
                    "content": pc,
                    "metadata": {
                        "source": source,
                        "title": title,
                        "char_count": len(pc)
                    }
                })
        else:
            chunk_id = hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]
            chunks.append({
                "id": chunk_id,
                "content": sec,
                "metadata": {
                    "source": source,
                    "title": title,
                    "char_count": len(sec)
                }
            })
    
    return chunks


# ====== 2. Ollama 嵌入生成 ======
def get_embedding(text: str) -> list[float]:
    """调用 Ollama API 生成嵌入向量（绕过沙箱代理）"""
    resp = requests.post(OLLAMA_API, json={
        "model": EMBED_MODEL,
        "prompt": text
    }, timeout=60, proxies={"http": None, "https": None})
    resp.raise_for_status()
    return resp.json()["embedding"]


# ====== 3. 主流程 ======
def main():
    print("=" * 60)
    print("麦肯锡案例库 — 向量化处理")
    print("=" * 60)
    
    # 检查 Ollama
    print(f"\n🔍 检查 Ollama 服务...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5, proxies={"http": None, "https": None})
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"   ✅ Ollama 在线，模型: {', '.join(models[:5])}")
    except Exception:
        print("   ❌ Ollama 未运行！请先启动: ollama serve")
        return
    
    # 读取文件
    print(f"\n📄 读取文档...")
    all_chunks = []
    stats = {}
    
    for fname in FILES:
        fpath = BASE_DIR / fname
        if not fpath.exists():
            print(f"   ⚠️  未找到: {fname}")
            continue
        
        content = fpath.read_text(encoding='utf-8')
        chunks = chunk_text(content, fname)
        all_chunks.extend(chunks)
        stats[fname] = {"total_chars": len(content), "chunks": len(chunks)}
        print(f"   📄 {fname}: {len(content)} 字符 → {len(chunks)} 个块")
    
    print(f"\n📊 总计: {len(all_chunks)} 个文本块")
    
    # 初始化 ChromaDB
    print(f"\n🗄️  初始化 ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # 删除旧集合（如果存在）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   🗑️  已删除旧集合: {COLLECTION_NAME}")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "麦肯锡案例库与方法论全集 - 向量化知识库"}
    )
    print(f"   ✅ 已创建集合: {COLLECTION_NAME}")
    
    # 批量嵌入
    print(f"\n🧠 生成嵌入向量（使用模型: {EMBED_MODEL}）...")
    
    batch_size = 20
    total_embedded = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        ids = [c["id"] for c in batch]
        documents = [c["content"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        # 生成嵌入
        embeddings = []
        for doc in documents:
            try:
                emb = get_embedding(doc)
                embeddings.append(emb)
                total_embedded += 1
            except Exception as e:
                print(f"   ⚠️  嵌入失败: {e}")
                continue
        
        # 存入 ChromaDB
        if embeddings:
            collection.add(
                ids=ids[:len(embeddings)],
                documents=documents[:len(embeddings)],
                metadatas=metadatas[:len(embeddings)],
                embeddings=embeddings
            )
        
        progress = min(i + batch_size, len(all_chunks))
        print(f"   进度: {progress}/{len(all_chunks)} ({total_embedded} 已嵌入)")
    
    # 验证
    print(f"\n✅ 向量化完成！")
    print(f"   集合名称: {COLLECTION_NAME}")
    print(f"   文档数量: {collection.count()}")
    print(f"   存储路径: {CHROMA_DIR}")
    
    # 保存统计信息
    stats_file = BASE_DIR / "vectorization_stats.json"
    stats["total_chunks"] = len(all_chunks)
    stats["total_embedded"] = total_embedded
    stats["embed_model"] = EMBED_MODEL
    stats["collection"] = COLLECTION_NAME
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"   统计文件: {stats_file}")
    
    # 测试检索
    print(f"\n🔍 测试检索: '麦肯锡七步问题解决法'")
    query_emb = get_embedding("麦肯锡七步问题解决法")
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        preview = doc[:100].replace('\n', ' ')
        print(f"   #{i+1} [{meta['source']}:{meta['title']}] dist={dist:.4f}")
        print(f"       {preview}...")


if __name__ == "__main__":
    main()
