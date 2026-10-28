"""在线检索：加载索引和元数据 -> 返回 top-k 文本块"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent

INDEX = Path(os.getenv("RAG_INDEX", ROOT / "knowledge_index.faiss"))
META = Path(os.getenv("RAG_META", ROOT / "knowledge_meta.json"))
MODEL_PATH = (
    "/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
    "/snapshots/7999e1d3359715c523056ef9478215996d62a620"
)

if not INDEX.exists() or not META.exists():
    raise SystemExit(
        "找不到索引或元数据，请先运行: python rag_build.py\n"
        f"INDEX={INDEX}\nMETA={META}"
    )

meta = json.loads(META.read_text(encoding="utf-8"))
chunks = meta["chunks"]

print("索引:", INDEX.name)
print("加载:", meta["count"], "块")
print("参数:", meta["config"])

index = faiss.read_index(str(INDEX))
model = SentenceTransformer(MODEL_PATH)


def search(query, k=3):
    """返回最相关的 k 个块，每个块包含分数、正文、来源和章节。"""
    q = np.asarray(
        model.encode([query], normalize_embeddings=True),
        dtype="float32",
    )
    scores, ids = index.search(q, k)

    results = []
    for score, i in zip(scores[0], ids[0]):
        chunk = chunks[int(i)]
        results.append({
            "score": float(score),
            "text": chunk["text"],
            "source": chunk["source"],
            "section": chunk.get("section", ""),
        })
    return results


def print_results(query, k=3):
    print("\n问题:", query)
    for rank, hit in enumerate(search(query, k=k), 1):
        print(
            f"  top{rank} {hit['score']:.3f} "
            f"[{hit['source']} / {hit['section']}]"
        )
        print(f"    {hit['text']}")


if __name__ == "__main__":
    for question in (
        "RAG 的基本工作方式是什么？",
        "切块时 max_len 和 overlap 有什么作用？",
        "FAISS 是向量数据库吗？",
    ):
        print_results(question)
