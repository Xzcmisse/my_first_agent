"""在线查询：加载索引 + 块 + 参数 -> 检索 top-k"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX = "/home/missxie/project/my_first_agent/rag_index.faiss"
META = "/home/missxie/project/my_first_agent/rag_meta.json"
MODEL_PATH = ("/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
              "/snapshots/7999e1d3359715c523056ef9478215996d62a620")

meta = json.load(open(META, encoding="utf-8"))
chunks = meta["chunks"]
print("加载:", meta["count"], "块")
print("参数:", meta["config"])

index = faiss.read_index(INDEX)
model = SentenceTransformer(MODEL_PATH)


def search(query, k=3):
    q = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
    scores, ids = index.search(q, k)
    print("\n问题:", query)
    for rank, (score, i) in enumerate(zip(scores[0], ids[0]), 1):
        text = chunks[i]["text"][:60].replace("\n", " ")
        print(f"  top{rank} {score:.3f} [{chunks[i]['source']}] {text}")


search("学习率太大会出现什么现象？")
search("zero_grad 为什么必须要有？")
search("神经网络参数是怎么算的？")