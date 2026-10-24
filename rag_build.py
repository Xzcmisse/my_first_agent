import glob
import json
import numpy as np
import faiss    
from sentence_transformers import SentenceTransformer

CONFIG = {
    "max_len": 300,
    "overlap": 50,
    "min_len": 12,
    "with_title": True,
}


NOTES_DIR = "/home/missxie/project/my_first_agent/notes"
OUT_INDEX = "/home/missxie/project/my_first_agent/rag_index.faiss"
OUT_META = "/home/missxie/project/my_first_agent/rag_meta.json"
MODEL_PATH = ("/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
              "/snapshots/7999e1d3359715c523056ef9478215996d62a620")

def split_long(text, max_len, overlap):
    """超过 max_len 就切开，相邻块留 overlap 个字的重叠"""
    if len(text) <= max_len:
        return [text]
    parts, start = [], 0
    while start < len(text):
        parts.append(text[start:start + max_len])
        start += max_len - overlap
    return parts


def build_chunks():
    chunks = []
    for path in sorted(glob.glob(NOTES_DIR + "/*.md")):
        section = ""
        for line in open(path, encoding="utf-8").read().splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                section = s.lstrip("# ").strip()
                continue
            if len(s) < CONFIG["min_len"]:
                continue
            prefix = (section + " | ") if CONFIG["with_title"] else ""
            for part in split_long(prefix + s, CONFIG["max_len"], CONFIG["overlap"]):
                chunks.append({"text": part, "source": path.split("/")[-1]})
    return chunks


chunks = build_chunks()
print("块数:", len(chunks))
print("参数:", CONFIG)

model = SentenceTransformer(MODEL_PATH)
vectors = np.asarray(
    model.encode([c["text"] for c in chunks], normalize_embeddings=True),
    dtype="float32",
)

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, OUT_INDEX)          # 索引存盘

meta = {
    "config": CONFIG,
    "model": MODEL_PATH,
    "dim": int(vectors.shape[1]),
    "count": len(chunks),
    "chunks": chunks,
}
json.dump(meta, open(OUT_META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("已保存:", OUT_INDEX)
print("已保存:", OUT_META)