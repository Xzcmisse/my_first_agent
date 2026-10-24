"""切分参数对比：3 组参数 -> 各自建索引 -> 同一套问题算命中率"""
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

NOTES_DIR = "/home/missxie/project/my_first_agent/notes"
MODEL_PATH = ("/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
              "/snapshots/7999e1d3359715c523056ef9478215996d62a620")

# ===== 三组切分参数（今天要选的就是这个）=====
PARAM_SETS = {
    "A_粗切":   {"max_len": 1000, "overlap": 0,  "min_len": 12, "with_title": True},
    "B_细切":   {"max_len": 300,  "overlap": 50, "min_len": 12, "with_title": True},
    "C_无标题": {"max_len": 300,  "overlap": 50, "min_len": 12, "with_title": False},
}

# ===== 评估集：注意都是「改写」的问法，不能和笔记原句一样 =====
EVAL = [
    ("调大 lr 以后模型会怎么样",             ["发散", "爆炸", "震荡"]),
    ("为什么每批训练前要把梯度清掉",         ["累加", "白纸", "清零"]),
    ("一个全连接层有多少个参数",             ["入×出", "偏置", "出×"]),
    ("过拟合长什么样",                       ["训练误差", "验证", "背答案"]),
    ("训练集和验证集干嘛要分开",             ["没见过", "考试", "验证集"]),
    ("矩阵乘法中间那一维有什么要求",         ["内维", "相等", "严丝合缝"]),
    ("把激活函数删了会怎样",                 ["退化", "一层", "线性"]),
    ("embedding 到底是什么",                 ["512", "向量", "数字"]),
]


def build_chunks(cfg):
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
            if len(s) < cfg["min_len"]:
                continue
            text = (section + " | " + s) if cfg["with_title"] else s
            if len(text) <= cfg["max_len"]:
                chunks.append(text)
            else:
                start = 0
                while start < len(text):
                    chunks.append(text[start:start + cfg["max_len"]])
                    start += cfg["max_len"] - cfg["overlap"]
    return chunks


def evaluate(model, chunks):
    vecs = np.asarray(model.encode(chunks, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    hit1 = hit3 = 0
    for query, keywords in EVAL:
        q = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
        scores, ids = index.search(q, 3)
        top1 = chunks[ids[0][0]]
        top3 = [chunks[i] for i in ids[0]]
        if any(k in top1 for k in keywords):
            hit1 += 1
        if any(any(k in c for k in keywords) for c in top3):
            hit3 += 1
    return hit1 / len(EVAL), hit3 / len(EVAL), len(chunks)


model = SentenceTransformer(MODEL_PATH)
print(f"{'参数组':<10}{'块数':<8}{'命中@1':<10}{'命中@3':<10}")
for name, cfg in PARAM_SETS.items():
    chunks = build_chunks(cfg)
    h1, h3, n = evaluate(model, chunks)
    print(f"{name:<10}{n:<8}{h1:<10.2f}{h3:<10.2f}")