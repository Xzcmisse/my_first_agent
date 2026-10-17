import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

NOTES_DIR = "/home/missxie/project/my_first_agent/notes"
MODEL_PATH = ("/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
              "/snapshots/7999e1d3359715c523056ef9478215996d62a620")

## -- 1.读文档+切块 --
chucks =[]
for path in sorted(glob.glob(NOTES_DIR + "/*.md")):
    section = ""
    for line in open(path, encoding="utf-8").read().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):                   # 遇到标题：记下来，不单独成块
            section = s.lstrip("# ").strip()    # lstrip 去掉开头的 # 和空格
            continue
        if len(s) < 12:                         # 太短的行（空行、符号）跳过
            continue
        text = (section + " | " + s) if section else s
        chucks.append({"text": text, "source": path.split("/")[-1]})
print("文档切块数",len(chucks))

## -- 2.向量化 --
model = SentenceTransformer(MODEL_PATH)
vectors = model.encode([c["text"] for c in chucks],normalize_embeddings=True)
vectors = np.asarray(vectors,dtype="float32")
print("向量形状",vectors.shape)

## -- 3.建faiss索引 --
index = faiss.IndexFlatIP(vectors.shape[1])  # 内积索引 
index.add(vectors)  # 添加向量到索引
print("索引向量数",index.ntotal)

## -- 4.查询 --
def search(query,k=3):
    q = model.encode([query],normalize_embeddings=True)
    q = np.asarray(q,dtype="float32")
    score,ids = index.search(q,k)  # 查询索引
    print("问题",query)
    for rank,(score,i) in enumerate(zip(score[0],ids[0])):
        print(f"top{rank} 相似度={score:.3f}  来源{chucks[i]['source']}")
        print("        ",chucks[i]["text"][:60].replace("\n"," ") + "...")

search("学习率太大会出什么现象？")
search("zero_grad()为什么必须要有？")
search("神经网络参数是怎么算的？")