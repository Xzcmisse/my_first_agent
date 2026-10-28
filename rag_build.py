"""离线建库：读取 Markdown -> 切块 -> 向量化 -> 保存 FAISS 索引和元数据"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    "/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
    "/snapshots/7999e1d3359715c523056ef9478215996d62a620"
)


def split_long(text, max_len, overlap):
    """超过 max_len 就切开，相邻块保留 overlap 个字符。"""
    if len(text) <= max_len:
        return [text]

    parts = []
    start = 0
    step = max_len - overlap
    while start < len(text):
        parts.append(text[start:start + max_len])
        start += step
    return parts


def build_chunks(input_dir, config):
    """读取目录下所有 Markdown，并返回带来源和章节的文本块。"""
    chunks = []
    for path in sorted(input_dir.glob("*.md")):
        section = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                section = s.lstrip("# ").strip()
                continue

            if len(s) < config["min_len"]:
                continue

            prefix = ""
            if config["with_title"] and section:
                prefix = section + " | "

            text = prefix + s
            for part in split_long(
                text,
                config["max_len"],
                config["overlap"],
            ):
                chunks.append({
                    "text": part,
                    "source": path.name,
                    "section": section,
                })
    return chunks


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 索引")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "knowledge",
        help="Markdown 知识目录，默认 knowledge/",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "knowledge_index.faiss",
        help="FAISS 索引输出路径",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=ROOT / "knowledge_meta.json",
        help="元数据输出路径",
    )
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--min-len", type=int, default=12)
    parser.add_argument("--without-title", action="store_true")
    args = parser.parse_args()

    config = {
        "max_len": args.max_len,
        "overlap": args.overlap,
        "min_len": args.min_len,
        "with_title": not args.without_title,
    }

    chunks = build_chunks(args.input_dir, config)
    if not chunks:
        raise SystemExit(f"没有找到可用知识块: {args.input_dir}")

    print("知识目录:", args.input_dir)
    print("块数:", len(chunks))
    print("参数:", config)

    model = SentenceTransformer(MODEL_PATH)
    vectors = np.asarray(
        model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
        ),
        dtype="float32",
    )

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(args.index))

    meta = {
        "config": config,
        "model": MODEL_PATH,
        "dim": int(vectors.shape[1]),
        "count": len(chunks),
        "source_dir": str(args.input_dir),
        "chunks": chunks,
    }
    args.meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print("已保存:", args.index)
    print("已保存:", args.meta)


if __name__ == "__main__":
    main()
