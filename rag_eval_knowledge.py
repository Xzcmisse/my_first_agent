"""在正式知识库上评测检索：20 题 -> 命中@1 / 命中@3 / MRR"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_build import build_chunks

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
QUESTIONS = ROOT / "eval" / "rag_questions.json"
MODEL_PATH = (
    "/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
    "/snapshots/7999e1d3359715c523056ef9478215996d62a620"
)

CONFIG = {
    "max_len": 300,
    "overlap": 50,
    "min_len": 12,
    "with_title": True,
}


def main():
    questions = json.loads(
        QUESTIONS.read_text(encoding="utf-8")
    )["questions"]
    chunks = build_chunks(KNOWLEDGE_DIR, CONFIG)
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

    query_vectors = np.asarray(
        model.encode(
            [q["question"] for q in questions],
            normalize_embeddings=True,
        ),
        dtype="float32",
    )
    scores, ids = index.search(query_vectors, 3)

    hit1 = 0
    hit3 = 0
    reciprocal_rank = 0.0

    print(f"知识块: {len(chunks)}")
    print(f"题目数: {len(questions)}")
    print()

    for q, row_scores, row_ids in zip(questions, scores, ids):
        expected = q["expected_sections"]
        found_rank = None
        for rank, (score, idx) in enumerate(zip(row_scores, row_ids), 1):
            chunk = chunks[int(idx)]
            if any(section in chunk["text"] for section in expected):
                found_rank = rank
                break

        if found_rank == 1:
            hit1 += 1
            reciprocal_rank += 1.0
            status = "PASS@1"
        elif found_rank and found_rank <= 3:
            hit3 += 1
            reciprocal_rank += 1.0 / found_rank
            status = f"PASS@{found_rank}"
        else:
            status = "MISS"

        top1 = chunks[int(row_ids[0])]
        print(
            f"{q['id']} {status:<7} "
            f"top1={top1['section']} | {q['question']}"
        )

    total = len(questions)
    hit3_total = hit1 + hit3
    print()
    print(f"命中@1: {hit1}/{total} = {hit1 / total:.2f}")
    print(f"命中@3: {hit3_total}/{total} = {hit3_total / total:.2f}")
    print(f"MRR:    {reciprocal_rank / total:.3f}")


if __name__ == "__main__":
    main()
