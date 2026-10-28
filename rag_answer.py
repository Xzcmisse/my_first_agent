"""RAG 问答：检索知识库 -> 拼上下文 -> 调用本地 API 模型生成答案"""
from openai import OpenAI

from rag_search import search

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL = "qwen2:1.5b"

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。
必须遵守以下规则：
1. 只根据用户提供的资料回答，不使用资料外的知识。
2. 先直接回答结论，再用资料中的事实解释。
3. 公式、定义和数字必须原样引用，不得自行推导或补充。
4. 每个关键结论后标注来源，格式为 [来源: 文件名 / 章节]。
5. 如果资料不足以回答，只回答“知识库中没有找到足够信息”，不要猜测。
回答尽量控制在 4 句话以内。"""


def answer(question, k=3):
    hits = search(question, k=k)

    for rank, hit in enumerate(hits, 1):
        print(
            f"[检索 {rank}] "
            f"score={hit['score']:.3f} "
            f"source={hit['source']} "
            f"section={hit['section']}"
        )
        print(hit["text"])

    context = "\n\n".join(
        f"[来源: {h['source']} / {h['section']}]\n{h['text']}"
        for h in hits
    )

    user_prompt = f"""资料：
{context}

用户问题：
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    questions = [
        "RAG 和微调的知识存放位置有什么不同？",
        "切块时为什么需要 overlap？",
        "FAISS 返回编号后，系统怎样找到原文？",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("问题：", question)
        print("回答：", answer(question))
