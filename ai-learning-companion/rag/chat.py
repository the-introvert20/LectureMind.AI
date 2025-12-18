import os
from typing import List

from .embeddings import EmbeddingModel
from .vector_store import LectureVectorStore


def _format_context(ctx_chunks: List[str]) -> str:
    ctx = "\n\n".join([f"- {c}" for c in ctx_chunks])
    return ctx


def _llm_answer_fallback(question: str, context: str) -> str:
    # Simple heuristic answer generator combining context and question
    answer = (
        "Based on the retrieved lecture context, here is a synthesized answer:\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Summary: The context suggests key points relevant to your question. "
        "Please review the bullet points above; they reflect the most relevant excerpts."
    )
    return answer


def answer_question(question: str, store: LectureVectorStore, top_k: int = 5) -> str:
    embedder = EmbeddingModel()
    q_emb = embedder.embed_query(question)
    results = store.search(q_emb, top_k=top_k)
    ctx_chunks = [t for t, _ in results]
    context = _format_context(ctx_chunks)

    # Optional OpenAI-compatible API usage if API key set
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    if api_key:
        try:
            import requests

            prompt = (
                "You are a helpful tutor. Answer the question using the provided context. "
                "Cite the most relevant points. If context is insufficient, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI tutor."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return _llm_answer_fallback(question, context)
    else:
        return _llm_answer_fallback(question, context)