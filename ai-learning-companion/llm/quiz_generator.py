import os
import re
from typing import Dict, List


def _fallback_quiz(text: str, num_mcq: int = 5, num_short: int = 5) -> Dict[str, List[Dict[str, str]]]:
    sentences = [s.strip() for s in re.split(r"[.!?]", text or "") if s.strip()]
    mcq = []
    for s in sentences[:num_mcq]:
        topic = " ".join(s.split()[:3])
        mcq.append({
            "question": f"Which statement best describes: {topic}?",
            "options": [
                f"It introduces {topic}",
                f"It contradicts {topic}",
                f"It expands on {topic}",
                f"It is unrelated",
            ],
            "answer": f"It introduces {topic}",
        })
    short = []
    for s in sentences[:num_short]:
        short.append({
            "question": f"Briefly explain: {s[:60]}...",
            "answer": "Key idea as discussed in the lecture.",
        })
    if not mcq:
        mcq.append({
            "question": "Which option summarizes the main concept?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option A",
        })
    if not short:
        short.append({"question": "State a key takeaway.", "answer": "Review the main takeaway."})
    return {"mcq": mcq, "short": short}


def generate_quiz(text: str, num_mcq: int = 5, num_short: int = 5) -> Dict[str, List[Dict[str, str]]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    if not text or not text.strip():
        return _fallback_quiz("", num_mcq, num_short)

    if api_key:
        try:
            import requests
            prompt = (
                "Generate quiz questions from the lecture content. Return two sections: "
                f"(1) {num_mcq} MCQs with 4 options and an answers key, (2) {num_short} short-answer questions. "
                "Use JSON-like structure.\n\nContent:\n" + text
            )
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You generate quizzes."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            }
            resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Very simple parse: try to extract JSON blocks if present
            import json
            try:
                # Attempt to parse as JSON directly
                parsed = json.loads(content)
                mcq = parsed.get("mcq") or parsed.get("MCQ") or []
                short = parsed.get("short") or parsed.get("Short") or []
                return {"mcq": mcq, "short": short}
            except Exception:
                return _fallback_quiz(text, num_mcq, num_short)
        except Exception:
            return _fallback_quiz(text, num_mcq, num_short)
    else:
        return _fallback_quiz(text, num_mcq, num_short)