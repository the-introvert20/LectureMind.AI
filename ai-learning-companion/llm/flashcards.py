import os
import re
from typing import List, Dict


def _fallback_flashcards(text: str, num_cards: int = 10) -> List[Dict[str, str]]:
    # Extract simple noun phrases as topics and generate Q&A
    sentences = [s.strip() for s in re.split(r"[.!?]", text or "") if s.strip()]
    topics = []
    for s in sentences:
        # naive topic: longest word sequence capitalized or first 3 words
        words = s.split()
        if not words:
            continue
        cap_words = [w for w in words if w[0].isupper()]
        topic = " ".join(cap_words[:3]) if cap_words else " ".join(words[:3])
        topics.append(topic)

    cards = []
    for i, t in enumerate(topics[:num_cards]):
        q = f"What is {t}?"
        a = f"{t} relates to concepts discussed in the lecture."
        cards.append({"question": q, "answer": a})
    if not cards:
        cards.append({"question": "What are the main ideas?", "answer": "Review key points and summaries."})
    return cards


def generate_flashcards(text: str, num_cards: int = 10, concepts: list = None) -> List[Dict[str, str]]:
    """Generate flashcards using Cloud or Local AI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
    model = os.environ.get("OPENAI_MODEL", "gemma3:1b")
    
    is_local = "localhost" in api_base or "127.0.0.1" in api_base

    if not text or not text.strip():
        return [{"question": "No content provided.", "answer": "Upload and process a lecture first."}]

    concept_str = f" focusing on these identified concepts: {', '.join(concepts)}" if concepts else ""

    if api_key or is_local:
        try:
            import requests
            prompt = (
                f"Generate concise concept-based flashcards (Q&A) from the following lecture content{concept_str}. "
                f"Each card should test a specific concept or technical detail. Return {num_cards} cards.\n\nContent:\n" + text
            )
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You create helpful flashcards."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            }
            resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Try to parse simple Q/A lines
            cards = []
            for block in content.split("\n"):
                if ":" in block:
                    q, a = block.split(":", 1)
                    cards.append({"question": q.strip(), "answer": a.strip()})
            return cards or _fallback_flashcards(text, num_cards)
        except Exception as e:
            print(f"Flashcard AI failed: {e}")
            return _fallback_flashcards(text, num_cards)
    else:
        return _fallback_flashcards(text, num_cards)