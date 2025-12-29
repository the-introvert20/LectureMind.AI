import os
import requests
from typing import List, Dict


def identify_concepts(text: str) -> List[str]:
    """Identify key concepts from the transcript using LLM or heuristic."""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
    model = os.environ.get("OPENAI_MODEL", "gemma3:1b")
    
    is_local = "localhost" in api_base or "127.0.0.1" in api_base

    if not text or not text.strip():
        return []

    if api_key or is_local:
        try:
            prompt = (
                "Identify the most important technical concepts, ideas, or topics from this lecture transcript. "
                "Return a comma-separated list of 5-10 concepts.\n\nTranscript:\n" + text
            )
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert at identifying key academic concepts from transcripts."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            }
            resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            concepts = [c.strip() for c in content.split(",") if c.strip()]
            return concepts
        except Exception as e:
            print(f"Concept identification AI failed: {e}")
            return _heuristic_concepts(text)
    else:
        return _heuristic_concepts(text)


def _heuristic_concepts(text: str) -> List[str]:
    """Fallback: extract capitalized phrases as concepts."""
    import re
    # Simple regex to find likely concepts (Capitalized words that aren't at start of sentence)
    words = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
    # Filter common words and get unique ones
    common = {"The", "This", "That", "When", "There", "Here", "What", "How"}
    unique = []
    for w in words:
        if w not in common and w not in unique:
            unique.append(w)
    return unique[:8] or ["Main Idea", "Lecture Topic"]


def filter_concepts(concepts: List[str]) -> List[str]:
    """Filter out noise or generic terms from identified concepts."""
    noise = {"thing", "something", "stuff", "lecture", "today", "talk", "everyone"}
    filtered = [c for c in concepts if c.lower() not in noise and len(c) > 2]
    return filtered
