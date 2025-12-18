import os


def _fallback_notes_md(transcript: str) -> str:
    # Simple structure: headings by heuristic keywords and bullet points by sentences
    lines = [l.strip() for l in (transcript or "").split("\n") if l.strip()]
    if not lines:
        return "# Notes\n\nNo transcript provided."

    md = ["# Lecture Notes", "", "## Key Concepts", ""]
    # Take first 8 sentences as bullets
    sentences = []
    for l in lines:
        sentences.extend([s.strip() for s in l.split('.') if s.strip()])
    for s in sentences[:8]:
        md.append(f"- {s}.")

    md.extend(["", "## Details", ""])
    for s in sentences[8:16]:
        md.append(f"- {s}.")

    md.extend(["", "## Summary", "- This lecture covers the above key points."])
    return "\n".join(md)


def generate_notes(transcript: str) -> str:
    """Generate structured notes (Markdown). Uses OpenAI-compatible API if available, else heuristic fallback."""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    if not transcript or not transcript.strip():
        return "# Notes\n\nTranscript empty."

    if api_key:
        try:
            import requests
            prompt = (
                "Create well-structured study notes from this lecture transcript. "
                "Use headings and bullet points (Markdown).\n\nTranscript:\n" + transcript
            )
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You generate concise, structured study notes."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return _fallback_notes_md(transcript)
    else:
        return _fallback_notes_md(transcript)