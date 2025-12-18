import re


FILLER_WORDS = [
    r"\bum\b",
    r"\bumm\b",
    r"\bum-hmm\b",
    r"\bum-huh\b",
    r"\bumm-hmm\b",
    r"\bum-hm\b",
    r"\buh\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bkinda\b",
    r"\bsorta\b",
    r"\bI mean\b",
]


def _remove_fillers(text: str) -> str:
    cleaned = text
    for fw in FILLER_WORDS:
        cleaned = re.sub(fw, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _fix_punctuation(text: str) -> str:
    # Normalize spaces and punctuation, add periods to long lines
    text = re.sub(r"\s+", " ", text)
    # Add period if line ends without punctuation
    text = re.sub(r"([^.!?])\n", r"\1.\n", text)
    # Ensure spacing after punctuation
    text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)
    return text


def _split_into_paragraphs(text: str, max_len: int = 800) -> str:
    # Split by existing newlines first
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        lines = [text.strip()]

    paragraphs = []
    for line in lines:
        start = 0
        while start < len(line):
            end = min(start + max_len, len(line))
            chunk = line[start:end]
            paragraphs.append(chunk)
            start = end
    return "\n\n".join(paragraphs)


def clean_transcript(raw_text: str) -> str:
    """Clean filler words, fix punctuation, and split into paragraphs."""
    if not raw_text or not raw_text.strip():
        return ""
    text = raw_text.strip()
    text = _remove_fillers(text)
    text = _fix_punctuation(text)
    text = _split_into_paragraphs(text)
    return text