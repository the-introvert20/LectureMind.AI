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
    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Split into sentences using a simple regex (handles . ! ?)
    # We look for punctuation followed by a space and a capital letter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    paragraphs = []
    current_para = []
    current_len = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds max_len, start a new paragraph
        if current_len + sentence_len > max_len and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = [sentence]
            current_len = sentence_len
        else:
            current_para.append(sentence)
            current_len += sentence_len + 1 # +1 for the space
            
    # Add the last paragraph
    if current_para:
        paragraphs.append(" ".join(current_para))
        
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