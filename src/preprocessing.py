"""
Text preprocessing and lightweight entity extraction utilities.
These are used both at training time (to normalize patterns) and at
inference time (to normalize user input and pull out useful slots
like order IDs).
"""

import re


def clean_text(text: str) -> str:
    """
    Normalizes raw user input:
    - Lowercases
    - Strips leading/trailing whitespace
    - Collapses multiple spaces into one
    - Removes characters that aren't alphanumeric, spaces, or basic punctuation

    Note: we deliberately keep this light. Sentence-transformer embeddings
    are trained on natural language, so aggressive cleaning (e.g. stemming,
    stopword removal) actually *hurts* embedding quality here.
    """
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s.,!?'#-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_order_id(text: str) -> str:
    """
    Extracts an order ID from free text using common patterns:
      - 'ORD12345', 'order 12345', '#12345'
    Returns a placeholder value if none is found, so response templates
    never break due to a missing slot.
    """
    patterns = [
        r"\bORD-?\d{4,10}\b",     # ORD12345 or ORD-12345
        r"#\d{4,10}\b",           # #12345
        r"\border\s*(?:id)?\s*[:#]?\s*(\d{4,10})\b",  # order id 12345
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            found = match.group(0).upper().replace(" ", "")
            return found

    return "on file"  # graceful fallback used in response templates


def extract_email(text: str) -> str:
    """Extracts an email address from text, if present."""
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else ""
