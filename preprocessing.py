import re
import unicodedata


def normalize_text(text: str) -> str:
    """Match the notebook preprocessing exactly."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return normalize_text(text).split()
