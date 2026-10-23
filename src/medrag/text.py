"""Small text helpers shared by ingestion, retrieval and logging."""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def norm_key(text: str) -> str:
    """Lowercase alphanumerics only, for fuzzy matching of questions."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def redact(text: str) -> str:
    """Remove obvious personal identifiers before a query is written to logs."""
    text = _EMAIL.sub("[email]", text or "")
    return _PHONE.sub("[phone]", text)


def chunk_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, max_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + max_words]
        chunks.append(" ".join(piece))
        if start + max_words >= len(words):
            break
    return chunks


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


_BULLET = re.compile(r"\s+-\s+(?=[A-Z])")


def sentences(text: str) -> list[str]:
    """Split prose into sentences and NIH-style ' - ' bullet lists into items."""
    out = []
    for part in _SENT.split(text or ""):
        out += [p.strip() for p in _BULLET.split(part) if p.strip()]
    return out
