"""
LLMs cannot efficiently process an entire collection of long documents for every question. We split the text into smaller, overlapping chunks.
Our initial chunker uses a simple character-based strategy. Later, we can compare it with token-based or sentence-aware chunking using our evaluation dataset.
"""

from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Chunk sizes here are measured in characters, not tokens.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = (
        settings.chunk_overlap
        if chunk_overlap is None
        else chunk_overlap
    )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)
    step = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def chunk_documents(
    documents: list[dict],
) -> list[dict]:
    """
    Chunk extracted documents while preserving source metadata.
    """
    results = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            results.append({
                "text": chunk,
                "source": document["source"],
                "page": document.get("page"),
                "chunk_index": index,
            })

    return results
