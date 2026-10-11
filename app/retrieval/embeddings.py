"""
An embedding model converts text into numerical vectors. We'll use the lightweight all-MiniLM-L6-v2 model as our initial baseline.
Our model produces 384-dimensional vectors. We'll store these in PostgreSQL with pgvector. Normalizing the vectors also makes cosine similarity a convenient retrieval metric.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the model once and reuse it.
    The first call downloads model files if necessary.
    """
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple text strings into embedding vectors.
    """
    if not texts:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype=np.float32).tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate the vector representation of one question.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    return embed_texts([query])[0]
