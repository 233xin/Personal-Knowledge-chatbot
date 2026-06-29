"""
Vector store module using ChromaDB with LangChain integration.
Uses a lightweight numpy-based embedding function — no native extensions,
no PyTorch, no ONNX runtime needed.

For production, swap NumpyEmbeddings for:
  - langchain_huggingface.HuggingFaceEmbeddings
  - Or any LangChain-compatible Embeddings implementation
"""

import logging
import re
from pathlib import Path
from typing import List

import numpy as np
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Persistence directory for ChromaDB
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Embedding dimension
EMBEDDING_DIM = 384

# ChromaDB collection name
COLLECTION_NAME = "knowledge_base"


class NumpyEmbeddings(Embeddings):
    """
    Lightweight embedding function using word/char n-gram hashing with
    random projection. Implements LangChain's Embeddings interface.

    Quality is comparable to TF-IDF + SVD — sufficient for personal use.
    Swap in a transformer model for production.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self._random_matrix: np.ndarray | None = None

    def _get_random_matrix(self, n_cols: int) -> np.ndarray:
        if self._random_matrix is None or self._random_matrix.shape[1] < n_cols:
            rng = np.random.RandomState(42)
            self._random_matrix = rng.randn(self.dim, max(n_cols, 8192)).astype(np.float32)
        return self._random_matrix[:, :n_cols]

    def _text_to_sparse_vec(self, text: str, vocab_size: int = 8192) -> np.ndarray:
        text_stripped = text.strip()
        if not text_stripped:
            return np.zeros(vocab_size, dtype=np.float32)

        vec = np.zeros(vocab_size, dtype=np.float32)

        tokens = re.findall(r'[a-zA-Z0-9一-鿿]+', text_stripped.lower())
        if not tokens:
            tokens = [text_stripped]

        token_weight = 3.0 / max(len(tokens), 1)
        for token in tokens:
            vec[hash(token) % vocab_size] += token_weight

        if len(tokens) >= 2:
            bigram_weight = 1.5 / max(len(tokens) - 1, 1)
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + "_" + tokens[i + 1]
                vec[hash(bigram) % vocab_size] += bigram_weight

        text_lower = text_stripped.lower()
        for n_size in (2, 3):
            for i in range(len(text_lower) - n_size + 1):
                ngram = text_lower[i:i + n_size]
                vec[hash(ngram) % vocab_size] += 0.3

        for char in text_stripped:
            cp = ord(char)
            if cp > 127:
                vec[cp % vocab_size] += 0.2

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec.astype(np.float32)

    def _embed_one(self, text: str) -> np.ndarray:
        hash_vec = self._text_to_sparse_vec(text)
        rand_matrix = self._get_random_matrix(len(hash_vec))
        embedding = rand_matrix @ hash_vec
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text).tolist()


_vector_store: Chroma | None = None


def get_vector_store() -> Chroma:
    """Get or create the ChromaDB vector store (singleton)."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing ChromaDB vector store.")
        embeddings = NumpyEmbeddings(dim=EMBEDDING_DIM)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        count = _vector_store._collection.count()
        logger.info(f"ChromaDB initialized. Collection has {count} documents.")

    return _vector_store
