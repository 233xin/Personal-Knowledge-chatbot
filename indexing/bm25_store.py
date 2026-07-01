"""
Persistent BM25 keyword index with Chinese tokenization (jieba).
Lives alongside ChromaDB as the second retrieval path.

Every chunk added to ChromaDB is also added here with the same
metadata, enabling keyword-based exact-match retrieval.
"""

import logging
import pickle
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import BM25_DIR, BM25_INDEX_FILE, BM25_K1, BM25_B

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 keyword search with jieba Chinese tokenization and pickle persistence."""

    def __init__(self):
        self.corpus: list[list[str]] = []       # Tokenized documents
        self.doc_ids: list[str] = []             # Chunk IDs (align with ChromaDB)
        self.metadatas: list[dict] = []          # Same metadata as ChromaDB
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = False

        # Lazy-import jieba
        self._jieba = None

    @property
    def jieba(self):
        if self._jieba is None:
            import jieba
            self._jieba = jieba
        return self._jieba

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Chinese-aware tokenization. jieba for CJK, whitespace for ASCII."""
        tokens = [t for t in self.jieba.cut(text) if t.strip()]
        return tokens if tokens else text.split()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add chunks to BM25 index. Rebuilds index after addition."""
        for doc in documents:
            tokens = self.tokenize(doc.page_content)
            self.corpus.append(tokens)
            self.metadatas.append(dict(doc.metadata))
            self.doc_ids.append(str(uuid.uuid4()))

        self._rebuild()
        self._dirty = True
        logger.info(
            f"[BM25] Added {len(documents)} documents. "
            f"Total corpus size: {len(self.corpus)}"
        )

    def _rebuild(self) -> None:
        """Rebuild BM25Okapi from current corpus."""
        if self.corpus:
            self._bm25 = BM25Okapi(self.corpus, k1=BM25_K1, b=BM25_B)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        BM25 keyword search.

        Args:
            query: Search query string.
            top_k: Number of results.
            doc_ids: Optional filter — only return chunks from these documents.

        Returns:
            List of dicts: {"content": str, "metadata": dict, "score": float}
        """
        if self._bm25 is None:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Score and filter
        scored = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            if doc_ids is not None:
                chunk_doc_id = self.metadatas[i].get("doc_id", "")
                if chunk_doc_id not in doc_ids:
                    continue
            scored.append((i, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "content": " ".join(self.corpus[i]),
                "metadata": self.metadatas[i],
                "score": score,
            }
            for i, score in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks for a document. Returns count removed."""
        keep = [
            i for i, m in enumerate(self.metadatas)
            if m.get("doc_id") != doc_id
        ]
        removed = len(self.corpus) - len(keep)

        if removed > 0:
            self.corpus = [self.corpus[i] for i in keep]
            self.doc_ids = [self.doc_ids[i] for i in keep]
            self.metadatas = [self.metadatas[i] for i in keep]
            self._rebuild()
            self._dirty = True
            logger.info(f"[BM25] Deleted {removed} chunks for doc_id={doc_id}")

        return removed

    def count(self) -> int:
        return len(self.corpus)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist BM25 state to disk as pickle."""
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        path = BM25_DIR / BM25_INDEX_FILE

        # We don't pickle the BM25Okapi object directly (version compat issues).
        # Instead, save the corpus + metadata and rebuild on load.
        data = {
            "corpus": self.corpus,
            "doc_ids": self.doc_ids,
            "metadatas": self.metadatas,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        self._dirty = False
        logger.info(f"[BM25] Saved to {path} ({len(self.corpus)} docs)")

    def load(self) -> bool:
        """Load BM25 state from disk. Returns True if index existed."""
        path = BM25_DIR / BM25_INDEX_FILE
        if not path.exists():
            logger.info(f"[BM25] No existing index at {path}")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.corpus = data.get("corpus", [])
        self.doc_ids = data.get("doc_ids", [])
        self.metadatas = data.get("metadatas", [])

        self._rebuild()
        self._dirty = False
        logger.info(f"[BM25] Loaded from {path} ({len(self.corpus)} docs)")
        return True
