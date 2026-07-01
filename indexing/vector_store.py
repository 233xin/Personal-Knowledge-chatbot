"""
Upgraded ChromaDB vector store wrapper.

Adds page-level and section-level filtering beyond the original
singleton implementation. Keeps BAAI/bge-small-zh-v1.5 embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_KWARGS,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper with semantic search, page, and section-level access."""

    def __init__(self):
        self._store: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._init_store()
        return self._store

    def _init_store(self):
        logger.info(f"Initializing ChromaDB with embedding model: {EMBEDDING_MODEL_NAME}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=EMBEDDING_MODEL_KWARGS,
        )

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        count = self._store._collection.count()
        logger.info(f"ChromaDB initialized. Collection has {count} documents.")

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and return their ChromaDB IDs."""
        return self.store.add_documents(documents)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        result = self.store.get(where={"doc_id": doc_id})
        ids = result.get("ids", [])
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> list[dict]:
        """List all unique documents in the store."""
        result = self.store.get()
        metadatas = result.get("metadatas", [])

        seen: dict[str, dict] = {}
        for meta in metadatas:
            if meta is None:
                continue
            doc_id = meta.get("doc_id", "")
            if not doc_id:
                continue
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_count": 0,
                    "page_count": 0,
                    "parser_used": meta.get("parser_source", "unknown"),
                }
            seen[doc_id]["chunk_count"] += 1
            pn = meta.get("page_num")
            if pn is not None and pn > seen[doc_id].get("_max_page", 0):
                seen[doc_id]["_max_page"] = pn

        # Convert internal _max_page to page_count
        for v in seen.values():
            v["page_count"] = v.pop("_max_page", 0)

        return list(seen.values())

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def as_retriever(self, search_kwargs: dict = None):
        """Get a LangChain retriever for use in RAG chains."""
        return self.store.as_retriever(search_kwargs=search_kwargs or {})

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[Document]:
        """Semantic (vector similarity) search."""
        search_filter = None
        if doc_ids:
            search_filter = {"doc_id": {"$in": list(doc_ids)}}

        retriever = self.store.as_retriever(
            search_kwargs={"k": top_k, "filter": search_filter}
        )
        return retriever.invoke(query)

    # ------------------------------------------------------------------
    # Page and section access
    # ------------------------------------------------------------------

    def get_by_page(self, doc_id: str, page_num: int) -> list[Document]:
        """Retrieve all chunks belonging to a specific page of a document."""
        result = self.store.get(
            where={"$and": [
                {"doc_id": {"$eq": doc_id}},
                {"page_num": {"$eq": page_num}},
            ]}
        )
        return self._docs_from_result(result)

    def get_by_section(self, doc_id: str, section_id: str) -> list[Document]:
        """Retrieve all chunks belonging to a specific section."""
        result = self.store.get(
            where={"$and": [
                {"doc_id": {"$eq": doc_id}},
                {"section_id": {"$eq": section_id}},
            ]}
        )
        return self._docs_from_result(result)

    def get_all(
        self,
        doc_id: Optional[str] = None,
    ) -> list[Document]:
        """Retrieve all chunks, optionally filtered by doc_id."""
        where = {"doc_id": {"$eq": doc_id}} if doc_id else None
        result = self.store.get(where=where)
        return self._docs_from_result(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _docs_from_result(result: dict) -> list[Document]:
        """Convert ChromaDB get() result to list of LangChain Documents."""
        docs = []
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            content = documents[i] if i < len(documents) else ""
            docs.append(Document(page_content=content, metadata=meta))

        return docs

    def count(self) -> int:
        """Return total number of chunks in the store."""
        return self.store._collection.count()
