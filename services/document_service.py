"""
DocumentService — coordinates parsing, chunking, and dual-index storage.

This is the central orchestrator for the upload pipeline:
  ParserRouter → HierarchyChunker → Dual Index (ChromaDB + BM25)
"""

import logging
import uuid
from typing import Optional

from langchain_core.documents import Document

from chunking.hierarchy_chunker import HierarchyChunker
from indexing.vector_store import VectorStore
from indexing.bm25_store import BM25Index
from parsers.router import ParserRouter

logger = logging.getLogger(__name__)


class DocumentService:
    """End-to-end document ingestion and lifecycle management."""

    def __init__(
        self,
        parser_router: Optional[ParserRouter] = None,
        chunker: Optional[HierarchyChunker] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
    ):
        self.parser = parser_router or ParserRouter()
        self.chunker = chunker or HierarchyChunker()
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()

        # Load BM25 from disk if it exists
        self.bm25_index.load()

    # ------------------------------------------------------------------
    # Upload pipeline
    # ------------------------------------------------------------------

    def ingest(self, file_path: str, filename: str) -> dict:
        """
        Full ingestion pipeline: parse → chunk → dual-index.

        Returns dict with doc_id, filename, chunk_count, page_count, parser_used.
        """
        # Step 1: Parse
        logger.info(f"[DocumentService] Parsing '{filename}'")
        parsed = self.parser.parse(file_path, filename)
        logger.info(
            f"[DocumentService] Parsed by {parsed.parser_used}: "
            f"{len(parsed.markdown)} chars, {len(parsed.pages)} pages"
        )

        # Step 2: Generate doc_id
        doc_id = str(uuid.uuid4())

        # Step 3: Chunk with lineage
        documents = self.chunker.chunk(
            parsed=parsed,
            doc_id=doc_id,
            filename=filename,
        )
        logger.info(f"[DocumentService] Chunked into {len(documents)} chunks")

        # Step 4: Store in dual indexes
        self.vector_store.add_documents(documents)
        self.bm25_index.add_documents(documents)
        self.bm25_index.save()

        logger.info(
            f"[DocumentService] Ingested '{filename}' as doc_id={doc_id}: "
            f"{len(documents)} chunks, {len(parsed.pages)} pages, "
            f"parser={parsed.parser_used}"
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(documents),
            "page_count": len(parsed.pages),
            "parser_used": parsed.parser_used,
        }

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------

    def delete(self, doc_id: str) -> bool:
        """Delete a document from both indexes."""
        vec_count = self.vector_store.delete_by_doc_id(doc_id)
        bm25_count = self.bm25_index.delete_by_doc_id(doc_id)
        if bm25_count > 0:
            self.bm25_index.save()
        deleted = (vec_count + bm25_count) > 0
        logger.info(
            f"[DocumentService] Deleted doc_id={doc_id}: "
            f"{vec_count} vector + {bm25_count} BM25 chunks"
        )
        return deleted

    def list_documents(self) -> list[dict]:
        """List all documents in the knowledge base."""
        return self.vector_store.list_documents()

    # ------------------------------------------------------------------
    # Search accessors (used by retrieval tools)
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[Document]:
        return self.vector_store.search(query, top_k=top_k, doc_ids=doc_ids)

    def search_keyword(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        return self.bm25_index.search(query, top_k=top_k, doc_ids=doc_ids)

    def get_chunks_by_page(self, doc_id: str, page_num: int) -> list[Document]:
        return self.vector_store.get_by_page(doc_id, page_num)

    def get_chunks_by_section(self, doc_id: str, section_id: str) -> list[Document]:
        return self.vector_store.get_by_section(doc_id, section_id)

    def get_all_chunks(self, doc_id: Optional[str] = None) -> list[Document]:
        return self.vector_store.get_all(doc_id=doc_id)
