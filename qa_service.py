"""
Question-answering service using LangChain LCEL (LangChain Expression Language)
for Retrieval-Augmented Generation (RAG).

Retrieves relevant document chunks and generates answers with Claude.
"""

import os
import uuid
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter

from langchain_core.runnables import RunnableLambda, RunnableParallel

from vector_store import get_vector_store

SYSTEM_PROMPT = """You are a helpful personal knowledge base assistant. Your task is to answer the user's question based ONLY on the provided document excerpts.

Rules:
1. Answer ONLY using information found in the provided context below.
2. If the context doesn't contain the answer, say "根据你提供的文档，我无法找到相关信息来回答这个问题。" (Based on your documents, I couldn't find relevant information to answer this question.)
3. When answering, cite which document(s) the information comes from — mention the filename.
4. Keep answers concise and well-structured. Use bullet points when appropriate.
5. If the context is partially relevant but incomplete, acknowledge what you know and what's missing.
6. Respond in the same language as the user's question (Chinese or English)."""


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "unknown")
        parts.append(f"[来源 {i+1}: {filename}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class QAService:
    """
    Handles question answering via LCEL RAG chain and document CRUD.
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=2000,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        """Build the LCEL RAG chain: retrieve context → prompt → LLM → string."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",
                "以下是我上传的文档中的相关内容片段：\n\n"
                "{context}\n\n"
                "---\n"
                "基于以上文档内容，请回答我的问题：{question}"),
        ])

        return (
            RunnableParallel({
                "context": itemgetter("context") | RunnableLambda(_format_docs),
                "question": itemgetter("question"),
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # --- Q&A ---

    def answer(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG over the knowledge base.

        Args:
            question: The user's question.
            doc_ids: Optional list of document IDs to restrict search to.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with 'question', 'answer', 'sources', and 'chunks'.
        """
        # Build filter for doc_ids (ChromaDB uses dict-based where filters)
        search_filter = None
        if doc_ids:
            search_filter = {"doc_id": {"$in": list(doc_ids)}}

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k, "filter": search_filter}
        )
        docs = retriever.invoke(question)

        if not docs:
            return {
                "question": question,
                "answer": "没有找到相关文档。请先上传一些文档后再提问。",
                "sources": [],
                "chunks": [],
            }

        # Run the LCEL chain
        answer = self._chain.invoke({"context": docs, "question": question})

        # Collect unique source filenames
        source_filenames = list(dict.fromkeys(
            doc.metadata.get("filename", "unknown") for doc in docs
        ))

        return {
            "question": question,
            "answer": answer,
            "sources": source_filenames,
            "chunks": [
                {
                    "filename": doc.metadata.get("filename", "unknown"),
                    "content_preview": doc.page_content[:200] + "...",
                }
                for doc in docs
            ],
        }

    # --- Document CRUD ---

    def add_document(self, documents: List[Document]) -> str:
        """
        Add document chunks to the vector store.

        Args:
            documents: List of Document objects with metadata.

        Returns:
            The generated document ID.
        """
        doc_id = str(uuid.uuid4())
        for doc in documents:
            doc.metadata["doc_id"] = doc_id

        self.vector_store.add_documents(documents)
        return doc_id

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks belonging to a document.
        """
        result = self.vector_store.get(where={"doc_id": doc_id})
        ids = result.get("ids", [])
        if ids:
            self.vector_store.delete(ids=ids)
            return True
        return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the knowledge base.
        """
        result = self.vector_store.get()
        metadatas = result.get("metadatas", [])

        seen: Dict[str, Dict[str, Any]] = {}
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
                }
            seen[doc_id]["chunk_count"] += 1

        return list(seen.values())
