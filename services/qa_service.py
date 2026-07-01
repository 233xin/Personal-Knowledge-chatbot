"""
Refactored QAService — Agent-based retrieval + citation-enforced generation + evaluation.

Pipeline:
  User question → RetrievalAgent (dynamic tool calls) →
  CitationGenerator (citation-enforced prompt) → answer with evidence chain →
  Evaluator (page number + table completeness checks)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_MAX_TOKENS,
)
from evaluation import Evaluator
from generation import CitationGenerator
from retrieval import RetrievalAgent

logger = logging.getLogger(__name__)


class QAService:
    """End-to-end Q&A with agent retrieval, citation generation, and evaluation."""

    def __init__(self, document_service=None):
        # LLM for the retrieval agent
        self._agent_llm = ChatOpenAI(
            model=DEEPSEEK_CHAT_MODEL,
            temperature=0.0,  # Deterministic for tool selection
            max_tokens=1000,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

        # LLM for answer generation
        self._gen_llm = ChatOpenAI(
            model=DEEPSEEK_CHAT_MODEL,
            temperature=DEEPSEEK_TEMPERATURE,
            max_tokens=DEEPSEEK_MAX_TOKENS,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

        # Wire up document service to retrieval tools if provided
        if document_service is not None:
            from retrieval.tools import set_document_service
            set_document_service(document_service)

        # Subsystems
        self.retrieval_agent = RetrievalAgent(llm=self._agent_llm)
        self.citation_generator = CitationGenerator(llm=self._gen_llm)
        self.evaluator = Evaluator()

        # Source markdown cache (for evaluation — stores per doc_id)
        self._source_markdowns: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Answer pipeline
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full Q&A pipeline:
        1. RetrievalAgent dynamically fetches relevant chunks
        2. CitationGenerator produces answer with page/section citations
        3. Evaluator verifies page accuracy and table completeness
        """
        # Step 1: Agent-based retrieval
        logger.info(f"[QAService] Question: {question[:80]}...")
        context = self.retrieval_agent.retrieve(
            question=question,
            doc_ids=doc_ids,
        )

        if not context or "未找到相关结果" in context:
            return {
                "question": question,
                "answer": "没有找到相关文档。请先上传一些文档后再提问。",
                "citations": [],
                "chunks": [],
                "evaluation": None,
            }

        # Step 2: Citation-enforced generation
        answer = self.citation_generator.generate(
            question=question,
            context=context,
        )

        # Step 3: Parse citations
        raw_citations = self.citation_generator.extract_citations(answer)
        citations = [
            {
                "filename": c["filename"],
                "page_num": c["page_num"],
                "section_path": c["section_path"],
                "quoted_text": c["quoted_text"],
            }
            for c in raw_citations
        ]

        # Step 4: Build chunk previews from retrieval context
        chunk_previews = self._extract_chunk_previews(context)

        # Step 5: Evaluation (hard pass checks)
        evaluation = None
        try:
            # Use any available source markdown
            source_md = ""
            if self._source_markdowns:
                source_md = list(self._source_markdowns.values())[0]

            # Convert chunk_previews to the format evaluator expects
            eval_chunks = [
                {"metadata": {"filename": cp["filename"], "page_num": cp.get("page_num")}}
                for cp in chunk_previews
            ]

            report = self.evaluator.evaluate(
                answer=answer,
                retrieved_chunks=eval_chunks,
                source_markdown=source_md,
                citation_extractor=self.citation_generator,
            )
            evaluation = report.model_dump()
        except Exception as e:
            logger.warning(f"[QAService] Evaluation skipped: {e}")

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "chunks": chunk_previews,
            "evaluation": evaluation,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chunk_previews(context: str) -> list[dict]:
        """Extract chunk previews from the agent's retrieved context string."""
        import re

        previews = []

        # Parse the structured format: [结果 N] CITATION: ... \n CONTENT:\n ...
        pattern = r"\[结果\s*(\d+)\]\s*CITATION:\s*(.+?)\nCONTENT:\n(.+?)(?:\n---\n|\n\[结果|\Z)"
        for m in re.finditer(pattern, context, re.DOTALL):
            citation_str = m.group(2).strip()
            content = m.group(3).strip()

            # Parse citation: 《filename》第N页，[hierarchy]（[extras]）
            filename = ""
            page_num = None
            section_path = ""

            fn_match = re.search(r"《(.+?)》", citation_str)
            if fn_match:
                filename = fn_match.group(1)

            pn_match = re.search(r"第(\d+)页", citation_str)
            if pn_match:
                page_num = int(pn_match.group(1))

            previews.append({
                "filename": filename,
                "page_num": page_num,
                "section_path": section_path,
                "content_preview": content[:300] + ("..." if len(content) > 300 else ""),
            })

        return previews

    def cache_source_markdown(self, doc_id: str, markdown: str) -> None:
        """Cache parsed markdown for later evaluation use."""
        self._source_markdowns[doc_id] = markdown
