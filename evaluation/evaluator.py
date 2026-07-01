"""
Evaluator orchestrator — runs page citation verification and
table completeness check, then produces a unified pass/fail report.

These are HARD PASS criteria for production deployment.
"""

import logging
from typing import Optional

from evaluation.page_verifier import PageCitationVerifier
from evaluation.completeness_checker import CompletenessChecker
from models.schemas import EvaluationReport

logger = logging.getLogger(__name__)


class Evaluator:
    """Run both hard-pass checks and produce a unified evaluation report."""

    def __init__(self):
        self.page_verifier = PageCitationVerifier()
        self.completeness_checker = CompletenessChecker()

    def evaluate(
        self,
        answer: str,
        retrieved_chunks: list[dict],
        source_markdown: str,
        citation_extractor=None,
    ) -> EvaluationReport:
        """
        Args:
            answer: The LLM-generated answer.
            retrieved_chunks: The context chunks used for generation.
            source_markdown: Original parsed markdown (for table check).
            citation_extractor: Object with extract_citations(answer) method.

        Returns:
            EvaluationReport with overall pass/fail and detailed breakdown.
        """
        details = []

        # Check 1: Page citation accuracy
        page_check = self.page_verifier.verify(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citation_extractor=citation_extractor,
        )

        if not page_check.passed:
            if page_check.hallucinated_pages:
                details.append(
                    f"页码幻觉：引用了不存在的页码 {page_check.hallucinated_pages}"
                )
            if page_check.total_cited_pages == 0:
                details.append("未找到任何页码引用 — 回答未提供证据链")
            if page_check.missing_page_references:
                details.append(
                    f"未引用的片段：{page_check.missing_page_references[:5]}"
                )

        # Check 2: Table completeness
        chunk_texts = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                content = chunk.get("content", "")
            elif hasattr(chunk, "page_content"):
                content = chunk.page_content
            else:
                content = str(chunk)
            chunk_texts.append(content)

        table_check = self.completeness_checker.verify(
            source_markdown=source_markdown,
            chunk_contents=chunk_texts,
        )

        if not table_check.passed:
            details.append(
                f"表格缺失：以下表格未在检索结果中找到 — {table_check.missing_tables}"
            )

        overall = page_check.passed and table_check.passed

        logger.info(
            f"[Evaluator] Overall={'PASS' if overall else 'FAIL'}: "
            f"page_check={'PASS' if page_check.passed else 'FAIL'}, "
            f"table_check={'PASS' if table_check.passed else 'FAIL'}"
        )

        return EvaluationReport(
            overall_pass=overall,
            page_citation_check=page_check,
            table_completeness_check=table_check,
            details=details,
        )
