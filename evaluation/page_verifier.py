"""
PageCitationVerifier — validates that every page number cited in an answer
actually exists in the source chunks used for retrieval.

This is a HARD PASS criterion for production deployment:
zero hallucinated page numbers allowed.
"""

import logging
from typing import Optional

from models.schemas import CitationVerification

logger = logging.getLogger(__name__)


class PageCitationVerifier:
    """Cross-check cited pages against retrieved chunk metadata."""

    def verify(
        self,
        answer: str,
        retrieved_chunks: list[dict],
        citation_extractor,
    ) -> CitationVerification:
        """
        Args:
            answer: The LLM-generated answer text.
            retrieved_chunks: List of chunks used as context,
                each with "metadata" dict containing at least "filename" and "page_num".
            citation_extractor: Function or object with extract_citations(answer) method.

        Returns:
            CitationVerification with pass/fail and detailed diagnostics.
        """
        # Step 1: Extract all cited (filename, page_num) from answer
        if hasattr(citation_extractor, "extract_citations"):
            cited = citation_extractor.extract_citations(answer)
        else:
            cited = citation_extractor(answer)

        cited_pages: set[tuple[str, int]] = set()
        for c in cited:
            fn = c.get("filename", "")
            pn = c.get("page_num")
            if fn and pn is not None:
                cited_pages.add((fn, int(pn)))

        # Step 2: Build the set of available (filename, page_num) from chunks
        source_pages: set[tuple[str, int]] = set()
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            if hasattr(chunk, "metadata"):
                meta = chunk.metadata
            fn = meta.get("filename", "")
            pn = meta.get("page_num")
            if fn and pn is not None:
                source_pages.add((str(fn), int(pn)))

        # Step 3: Cross-check — which cited pages actually exist?
        verified = 0
        hallucinated = []
        for fn, pn in cited_pages:
            if (fn, pn) in source_pages:
                verified += 1
            else:
                hallucinated.append(pn)

        # Step 4: Check for chunks that should have been cited but weren't
        cited_filenames = {fn for fn, _ in cited_pages}
        missing_refs = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            if hasattr(chunk, "metadata"):
                meta = chunk.metadata
            fn = meta.get("filename", "")
            pn = meta.get("page_num")
            if fn and fn not in cited_filenames and pn is not None:
                missing_refs.append(f"{fn} p.{pn}")

        passed = len(hallucinated) == 0 and len(cited_pages) > 0

        logger.info(
            f"[PageVerifier] Passed={passed}: {verified}/{len(cited_pages)} "
            f"citations verified. Hallucinated: {hallucinated}. "
            f"Uncited chunks: {len(missing_refs)}"
        )

        return CitationVerification(
            passed=passed,
            total_cited_pages=len(cited_pages),
            verified_pages=verified,
            hallucinated_pages=hallucinated,
            missing_page_references=missing_refs[:10],  # Cap for API response
        )
