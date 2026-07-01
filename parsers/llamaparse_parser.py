"""
LlamaParse parser for long/complex documents.
Cloud API — requires LLAMA_CLOUD_API_KEY in environment.

LlamaParse excels at: long PDFs (50+ pages), complex layouts,
multi-column text, forms, and documents with heavy table content.
"""

import logging
from typing import Optional

from config import (
    LLAMA_CLOUD_API_KEY,
    LLAMAPARSE_RESULT_TYPE,
    LLAMAPARSE_MAX_PAGES,
)
from models.schemas import ParsedDocument, PageBlock, TOCEntry
from parsers.base import BaseParser

logger = logging.getLogger(__name__)


class LlamaParseParser(BaseParser):
    """Cloud-based parsing via LlamaParse for long or complex documents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_pages: Optional[int] = None,
    ):
        self._api_key = api_key or LLAMA_CLOUD_API_KEY
        self._max_pages = max_pages or LLAMAPARSE_MAX_PAGES

    @property
    def name(self) -> str:
        return "llamaparse"

    def parse(self, file_path: str, filename: str) -> ParsedDocument:
        if not self._api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not set. "
                "LlamaParse requires an API key from https://cloud.llamaindex.ai"
            )

        from llama_parse import LlamaParse

        logger.info(
            f"[{self.name}] Parsing {filename} with LlamaParse "
            f"(max_pages={self._max_pages})"
        )

        parser = LlamaParse(
            api_key=self._api_key,
            result_type=LLAMAPARSE_RESULT_TYPE,
            verbose=True,
            language="ch_sim",  # Simplified Chinese
            split_by_page=True,
        )

        # LlamaParse returns List[Document] when split_by_page=True
        documents = parser.load_data(file_path)

        if not documents:
            raise ValueError(
                f"LlamaParse returned no content for '{filename}'."
            )

        # Build per-page content
        pages = []
        full_parts = []
        toc = []

        for i, doc in enumerate(documents):
            page_num = doc.metadata.get("page", i + 1) if doc.metadata else i + 1
            md = doc.text if hasattr(doc, "text") else str(doc)
            pages.append(PageBlock(page_num=page_num, markdown=md))
            full_parts.append(f"[PAGE: {page_num}]\n{md}")

        # Try to get structured JSON for TOC extraction
        try:
            json_result = parser.get_json_result(file_path)
            if json_result:
                toc = self._extract_toc_from_json(json_result)
        except Exception as e:
            logger.warning(f"[{self.name}] Could not extract TOC: {e}")

        return ParsedDocument(
            markdown="\n\n".join(full_parts),
            pages=pages,
            toc=toc,
            parser_used=self.name,
        )

    @staticmethod
    def _extract_toc_from_json(
        json_result: list[dict],
    ) -> list[TOCEntry]:
        """Extract TOC entries from LlamaParse JSON result."""
        entries = []
        if not json_result:
            return entries

        pages_data = json_result[0].get("pages", [])
        for page_data in pages_data:
            for item in page_data.get("items", []):
                if item.get("type") in ("heading", "section"):
                    entries.append(TOCEntry(
                        title=item.get("text", item.get("md", "")),
                        heading_level=item.get("level", 1),
                        page_num=page_data.get("page"),
                        section_id="",
                    ))
        return entries
