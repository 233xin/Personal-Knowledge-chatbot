"""
ParserRouter — classifies incoming files and dispatches to the correct parser.

Decision tree:
  .pdf  → classify (digital vs scanned) → route by length/type
  .docx → MarkerPdfParser (or DashScopeMarkdownParser fallback)
  .doc  → PlainTextParser with warning
  images → DashScopeOCRParser
  text/code → PlainTextParser
"""

import logging
import os
from typing import Optional

from config import (
    PDF_TEXT_THRESHOLD,
    LONG_DOC_THRESHOLD,
    PDF_SAMPLE_PAGES,
    MARKER_EXTENSIONS,
    DASHSCOPE_OCR_IMAGE_EXTS,
    PLAIN_TEXT_EXTENSIONS,
)
from models.schemas import ParsedDocument
from parsers.base import BaseParser
from parsers.plain_text import PlainTextParser
from parsers.dashscope_ocr import DashScopeOCRParser
from parsers.dashscope_markdown import DashScopeMarkdownParser

logger = logging.getLogger(__name__)


class ParserRouter:
    """Route each file to the appropriate parser based on type and content."""

    def __init__(
        self,
        marker_parser: Optional[BaseParser] = None,
        llamaparse_parser: Optional[BaseParser] = None,
        dashscope_ocr: Optional[BaseParser] = None,
        dashscope_markdown: Optional[BaseParser] = None,
        plain_parser: Optional[BaseParser] = None,
    ):
        # Lazy-init for heavy parsers
        self._marker = marker_parser
        self._llamaparse = llamaparse_parser
        self._dashscope_ocr = dashscope_ocr
        self._dashscope_md = dashscope_markdown
        self._plain = plain_parser or PlainTextParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str, filename: str) -> ParsedDocument:
        """Classify the file and dispatch to the correct parser."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path, filename)
        elif ext in MARKER_EXTENSIONS:  # .docx
            return self._parse_docx(file_path, filename)
        elif ext in DASHSCOPE_OCR_IMAGE_EXTS:
            return self._get_dashscope_ocr().parse(file_path, filename)
        elif ext in PLAIN_TEXT_EXTENSIONS:
            return self._plain.parse(file_path, filename)
        else:
            # Unknown extension — try as plain text
            logger.info(
                f"[ParserRouter] Unknown extension '{ext}' for '{filename}', "
                f"trying plain text"
            )
            try:
                return self._plain.parse(file_path, filename)
            except ValueError:
                raise ValueError(
                    f"Unsupported file type: {ext}. "
                    f"Supported: PDF, DOCX, TXT, MD, CSV, JSON, HTML, "
                    f"code files, PNG, JPG, TIFF."
                )

    # ------------------------------------------------------------------
    # PDF classification & routing
    # ------------------------------------------------------------------

    def _parse_pdf(self, file_path: str, filename: str) -> ParsedDocument:
        classification, page_count = self._classify_pdf(file_path)
        logger.info(
            f"[ParserRouter] PDF classification: {classification}, "
            f"pages={page_count}, file={filename}"
        )

        if classification == "scanned":
            return self._get_dashscope_ocr().parse(file_path, filename)
        elif page_count >= LONG_DOC_THRESHOLD:
            # Long digital PDF → LlamaParse
            return self._get_llamaparse().parse(file_path, filename)
        else:
            # Short digital PDF → Marker
            return self._get_marker().parse(file_path, filename)

    def _classify_pdf(self, file_path: str) -> tuple[str, int]:
        """
        Classify a PDF as 'digital' or 'scanned' by sampling text density.

        Returns (classification, page_count).
        """
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        page_count = len(reader.pages)

        total_chars = 0
        sampled = 0
        for i, page in enumerate(reader.pages):
            if i >= PDF_SAMPLE_PAGES:
                break
            text = page.extract_text() or ""
            total_chars += len(text)
            sampled += 1

        avg_chars = total_chars / max(sampled, 1)
        classification = "digital" if avg_chars >= PDF_TEXT_THRESHOLD else "scanned"
        return classification, page_count

    # ------------------------------------------------------------------
    # DOCX routing
    # ------------------------------------------------------------------

    def _parse_docx(self, file_path: str, filename: str) -> ParsedDocument:
        """Try Marker first, fall back to DashScope if Marker unavailable."""
        marker = self._get_marker()
        try:
            return marker.parse(file_path, filename)
        except ValueError as e:
            logger.warning(
                f"[ParserRouter] Marker failed for '{filename}': {e}. "
                f"Falling back to DashScopeMarkdownParser."
            )
            return self._get_dashscope_md().parse(file_path, filename)

    # ------------------------------------------------------------------
    # Lazy parser accessors
    # ------------------------------------------------------------------

    def _get_marker(self) -> BaseParser:
        if self._marker is None:
            from parsers.marker_parser import MarkerPdfParser
            fallback = self._get_dashscope_md()
            self._marker = MarkerPdfParser(fallback_parser=fallback)
        return self._marker

    def _get_llamaparse(self) -> BaseParser:
        if self._llamaparse is None:
            from parsers.llamaparse_parser import LlamaParseParser
            self._llamaparse = LlamaParseParser()
        return self._llamaparse

    def _get_dashscope_ocr(self) -> BaseParser:
        if self._dashscope_ocr is None:
            self._dashscope_ocr = DashScopeOCRParser()
        return self._dashscope_ocr

    def _get_dashscope_md(self) -> BaseParser:
        if self._dashscope_md is None:
            self._dashscope_md = DashScopeMarkdownParser()
        return self._dashscope_md
