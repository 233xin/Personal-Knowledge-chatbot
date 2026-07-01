"""
Marker PDF/DOCX parser — local ML-based conversion to markdown
with heading hierarchy and page number preservation.

Requires: marker-pdf (PyTorch-based, GPU recommended).
Falls back to DashScopeMarkdownParser if GPU unavailable.
"""

import logging
from typing import Optional

from models.schemas import ParsedDocument, PageBlock, TOCEntry
from parsers.base import BaseParser

logger = logging.getLogger(__name__)


class MarkerPdfParser(BaseParser):
    """Use Marker (marker-pdf) to convert PDF/DOCX to structured markdown.

    Marker preserves heading hierarchy (# ## ###), extracts tables,
    and optionally embeds page separators via paginate_output=True.
    """

    def __init__(self, fallback_parser: Optional["BaseParser"] = None):
        self._fallback = fallback_parser
        self._available = None  # Tri-state: None=unchecked, True/False

    @property
    def name(self) -> str:
        return "marker"

    def _check_availability(self) -> bool:
        """Lazy-check if Marker is importable and has GPU."""
        if self._available is not None:
            return self._available

        try:
            import torch
            self._available = torch.cuda.is_available()
            if not self._available:
                logger.warning(
                    "[Marker] No GPU detected. Marker performance will be slow. "
                    "Consider using DashScopeMarkdownParser as fallback."
                )
                # Still mark as available — CPU fallback works, just slow
                self._available = True
        except ImportError:
            logger.warning(
                "[Marker] PyTorch not installed. Marker unavailable. "
                "Install with: pip install marker-pdf"
            )
            self._available = False

        return self._available

    def parse(self, file_path: str, filename: str) -> ParsedDocument:
        if not self._check_availability():
            if self._fallback:
                logger.info(
                    f"[{self.name}] Marker unavailable, "
                    f"falling back to {self._fallback.name}"
                )
                return self._fallback.parse(file_path, filename)
            raise ValueError(
                "Marker parser is not available (PyTorch not found). "
                "Install marker-pdf or configure a fallback parser."
            )

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        logger.info(f"[{self.name}] Parsing {filename} with Marker")

        model_dict = create_model_dict()
        converter = PdfConverter(
            artifact_dict=model_dict,
            config={
                "output_format": "markdown",
                "paginate_output": True,
                "extract_images": False,
            },
        )

        rendered = converter(file_path)
        markdown_text, metadata, _images = text_from_rendered(rendered)

        if not markdown_text or not markdown_text.strip():
            raise ValueError(
                f"Marker could not extract any text from '{filename}'."
            )

        # Parse TOC from metadata
        toc = []
        toc_list = metadata.get("table_of_contents", []) if metadata else []
        if toc_list:
            for entry in toc_list:
                toc.append(TOCEntry(
                    title=entry.get("title", ""),
                    heading_level=entry.get("heading_level", 1),
                    page_num=entry.get("page_id"),
                    section_id="",
                ))

        # Split into pages using Marker's page separators
        pages = self._split_by_marker_pages(markdown_text)

        return ParsedDocument(
            markdown=markdown_text,
            pages=pages,
            toc=toc,
            parser_used=self.name,
        )

    @staticmethod
    def _split_by_marker_pages(markdown: str) -> list[PageBlock]:
        """
        Marker's paginate_output inserts page separators as:

        (content)

        3
        ------------------------------------------------

        (next page content)

        We split on this pattern.
        """
        import re
        # Pattern: newline, page number, newline, 48 dashes, newline
        pattern = r"\n+(\d+)\n[-]{40,}\n+"
        parts = re.split(pattern, markdown)

        pages = []
        # parts[0] = content before first page marker
        # parts[1] = page_num, parts[2] = content, parts[3] = page_num, ...
        i = 0
        if parts and not re.match(r"^\d+$", parts[0].strip()):
            if parts[0].strip():
                pages.append(PageBlock(page_num=1, markdown=parts[0].strip()))
            i = 1

        while i + 1 < len(parts):
            try:
                page_num = int(parts[i].strip())
            except ValueError:
                page_num = len(pages) + 1
            content = parts[i + 1].strip()
            if content:
                pages.append(PageBlock(page_num=page_num, markdown=content))
            i += 2

        return pages
