"""
Plain text parser for .txt, .md, .csv, .json, .html, code files, etc.
Extracted from the original document_processor.py _extract_from_text logic.
"""

from models.schemas import ParsedDocument, PageBlock
from parsers.base import BaseParser


class PlainTextParser(BaseParser):
    """Extracts text from plain-text file formats, trying common encodings."""

    ENCODINGS = ["utf-8", "gbk", "gb2312", "latin-1", "cp1252"]

    @property
    def name(self) -> str:
        return "plain_text"

    def parse(self, file_path: str, filename: str) -> ParsedDocument:
        text = self._read_with_encodings(file_path)
        if not text.strip():
            raise ValueError(
                f"Could not read '{filename}' with any supported encoding. "
                f"Tried: {', '.join(self.ENCODINGS)}"
            )

        return ParsedDocument(
            markdown=text,
            pages=[PageBlock(page_num=1, markdown=text)],
            toc=[],
            parser_used=self.name,
        )

    def _read_with_encodings(self, file_path: str) -> str:
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                if text.strip():
                    return text
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError("No working encoding found.")
