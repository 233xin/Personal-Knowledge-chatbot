"""
Abstract base class for all document parsers.
"""

from abc import ABC, abstractmethod

from models.schemas import ParsedDocument


class BaseParser(ABC):
    """Every parser extracts text from a file and returns a ParsedDocument."""

    @abstractmethod
    def parse(self, file_path: str, filename: str) -> ParsedDocument:
        """Extract structured content from a file.

        Args:
            file_path: Absolute path to the temp file on disk.
            filename: Original filename (for metadata and extension detection).

        Returns:
            ParsedDocument with markdown, per-page blocks, and TOC.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable parser identifier for lineage tracking."""
        ...
