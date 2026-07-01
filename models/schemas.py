"""
Pydantic models for API request/response schemas and internal data structures.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Parser Output Structures
# ---------------------------------------------------------------------------

class PageBlock(BaseModel):
    """Content of a single page with its page number."""
    page_num: int                           # 1-indexed
    markdown: str = ""                      # Markdown content on this page
    items: list[dict] = Field(default_factory=list)  # Structured items (headings, tables, text)


class TOCEntry(BaseModel):
    """A single entry in the table of contents."""
    title: str
    heading_level: int                      # 1-6
    page_num: Optional[int] = None
    section_id: str = ""                    # e.g. "$3.2"


class ParsedDocument(BaseModel):
    """Output of any parser: full markdown + per-page breakdown + TOC."""
    markdown: str                           # Full document as markdown
    pages: list[PageBlock] = Field(default_factory=list)
    toc: list[TOCEntry] = Field(default_factory=list)
    parser_used: str = "unknown"            # "marker" / "llamaparse" / "dashscope_ocr" / "plain_text"


# ---------------------------------------------------------------------------
# API Request Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class CitationInfo(BaseModel):
    """A single citation extracted from or attached to an answer."""
    filename: str
    page_num: Optional[int] = None
    section_path: str = ""
    quoted_text: str = ""                   # The actual text snippet cited
    table_id: Optional[str] = None
    clause_num: Optional[str] = None


class ChunkPreview(BaseModel):
    filename: str
    page_num: Optional[int] = None
    section_path: str = ""
    content_preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationInfo] = Field(default_factory=list)
    chunks: list[ChunkPreview] = Field(default_factory=list)
    evaluation: Optional[dict] = None       # Populated after evaluation pass


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int = 0
    parser_used: str = "unknown"


# ---------------------------------------------------------------------------
# Evaluation Structures
# ---------------------------------------------------------------------------

class CitationVerification(BaseModel):
    passed: bool = False
    total_cited_pages: int = 0
    verified_pages: int = 0
    hallucinated_pages: list[int] = Field(default_factory=list)
    missing_page_references: list[str] = Field(default_factory=list)


class TableCompleteness(BaseModel):
    passed: bool = False
    tables_in_source: int = 0
    tables_in_chunks: int = 0
    missing_tables: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    overall_pass: bool
    page_citation_check: CitationVerification = Field(default_factory=CitationVerification)
    table_completeness_check: TableCompleteness = Field(default_factory=TableCompleteness)
    details: list[str] = Field(default_factory=list)
