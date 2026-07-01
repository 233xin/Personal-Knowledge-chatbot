"""
Lineage metadata for document chunks — the "bloodline" that enables
full traceability from answer back to source page and section.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

from langchain_core.documents import Document


@dataclass
class LineageMetadata:
    """Bloodline metadata attached to every chunk for full traceability.

    Every chunk carries its provenance: which document, which page,
    which section hierarchy, and optionally which table or clause.
    """

    doc_id: str = ""
    filename: str = ""
    page_num: Optional[int] = None
    section_path: list[str] = field(default_factory=list)  # ["Chapter 3", "3.2"]
    section_id: Optional[str] = None       # Compact: "$3.2"
    heading_hierarchy: str = ""            # "Chapter 3 > 3.2 > Foo"
    table_id: Optional[str] = None         # "Table 4.1" / "表4.1"
    clause_num: Optional[str] = None       # "Article 17(3)" / "第17条第3款"
    chunk_index: int = 0
    total_chunks: int = 1
    parser_source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LineageMetadata":
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: d.get(k) for k in valid_keys if k in d})


class ChunkDocument:
    """Factory for LangChain Documents with typed lineage metadata.

    All metadata is stored as a plain dict in Document.metadata for
    ChromaDB compatibility. This class provides typed accessors.
    """

    @staticmethod
    def create(page_content: str, lineage: LineageMetadata) -> Document:
        return Document(
            page_content=page_content,
            metadata=lineage.to_dict(),
        )

    @staticmethod
    def get_lineage(doc: Document) -> LineageMetadata:
        return LineageMetadata.from_dict(doc.metadata)

    @staticmethod
    def format_citation(doc: Document) -> str:
        """Produce a Chinese-format citation string from a chunk."""
        lm = ChunkDocument.get_lineage(doc)
        parts = [f"《{lm.filename}》"]
        if lm.page_num is not None:
            parts.append(f"第{lm.page_num}页")
        if lm.heading_hierarchy:
            parts.append(f"{lm.heading_hierarchy}")
        if lm.table_id:
            parts.append(f"（{lm.table_id}）")
        if lm.clause_num:
            parts.append(f"（{lm.clause_num}）")
        return "，".join(parts)

    @staticmethod
    def make_citation_ref(filename: str, page_num: Optional[int] = None,
                          heading: str = "", table_id: str = "",
                          clause_num: str = "") -> str:
        """Quick citation ref without needing a full Document object."""
        parts = [f"《{filename}》"]
        if page_num is not None:
            parts.append(f"第{page_num}页")
        if heading:
            parts.append(heading)
        if table_id:
            parts.append(f"（{table_id}）")
        if clause_num:
            parts.append(f"（{clause_num}）")
        return "，".join(parts)
