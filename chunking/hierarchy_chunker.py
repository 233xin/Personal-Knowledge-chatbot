"""
Heading-aware document chunker with lineage metadata attachment.

Core constraints:
1. Split ONLY at heading boundaries (H1, H2, H3) — NEVER mid-paragraph
2. NEVER concatenate content from different headings into one chunk
3. Attach bloodline metadata to every chunk: doc_id, page_num,
   section_path, heading_hierarchy, table_id, clause_num
4. Fall back to paragraph-boundary splitting when no headings detected
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from config import CHUNK_SIZE, MIN_CHUNK_SIZE, HEADING_PATTERN
from models.lineage import LineageMetadata, ChunkDocument
from models.schemas import ParsedDocument, PageBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heading tree data structures
# ---------------------------------------------------------------------------

@dataclass
class HeadingNode:
    """A single heading in the document tree."""
    level: int                          # 1-6
    title: str                          # Raw heading text (without # prefix)
    char_start: int                     # Offset where this heading starts
    char_end: int = 0                   # Offset where next heading at same/higher level starts
    parent: Optional["HeadingNode"] = None
    children: list["HeadingNode"] = field(default_factory=list)
    page_num: Optional[int] = None

    @property
    def path(self) -> list[str]:
        """Walk up the tree to build the full heading path."""
        node = self
        parts = [node.title]
        while node.parent is not None and node.parent.level > 0:
            node = node.parent
            parts.insert(0, node.title)
        return parts

    @property
    def section_id(self) -> str:
        """Derive a compact section identifier from the heading text."""
        # Try to find numbered patterns like "3.2" or "§3.2" or "第3章"
        numbered = re.search(r"[\d]+[\.\-\s]*[\d]*", self.title)
        if numbered:
            return f"§{numbered.group().strip()}"
        # Use first 20 chars as identifier
        return self.title[:20]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class HierarchyChunker:
    """Splits parsed documents by heading hierarchy with lineage metadata."""

    # Patterns for extracting table and clause references from headings
    TABLE_PATTERN = re.compile(r"(?:Table|表)\s*[\d]+[：:.\s]*([^\n]*)", re.IGNORECASE)
    CLAUSE_PATTERN = re.compile(
        r"(?:第[\d一二三四五六七八九十百]+条|Article\s*\d+|Clause\s*\d+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        max_chunk_size: int = CHUNK_SIZE,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        parsed: ParsedDocument,
        doc_id: str,
        filename: str,
    ) -> list[Document]:
        """Main entry point: parsed doc → list of Documents with lineage metadata.

        Args:
            parsed: ParsedDocument with markdown + page info.
            doc_id: UUID for this document (assigned by DocumentService).
            filename: Original filename.

        Returns:
            List of LangChain Documents, each with full LineageMetadata in metadata.
        """
        markdown = parsed.markdown

        # Step 1: Parse heading tree
        headings = self._parse_headings(markdown)

        # Step 2: Build page offset map
        page_map = self._build_page_map(markdown, parsed.pages)

        # Step 3: Assign page numbers to headings
        for h in headings:
            h.page_num = self._locate_page(h.char_start, page_map)

        # Step 4: Split into chunks at heading boundaries
        if not headings:
            # No headings found — fall back to paragraph splitting
            logger.warning(
                f"[HierarchyChunker] No headings found in '{filename}', "
                f"falling back to paragraph-level splitting."
            )
            chunks = self._fallback_paragraph_split(markdown)
        else:
            chunks = self._split_by_headings(markdown, headings)

        # Step 5: Build documents with lineage metadata
        documents = []
        for i, (chunk_text, section_heading) in enumerate(chunks):
            lineage = self._build_lineage(
                chunk_text=chunk_text,
                section_heading=section_heading,
                page_map=page_map,
                doc_id=doc_id,
                filename=filename,
                chunk_index=i,
                total_chunks=len(chunks),
                parser_source=parsed.parser_used,
            )
            documents.append(ChunkDocument.create(chunk_text, lineage))

        logger.info(
            f"[HierarchyChunker] '{filename}': {len(headings)} headings, "
            f"{len(documents)} chunks produced"
        )
        return documents

    # ------------------------------------------------------------------
    # Heading parsing
    # ------------------------------------------------------------------

    def _parse_headings(self, markdown: str) -> list[HeadingNode]:
        """Parse # headings into a flat list with character offsets.

        Uses a stack to build parent-child relationships.
        """
        heading_lines = []
        for m in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE):
            level = len(m.group(1))
            title = m.group(2).strip()
            char_start = m.start()
            heading_lines.append((level, title, char_start))

        if not heading_lines:
            return []

        # Build nodes with parent-child via stack
        nodes = []
        stack: list[HeadingNode] = []  # Stack of ancestors

        for i, (level, title, char_start) in enumerate(heading_lines):
            node = HeadingNode(
                level=level,
                title=title,
                char_start=char_start,
            )

            # Pop stack until we find a parent (lower level number = higher in hierarchy)
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)

            stack.append(node)
            nodes.append(node)

        # Set char_end for each node
        for i, node in enumerate(nodes):
            if i + 1 < len(nodes):
                # End at the next heading
                node.char_end = nodes[i + 1].char_start
            else:
                # Last heading extends to end of document
                node.char_end = len(markdown)

        return nodes

    # ------------------------------------------------------------------
    # Page mapping
    # ------------------------------------------------------------------

    def _build_page_map(
        self,
        markdown: str,
        pages: list[PageBlock],
    ) -> list[tuple[int, int]]:
        """Build (char_offset, page_num) mapping from page blocks.

        We concatenate all page content and track where each page starts.
        """
        page_map = []
        cum_offset = 0

        # Sort pages by page_num just in case
        sorted_pages = sorted(pages, key=lambda p: p.page_num)

        for page in sorted_pages:
            page_map.append((cum_offset, page.page_num))
            # Approximate: the page content occupies this many characters
            # plus some markdown overhead
            cum_offset += len(page.markdown) + 4  # +4 for "\n\n" separators

        return page_map

    @staticmethod
    def _locate_page(char_offset: int, page_map: list[tuple[int, int]]) -> Optional[int]:
        """Binary search: which page does this character offset belong to?"""
        if not page_map:
            return None

        page_num = page_map[0][1]  # Default to first page
        for offset, pn in page_map:
            if char_offset >= offset:
                page_num = pn
            else:
                break
        return page_num

    # ------------------------------------------------------------------
    # Chunk splitting
    # ------------------------------------------------------------------

    def _split_by_headings(
        self,
        markdown: str,
        headings: list[HeadingNode],
    ) -> list[tuple[str, Optional[HeadingNode]]]:
        """Split markdown at H1 and H2 boundaries. Never cross headings.

        Returns list of (chunk_text, heading_node_for_lineage).
        heading_node_for_lineage is the deepest heading that starts this chunk.
        """
        chunks = []

        # Primary boundaries: H1 and H2
        primary_levels = {1, 2}

        # Find all H1/H2 headings
        primary = [h for h in headings if h.level in primary_levels]

        if not primary:
            # No H1/H2 — use whatever headings we have
            primary = headings

        # Build chunks between primary headings
        for i, h in enumerate(primary):
            start = h.char_start
            if i + 1 < len(primary):
                end = primary[i + 1].char_start
            else:
                end = len(markdown)

            chunk_text = markdown[start:end].strip()

            if not chunk_text:
                continue

            # If this chunk is too large, sub-split at H3 boundaries
            if len(chunk_text) > self.max_chunk_size:
                sub_headings = [
                    sub for sub in headings
                    if sub.level >= 3
                    and sub.char_start >= start
                    and sub.char_start < end
                    and sub.parent is not None
                    and sub.parent is h
                ]
                if sub_headings:
                    sub_chunks = self._sub_split(chunk_text, sub_headings, start)
                    for sub_text, sub_h in sub_chunks:
                        chunks.append((sub_text, sub_h))
                    continue

            chunks.append((chunk_text, h))

        # Handle content before the first primary heading
        if primary and primary[0].char_start > 0:
            intro_text = markdown[:primary[0].char_start].strip()
            if intro_text and len(intro_text) >= self.min_chunk_size:
                # Insert at beginning
                chunks.insert(0, (intro_text, None))

        return chunks

    def _sub_split(
        self,
        text: str,
        sub_headings: list[HeadingNode],
        base_offset: int,
    ) -> list[tuple[str, HeadingNode]]:
        """Split a large section at H3+ boundaries."""
        chunks = []

        for i, h in enumerate(sub_headings):
            rel_start = h.char_start - base_offset
            if i + 1 < len(sub_headings):
                rel_end = sub_headings[i + 1].char_start - base_offset
            else:
                rel_end = len(text)

            chunk_text = text[rel_start:rel_end].strip()
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append((chunk_text, h))

        return chunks

    # ------------------------------------------------------------------
    # Fallback — no headings detected
    # ------------------------------------------------------------------

    def _fallback_paragraph_split(self, text: str) -> list[tuple[str, None]]:
        """Split by double-newline (paragraph) boundaries when no headings."""
        paragraphs = re.split(r"\n\n+", text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= self.max_chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current and len(current) >= self.min_chunk_size:
                    chunks.append((current, None))
                current = para

        if current and len(current) >= self.min_chunk_size:
            chunks.append((current, None))

        return chunks

    # ------------------------------------------------------------------
    # Lineage metadata construction
    # ------------------------------------------------------------------

    def _build_lineage(
        self,
        chunk_text: str,
        section_heading: Optional[HeadingNode],
        page_map: list[tuple[int, int]],
        doc_id: str,
        filename: str,
        chunk_index: int,
        total_chunks: int,
        parser_source: str,
    ) -> LineageMetadata:
        """Build full LineageMetadata for a single chunk."""
        # Determine page number
        page_num = None
        if section_heading and section_heading.page_num:
            page_num = section_heading.page_num
        else:
            # Use first page in map
            page_num = page_map[0][1] if page_map else None

        # Build section path
        section_path = section_heading.path if section_heading else []
        section_id = section_heading.section_id if section_heading else ""
        heading_hierarchy = " > ".join(section_path) if section_path else ""

        # Extract table/clause references from chunk text and heading
        table_id = self._extract_table_id(chunk_text, section_heading)
        clause_num = self._extract_clause(chunk_text, section_heading)

        return LineageMetadata(
            doc_id=doc_id,
            filename=filename,
            page_num=page_num,
            section_path=section_path,
            section_id=section_id,
            heading_hierarchy=heading_hierarchy,
            table_id=table_id,
            clause_num=clause_num,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            parser_source=parser_source,
        )

    def _extract_table_id(
        self,
        chunk_text: str,
        heading: Optional[HeadingNode],
    ) -> Optional[str]:
        """Extract table identifier from heading or chunk content."""
        # Check heading first
        if heading:
            m = self.TABLE_PATTERN.search(heading.title)
            if m:
                return f"表{m.group().strip()}"
        # Check chunk text
        m = self.TABLE_PATTERN.search(chunk_text[:200])
        if m:
            return f"表{m.group().strip()}"
        return None

    def _extract_clause(
        self,
        chunk_text: str,
        heading: Optional[HeadingNode],
    ) -> Optional[str]:
        """Extract clause number reference."""
        text_to_search = (heading.title if heading else "") + " " + chunk_text[:300]
        m = self.CLAUSE_PATTERN.search(text_to_search)
        return m.group() if m else None
