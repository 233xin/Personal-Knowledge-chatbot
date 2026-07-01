"""
CompletenessChecker — verifies that all tables from the source document
are captured in the retrieved chunks.

This is a HARD PASS criterion: tables containing critical data must
not be lost during parsing or chunking.
"""

import re
import logging

from models.schemas import TableCompleteness

logger = logging.getLogger(__name__)


class CompletenessChecker:
    """Verify that source document tables are present in retrieval chunks."""

    # Regex for markdown tables (lines with | ... | ... | structure)
    MD_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)

    # Regex for named tables
    NAMED_TABLE_RE = re.compile(
        r"(?:Table|表)\s*[\d]+[：:.\s]*",
        re.IGNORECASE,
    )

    def verify(
        self,
        source_markdown: str,
        chunk_contents: list[str],
    ) -> TableCompleteness:
        """
        Args:
            source_markdown: The original markdown after parsing.
            chunk_contents: List of chunk text strings from retrieval results.

        Returns:
            TableCompleteness with pass/fail and missing table diagnostics.
        """
        # Detect tables in source
        source_tables = self._detect_tables(source_markdown)

        # Detect which appear in chunks
        all_chunk_text = "\n".join(chunk_contents)

        tables_found = 0
        missing = []
        for table_name, table_preview in source_tables:
            if table_preview[:40] in all_chunk_text:
                tables_found += 1
            else:
                missing.append(table_name)

        passed = len(missing) == 0 and len(source_tables) > 0

        logger.info(
            f"[Completeness] Passed={passed}: {tables_found}/{len(source_tables)} "
            f"tables found in chunks. Missing: {missing}"
        )

        return TableCompleteness(
            passed=passed,
            tables_in_source=len(source_tables),
            tables_in_chunks=tables_found,
            missing_tables=missing,
        )

    def _detect_tables(self, markdown: str) -> list[tuple[str, str]]:
        """Find all tables in markdown. Returns (name, preview_content) tuples."""
        tables = []

        # Find markdown table blocks (consecutive |...| lines)
        lines = markdown.split("\n")
        in_table = False
        table_lines = []
        table_name = ""

        for i, line in enumerate(lines):
            stripped = line.strip()
            if self.MD_TABLE_ROW_RE.match(stripped) or (
                "|---" in stripped and "|" in stripped
            ):
                if not in_table:
                    # Check the line before for a table name
                    if i > 0 and not lines[i - 1].startswith("|"):
                        # Check if previous non-empty line is a heading or text
                        prev = lines[i - 1].strip()
                        prev_prev = lines[i - 2].strip() if i > 1 else ""
                        table_name = prev or prev_prev
                    in_table = True
                table_lines.append(stripped)
            else:
                if in_table and table_lines:
                    preview = "\n".join(table_lines[:5])
                    name = table_name or "未命名表格"
                    tables.append((name, preview))
                in_table = False
                table_lines = []
                table_name = ""

        # Last table if at end of document
        if in_table and table_lines:
            preview = "\n".join(table_lines[:5])
            name = table_name or "未命名表格"
            tables.append((name, preview))

        return tables
