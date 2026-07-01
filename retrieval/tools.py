"""
Four retrieval tools exposed to the Agent for dynamic, intent-driven search.

The agent decides WHEN and WHICH tool to call — we never dump all
retrieval results into context at once.
"""

from typing import Optional

from langchain_core.tools import tool

from models.lineage import ChunkDocument


# The DocumentService singleton — injected at module load by app.py
_document_service = None


def set_document_service(svc) -> None:
    """Called during app startup to inject the DocumentService singleton."""
    global _document_service
    _document_service = svc


def _format_results(docs: list, source: str = "search") -> str:
    """Format retrieved chunks into a structured string with full lineage metadata.

    Each chunk includes its CITATION info so the agent and generation step
    have all evidence needed for page/section references.
    """
    if not docs:
        return f"[{source}] 未找到相关结果。"

    parts = []
    for i, doc in enumerate(docs):
        if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
            # LangChain Document
            citation = ChunkDocument.format_citation(doc)
            content = doc.page_content
        else:
            # BM25 result dict
            citation = ChunkDocument.make_citation_ref(
                filename=doc.get("metadata", {}).get("filename", "unknown"),
                page_num=doc.get("metadata", {}).get("page_num"),
                heading=doc.get("metadata", {}).get("heading_hierarchy", ""),
                table_id=doc.get("metadata", {}).get("table_id"),
                clause_num=doc.get("metadata", {}).get("clause_num"),
            )
            content = doc.get("content", "")

        parts.append(
            f"[结果 {i+1}] CITATION: {citation}\n"
            f"CONTENT:\n{content}\n"
        )

    return f"[{source}] 找到 {len(docs)} 条结果：\n\n" + "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 1: Semantic Search
# ---------------------------------------------------------------------------

@tool
def search_by_semantic(
    query: str,
    doc_ids: Optional[list[str]] = None,
    top_k: int = 5,
) -> str:
    """
    语义搜索工具。根据概念、含义和上下文相似度检索文档片段。

    适用场景：
    - 概念性问题："什么是..."、"解释..."、"如何..."
    - 含义相近的搜索
    - 需要理解上下文的问题

    返回每个结果附带的 CITATION 字段包含文件名、页码、章节路径和表格/条款编号，
    这些信息必须用于生成回答时的引用。
    """
    if _document_service is None:
        return "错误：文档服务未初始化。"

    docs = _document_service.search_semantic(
        query=query,
        top_k=top_k,
        doc_ids=doc_ids,
    )
    return _format_results(docs, source="语义搜索")


# ---------------------------------------------------------------------------
# Tool 2: Keyword Search (BM25)
# ---------------------------------------------------------------------------

@tool
def search_by_keyword(
    keywords: str,
    doc_ids: Optional[list[str]] = None,
    top_k: int = 5,
) -> str:
    """
    关键词精确搜索工具（BM25）。根据精确的词语匹配检索文档片段。

    适用场景：
    - 查找具体术语、数字、专有名词
    - 查找表格名称（如"表4.1"）
    - 查找条款编号（如"第17条"）
    - 精确匹配搜索

    返回每个结果附带的 CITATION 字段包含文件名、页码、章节路径和表格/条款编号。
    """
    if _document_service is None:
        return "错误：文档服务未初始化。"

    results = _document_service.search_keyword(
        query=keywords,
        top_k=top_k,
        doc_ids=doc_ids,
    )
    return _format_results(results, source="关键词搜索")


# ---------------------------------------------------------------------------
# Tool 3: Page-level Access
# ---------------------------------------------------------------------------

@tool
def get_chunks_by_page(doc_id: str, page_num: int) -> str:
    """
    按页码获取文档片段。返回指定文档中某一页的所有内容块。

    适用场景：
    - 用户指定了页码："第3页的内容是什么？"
    - 需要核实之前答案中引用的页码
    - 查看某一页的完整内容

    返回每个结果附带的 CITATION 字段。
    """
    if _document_service is None:
        return "错误：文档服务未初始化。"

    docs = _document_service.get_chunks_by_page(
        doc_id=doc_id,
        page_num=page_num,
    )
    return _format_results(docs, source=f"第{page_num}页")


# ---------------------------------------------------------------------------
# Tool 4: Section-level Access
# ---------------------------------------------------------------------------

@tool
def get_chunks_by_section(doc_id: str, section_id: str) -> str:
    """
    按章节获取文档片段。返回指定文档中某一章节的所有内容块。

    适用场景：
    - 用户指定了章节："第3章的内容"、"§2.1的内容"
    - 需要获取某一节的全部内容
    - 深入了解文档的某个特定部分

    section_id 格式举例："§3.2"、"第3章"、"Chapter 2"

    返回每个结果附带的 CITATION 字段。
    """
    if _document_service is None:
        return "错误：文档服务未初始化。"

    docs = _document_service.get_chunks_by_section(
        doc_id=doc_id,
        section_id=section_id,
    )
    return _format_results(docs, source=f"章节：{section_id}")
