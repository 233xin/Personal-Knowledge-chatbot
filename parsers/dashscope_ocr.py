"""
DashScope (Bailian) multimodal OCR parser for scanned/image-based PDFs.
Uses qwen-vl-ocr via OpenAI-compatible API. Each PDF page is rendered
to a PNG image and sent to the multimodal model for text extraction.
"""

import base64
import logging
from typing import Optional

import fitz  # pymupdf — renders PDF pages to images
from openai import OpenAI

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_OCR_MODEL,
    DASHSCOPE_VISION_MAX_TOKENS,
    DASHSCOPE_PAGE_IMAGE_DPI,
)
from models.schemas import ParsedDocument, PageBlock
from parsers.base import BaseParser

logger = logging.getLogger(__name__)

OCR_PROMPT = """请提取这张图片中的所有文字内容，按自然阅读顺序输出为Markdown格式。

要求：
1. 保留标题层级（使用 #、##、###）
2. 保留表格结构（使用 | ... | 格式）
3. 保留列表和编号
4. 每个章节标题后标注 [PAGE: 当前页]
5. 直接输出Markdown，不要添加任何额外解释"""


class DashScopeOCRParser(BaseParser):
    """Multimodal OCR via DashScope qwen-vl-ocr for scanned/image PDFs."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or DASHSCOPE_API_KEY
        self._model = model or DASHSCOPE_OCR_MODEL
        self._client: Optional[OpenAI] = None
        self._checked = False

    @property
    def name(self) -> str:
        return "dashscope_ocr"

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. Cannot perform OCR on scanned PDF."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=DASHSCOPE_BASE_URL,
            )

    def parse(self, file_path: str, filename: str) -> ParsedDocument:
        self._ensure_client()

        doc = fitz.open(file_path)
        pages = []
        full_markdown_parts = []

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            logger.info(
                f"[{self.name}] Processing page {page_num}/{len(doc)} of {filename}"
            )

            try:
                page_md = self._extract_page(doc[page_idx], page_num)
                pages.append(PageBlock(page_num=page_num, markdown=page_md))
                full_markdown_parts.append(page_md)
            except Exception as e:
                logger.warning(
                    f"[{self.name}] Page {page_num} OCR failed: {e}. "
                    f"Inserting placeholder."
                )
                placeholder = f"[PAGE {page_num}: OCR 提取失败 — {e}]"
                pages.append(PageBlock(page_num=page_num, markdown=placeholder))
                full_markdown_parts.append(placeholder)

        doc.close()

        full_markdown = "\n\n".join(full_markdown_parts)

        if not full_markdown.strip():
            raise ValueError(
                f"OCR could not extract any text from '{filename}'. "
                f"The file may be entirely image-based or corrupted."
            )

        return ParsedDocument(
            markdown=full_markdown,
            pages=pages,
            toc=[],
            parser_used=self.name,
        )

    def _extract_page(self, page, page_num: int) -> str:
        """Render one PDF page to PNG, send to qwen-vl-ocr, return markdown."""
        pix = page.get_pixmap(dpi=DASHSCOPE_PAGE_IMAGE_DPI)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": OCR_PROMPT,
                        },
                    ],
                }
            ],
            max_tokens=DASHSCOPE_VISION_MAX_TOKENS,
        )

        content = response.choices[0].message.content or ""
        # Ensure the page marker is present
        if f"[PAGE: {page_num}]" not in content:
            content = f"[PAGE: {page_num}]\n\n{content}"
        return content
