"""
Citation-enforced answer generation.

The prompt mandates that every factual claim includes a source citation
with filename, page number, and section path. This is enforced at generation
time, not post-hoc.
"""

import re
import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


CITATION_SYSTEM_PROMPT = """你是一个注重精确引用的知识库助手。你的回答必须严格遵循以下引用规则。

## 强制引用规则：
1. 每条事实声明必须标注引用来源，格式为：
   根据《文档名》第X页，[章节路径]："引用原文内容"

2. 如果你无法确定具体出处，必须明确说明：
   （无法确定具体出处，以下为根据文档内容推断）

3. 禁止编造页码、章节名或文件名。只能引用检索片段中 CITATION 字段列出的信息。

4. 如果一个结论由多个片段支持，引用所有相关片段。

5. 回答结构：
   - 回答正文
   - 每个关键点附带引用
   - 末尾的"参考文献"部分列出所有引用来源

6. 引用格式示例：
   根据《2023年度报告》第15页，第3章 > 3.2 收入分析："2023年公司总收入达到500亿元"

## 参考文献格式：
[1] 《文档名》第X页，[章节路径]
[2] 《文档名》第X页，[章节路径]

## 回答语言：
请使用与用户问题相同的语言回答（中文或英文）。"""


class CitationGenerator:
    """Generates answers with mandatory page/section citations."""

    # Regex to extract citations from generated answers
    CITATION_RE = re.compile(
        r"根据《(.+?)》第(\d+)页[，,]?\s*(.*?)[：:]\s*\"(.+?)\"",
    )

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CITATION_SYSTEM_PROMPT),
            ("human",
                "以下是从知识库中检索到的文档片段及其引用信息：\n\n"
                "{context}\n\n"
                "---\n"
                "用户问题：{question}\n\n"
                "请根据上述文档片段回答问题，严格遵循引用规则。"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, context: str) -> str:
        """Generate an answer with enforced citations.

        Args:
            question: The user's question.
            context: The retrieval agent's output (structured with CITATION fields).

        Returns:
            Answer string with inline citations and references section.
        """
        logger.info(
            f"[CitationGenerator] Generating answer for: {question[:80]}..."
        )

        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        logger.info(f"[CitationGenerator] Generated {len(answer)} chars")
        return answer

    @staticmethod
    def extract_citations(answer: str) -> list[dict]:
        """Parse the answer to extract all cited (filename, page_num, section, text) tuples.

        Returns list of dicts with keys: filename, page_num, section_path, quoted_text.
        """
        citations = []
        for m in CitationGenerator.CITATION_RE.finditer(answer):
            citations.append({
                "filename": m.group(1).strip(),
                "page_num": int(m.group(2)),
                "section_path": m.group(3).strip(),
                "quoted_text": m.group(4).strip(),
            })
        return citations

    @staticmethod
    def extract_references(answer: str) -> list[dict]:
        """Extract the numbered reference list at the end of the answer."""
        ref_pattern = re.compile(
            r"\[(\d+)\]\s*《(.+?)》第(\d+)页[，,]?\s*(.*?)(?:\n|$)"
        )
        refs = []
        for m in ref_pattern.finditer(answer):
            refs.append({
                "ref_num": int(m.group(1)),
                "filename": m.group(2).strip(),
                "page_num": int(m.group(3)),
                "section_path": m.group(4).strip().rstrip("。，,;；"),
            })
        return refs
