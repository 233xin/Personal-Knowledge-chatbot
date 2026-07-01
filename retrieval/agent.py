"""
RetrievalAgent — orchestrates tool-based dynamic retrieval.

The agent receives a user question, decides WHICH tool(s) to call
and in WHAT ORDER, then assembles a precise context. This replaces
the old "dump top-5 chunks into context" approach.

Key principle: the agent fetches ONLY what's needed, NEVER pre-fetches
everything from multiple paths and stuffs it into context.

Uses LangChain 1.x create_agent API.
"""

import logging
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI

from config import AGENT_MAX_ITERATIONS
from retrieval.tools import (
    search_by_semantic,
    search_by_keyword,
    get_chunks_by_page,
    get_chunks_by_section,
)

logger = logging.getLogger(__name__)


RETRIEVAL_AGENT_SYSTEM_PROMPT = """你是一个知识库检索助手。你的任务是根据用户的问题，选择合适的工具
从文档知识库中精准获取相关片段。

你有四个工具可以使用：
1. search_by_semantic — 语义搜索，用于概念性、含义类问题
2. search_by_keyword — 关键词搜索，用于精确术语、数字、专有名词查找
3. get_chunks_by_page — 按页码获取，用于用户指定页码时
4. get_chunks_by_section — 按章节获取，用于用户指定章节时

## 检索策略：
- 概念性问题（"什么是"、"解释"、"如何"）→ 优先用 search_by_semantic
- 具体术语/数字/名称 → 优先用 search_by_keyword
- 用户明确说了页码或章节 → 用 get_chunks_by_page 或 get_chunks_by_section
- 你可以使用多个工具进行多轮检索，但每次只调用需要的工具
- 禁止一次性把所有工具的结果堆砌在一起

## 重要：
- 当你收集到足够回答问题的信息后，整理并输出所有片段
- 每条片段必须保留其 CITATION 信息（文件名、页码、章节路径）
- 如果未找到相关信息，如实说明
- 不要在未找到信息时编造内容"""


class RetrievalAgent:
    """Agent-based retrieval with tool-calling capability.

    Uses LangChain 1.x create_agent + ToolCallLimitMiddleware
    to prevent infinite tool-calling loops.
    """

    def __init__(self, llm: ChatOpenAI):
        tools = [
            search_by_semantic,
            search_by_keyword,
            get_chunks_by_page,
            get_chunks_by_section,
        ]

        self._agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=RETRIEVAL_AGENT_SYSTEM_PROMPT,
            middleware=[
                ToolCallLimitMiddleware(run_limit=AGENT_MAX_ITERATIONS),
            ],
        )

    def retrieve(
        self,
        question: str,
        doc_ids: Optional[list[str]] = None,
    ) -> str:
        """
        Run the retrieval agent to gather relevant context.

        Returns a structured string with retrieved chunks and their
        lineage metadata, ready for citation generation.
        """
        input_text = question
        if doc_ids:
            input_text += (
                f"\n\n（注意：只搜索以下文档ID：{', '.join(doc_ids)}）"
            )

        logger.info(
            f"[RetrievalAgent] Starting retrieval for: {question[:80]}..."
        )

        # LangChain 1.x: invoke with {"messages": [...]} format
        result = self._agent.invoke({
            "messages": [{"role": "user", "content": input_text}],
        })

        # Extract the final AI message content
        messages = result.get("messages", [])
        output = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                output = msg.content
                break

        logger.info(
            f"[RetrievalAgent] Retrieval complete. "
            f"Output length: {len(output)} chars"
        )

        return output
