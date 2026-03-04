"""Conversation engine: connects RAG retrieval to LLM generation.

Takes a user question, retrieves relevant transcript context,
builds persona prompts, and generates a streaming response via Ollama.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import ollama as ollama_client

from ..config import get_settings, load_persona
from ..retrieval.context_builder import ContextBuilder
from ..retrieval.retriever import Retriever, RetrievalRequest

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ConversationContext:
    """Full context for a conversation turn."""

    persona_slug: str
    system_prompt: str
    user_prompt: str
    sources: list[dict[str, str]] = field(default_factory=list)
    embedding_used: str = ""
    vectorstore_used: str = ""
    chunks_retrieved: int = 0


class ConversationEngine:
    """Orchestrates RAG retrieval → LLM generation for persona conversations.

    Uses the existing Retriever and ContextBuilder, then streams
    the response through Ollama (native, Metal GPU).
    """

    def __init__(
        self,
        ollama_host: str | None = None,
        llm_model: str = "mistral-large:123b",
    ) -> None:
        settings = get_settings()
        self._ollama_host = ollama_host or settings.ollama_host
        self._llm_model = llm_model
        self._client = ollama_client.AsyncClient(host=self._ollama_host)
        self._retriever = Retriever()
        self._builder = ContextBuilder()

    async def prepare_context(
        self,
        question: str,
        persona_slug: str,
        embedding_model: str | None = None,
        vectorstore_backend: str | None = None,
        top_k: int | None = None,
        other_response: str | None = None,
    ) -> ConversationContext:
        """Retrieve context and build prompts without generating a response."""
        # Retrieve relevant transcript chunks
        retrieval = await self._retriever.retrieve(
            RetrievalRequest(
                question=question,
                persona=persona_slug,
                embedding_model=embedding_model,
                vectorstore_backend=vectorstore_backend,
                top_k=top_k,
            )
        )

        # Build prompts
        system_prompt = self._builder.build_system_prompt(
            persona_slug=persona_slug,
            retrieval=retrieval,
        )
        user_prompt = self._builder.build_debate_prompt(
            question=question,
            persona_slug=persona_slug,
            other_persona_response=other_response,
        )

        # Collect sources
        sources = []
        for r in retrieval.results:
            source: dict[str, str] = {}
            if r.document.video_title:
                source["title"] = r.document.video_title
            if r.document.video_date:
                source["date"] = r.document.video_date
            if r.document.video_url:
                source["url"] = r.document.video_url
            source["score"] = f"{r.score:.4f}"
            sources.append(source)

        return ConversationContext(
            persona_slug=persona_slug,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sources=sources,
            embedding_used=retrieval.embedding_model,
            vectorstore_used=retrieval.vectorstore_backend,
            chunks_retrieved=len(retrieval.results),
        )

    async def chat(
        self,
        question: str,
        persona_slug: str,
        history: list[ChatMessage] | None = None,
        embedding_model: str | None = None,
        vectorstore_backend: str | None = None,
        top_k: int | None = None,
        other_response: str | None = None,
    ) -> str:
        """Generate a complete (non-streaming) response."""
        ctx = await self.prepare_context(
            question=question,
            persona_slug=persona_slug,
            embedding_model=embedding_model,
            vectorstore_backend=vectorstore_backend,
            top_k=top_k,
            other_response=other_response,
        )

        messages = self._build_messages(ctx, history)

        response = await self._client.chat(
            model=self._llm_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        return response["message"]["content"]

    async def chat_stream(
        self,
        question: str,
        persona_slug: str,
        history: list[ChatMessage] | None = None,
        embedding_model: str | None = None,
        vectorstore_backend: str | None = None,
        top_k: int | None = None,
        other_response: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response, yielding tokens as they arrive."""
        ctx = await self.prepare_context(
            question=question,
            persona_slug=persona_slug,
            embedding_model=embedding_model,
            vectorstore_backend=vectorstore_backend,
            top_k=top_k,
            other_response=other_response,
        )

        messages = self._build_messages(ctx, history)

        stream = await self._client.chat(
            model=self._llm_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
        )

        async for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    def _build_messages(
        self,
        ctx: ConversationContext,
        history: list[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        """Build the full message list for the LLM."""
        messages = [ChatMessage(role="system", content=ctx.system_prompt)]

        # Add conversation history if present
        if history:
            messages.extend(history)

        # Add the current user turn
        messages.append(ChatMessage(role="user", content=ctx.user_prompt))

        return messages
