"""Retrieval module: query vector stores, rank results.

Takes a natural-language question, embeds it, searches the
specified vector store, and returns ranked transcript chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import get_settings
from ..embeddings.base import create_embedding_provider
from ..vectorstores.base import SearchResult, create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalRequest:
    """A retrieval query."""

    question: str
    persona: str
    embedding_model: str | None = None  # default from config
    vectorstore_backend: str | None = None  # default from config
    top_k: int | None = None  # default from config


@dataclass
class RetrievalResponse:
    """Results from a retrieval query."""

    results: list[SearchResult]
    embedding_model: str
    vectorstore_backend: str
    question: str
    persona: str


class Retriever:
    """Queries vector stores for relevant transcript chunks."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Embed query and search the specified vector store."""
        settings = get_settings()

        # Resolve defaults
        model_name = request.embedding_model or settings.embedding.default
        backend_name = request.vectorstore_backend or settings.vectorstore.default
        top_k = request.top_k or settings.retrieval.top_k

        # Embed the query
        embedder = create_embedding_provider(
            model_name=model_name,
            ollama_host=settings.ollama_host,
        )
        query_embedding = await embedder.embed_text(request.question)

        # Search
        store_kwargs = {}
        if backend_name == "pgvector":
            store_kwargs["dsn"] = settings.postgres_dsn
        elif backend_name == "lancedb":
            lb_config = settings.vectorstore.backends.get("lancedb")
            store_kwargs["path"] = (
                lb_config.path if lb_config else "./data/vectordb/lancedb"
            )

        store = create_vector_store(backend_name, **store_kwargs)
        await store.initialize()

        results = await store.search(
            query_embedding=query_embedding,
            embedding_model=model_name,
            persona=request.persona,
            top_k=top_k,
        )

        await store.close()

        logger.info(
            "Retrieved %d results from %s/%s for persona %s",
            len(results), backend_name, model_name, request.persona,
        )

        return RetrievalResponse(
            results=results,
            embedding_model=model_name,
            vectorstore_backend=backend_name,
            question=request.question,
            persona=request.persona,
        )
