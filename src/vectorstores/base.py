"""Abstract base for vector store backends.

Strategy pattern — swap pgvector/LanceDB without changing calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A text chunk with metadata and optional embedding vector."""

    content: str
    persona: str
    source_file: str = ""
    video_title: str = ""
    video_date: str = ""
    video_url: str = ""
    chunk_index: int = 0
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A ranked search result from a vector store."""

    document: Document
    score: float
    rank: int = 0


class VectorStore(ABC):
    """Abstract interface for vector store backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier for this backend."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up tables/collections. Idempotent."""

    @abstractmethod
    async def add_documents(
        self, documents: list[Document], embedding_model: str
    ) -> int:
        """Add documents with their embeddings. Returns count added."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        embedding_model: str,
        persona: str | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for similar documents by embedding vector."""

    @abstractmethod
    async def count(self, embedding_model: str, persona: str | None = None) -> int:
        """Count documents, optionally filtered by persona."""

    @abstractmethod
    async def delete_by_persona(
        self, persona: str, embedding_model: str
    ) -> int:
        """Delete all documents for a persona. Returns count deleted."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""


def create_vector_store(
    backend: str, **kwargs: Any
) -> VectorStore:
    """Factory: create a vector store by backend name.

    Args:
        backend: One of 'pgvector' or 'lancedb'.
        **kwargs: Backend-specific config (dsn, path, etc.).
    """
    from .pgvector_store import PgVectorStore
    from .lancedb_store import LanceDBStore

    stores: dict[str, type[VectorStore]] = {
        "pgvector": PgVectorStore,
        "lancedb": LanceDBStore,
    }

    if backend not in stores:
        raise ValueError(
            f"Unknown vector store '{backend}'. Available: {list(stores.keys())}"
        )

    return stores[backend](**kwargs)
