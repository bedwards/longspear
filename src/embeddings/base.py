"""Abstract base for embedding providers.

Follows the Strategy pattern — swap implementations without
changing calling code. Factory function creates by model name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for this embedding model."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""


def create_embedding_provider(
    model_name: str, ollama_host: str
) -> EmbeddingProvider:
    """Factory: create an embedding provider by model name.

    Args:
        model_name: One of 'nomic-embed-text' or 'mxbai-embed-large'.
        ollama_host: Ollama server URL (e.g. http://ollama:11434).
    """
    from .nomic import NomicEmbedding
    from .mxbai import MxbaiEmbedding

    providers: dict[str, type[EmbeddingProvider]] = {
        "nomic-embed-text": NomicEmbedding,
        "mxbai-embed-large": MxbaiEmbedding,
    }

    if model_name not in providers:
        raise ValueError(
            f"Unknown embedding model '{model_name}'. "
            f"Available: {list(providers.keys())}"
        )

    return providers[model_name](ollama_host=ollama_host)
