"""nomic-embed-text embedding provider via Ollama."""

from __future__ import annotations

from .ollama_base import OllamaEmbeddingBase


class NomicEmbedding(OllamaEmbeddingBase):
    """nomic-embed-text: 768-dim, fast general-purpose embeddings."""

    @property
    def model_name(self) -> str:
        return "nomic-embed-text"

    @property
    def dimensions(self) -> int:
        return 768
