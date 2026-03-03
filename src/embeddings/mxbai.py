"""mxbai-embed-large embedding provider via Ollama."""

from __future__ import annotations

from .ollama_base import OllamaEmbeddingBase


class MxbaiEmbedding(OllamaEmbeddingBase):
    """mxbai-embed-large: 1024-dim, high-quality embeddings."""

    @property
    def model_name(self) -> str:
        return "mxbai-embed-large"

    @property
    def dimensions(self) -> int:
        return 1024
