"""Ollama-based embedding provider base.

Both nomic and mxbai use the same Ollama API — this base class
eliminates duplication (DRY). Subclasses only set model_name + dimensions.
"""

from __future__ import annotations

import logging
from typing import Any

import ollama as ollama_client

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingBase(EmbeddingProvider):
    """Base class for Ollama-hosted embedding models."""

    def __init__(self, ollama_host: str) -> None:
        self._client = ollama_client.AsyncClient(host=ollama_host)

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text via Ollama API."""
        response = await self._client.embed(
            model=self.model_name, input=text
        )
        return response["embeddings"][0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ollama API.

        Ollama's embed endpoint supports batch input natively.
        """
        if not texts:
            return []

        response = await self._client.embed(
            model=self.model_name, input=texts
        )
        return response["embeddings"]
