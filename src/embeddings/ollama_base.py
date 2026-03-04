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

# Max texts per batch to avoid overwhelming Ollama
_BATCH_SIZE = 64


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

        Splits into sub-batches to avoid memory issues and provide
        progress logging for large ingestion runs.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            batch_num = (i // _BATCH_SIZE) + 1
            total_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE

            logger.info(
                "Embedding batch %d/%d (%d texts) with %s...",
                batch_num, total_batches, len(batch), self.model_name,
            )

            response = await self._client.embed(
                model=self.model_name, input=batch
            )
            all_embeddings.extend(response["embeddings"])

        logger.info(
            "Embedded %d texts total with %s", total, self.model_name
        )
        return all_embeddings
