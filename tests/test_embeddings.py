"""Tests for embedding providers (structural tests — no Ollama needed)."""

from src.embeddings.base import create_embedding_provider
from src.embeddings.nomic import NomicEmbedding
from src.embeddings.mxbai import MxbaiEmbedding

import pytest


def test_create_nomic_provider():
    """Factory creates NomicEmbedding."""
    provider = create_embedding_provider(
        "nomic-embed-text", ollama_host="http://localhost:11434"
    )
    assert isinstance(provider, NomicEmbedding)
    assert provider.model_name == "nomic-embed-text"
    assert provider.dimensions == 768


def test_create_mxbai_provider():
    """Factory creates MxbaiEmbedding."""
    provider = create_embedding_provider(
        "mxbai-embed-large", ollama_host="http://localhost:11434"
    )
    assert isinstance(provider, MxbaiEmbedding)
    assert provider.model_name == "mxbai-embed-large"
    assert provider.dimensions == 1024


def test_create_unknown_provider():
    """Factory raises for unknown model."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        create_embedding_provider(
            "nonexistent-model", ollama_host="http://localhost:11434"
        )
